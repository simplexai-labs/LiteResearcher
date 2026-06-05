"""
简化版 Browse 服务
流程：先查 SQL → 爬虫 → 异步存 SQL → Summary
"""
import os
import re
import json
import threading
import glob # For log archiving
import shutil # For log archiving
import resource  # For FD limit

# ============ 提升 File Descriptor 限制 ============
# 独立进程启动时不会继承 shell 的 ulimit 设置，必须在代码中显式提升
def raise_fd_limit(target: int = 65535):
    """提升当前进程的 File Descriptor (RLIMIT_NOFILE) Soft Limit"""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(target, hard)  # 不能超过 hard limit
        if soft < new_soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            print(f"✅ FD Limit 已提升: {soft} → {new_soft} (hard={hard})")
        else:
            print(f"ℹ️  FD Limit 已满足: soft={soft}, hard={hard}")
    except Exception as e:
        print(f"⚠️  提升 FD Limit 失败: {e}（当前 soft={soft}, hard={hard}）")

# 在模块加载时立即提升（uvicorn fork 之前）
raise_fd_limit(65535)

# 自动加载 .env 文件
from dotenv import load_dotenv
load_dotenv()
import time
import asyncio
import logging
import urllib.parse
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests
import psycopg2
from psycopg2 import pool
import tiktoken
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI

# ============ 配置 ============
SCRAPEDO_API_KEY = os.environ.get("SCRAPEDO_API_KEY", "")
PROXY = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

DB_CONFIG = {
    "host": os.environ.get("PG_HOST", "47.111.147.142"),
    "port": int(os.environ.get("PG_PORT", 8432)),
    "user": os.environ.get("PG_USER", "postgres"),
    "password": os.environ.get("PG_PASSWORD", "pass123"),
    "database": os.environ.get("PG_DATABASE", "postgres"),
}
TABLE_NAME = "serper_wiki"

# Summary 配置
SUMMARY_API_BASE = os.environ.get("SUMMARY_API_BASE", "http://127.0.0.1:7001/v1")
SUMMARY_MODEL_NAME = os.environ.get("SUMMARY_MODEL_NAME", "")
SUMMARY_API_KEY = os.environ.get("SUMMARY_API_KEY", "EMPTY")

# ============ 各后端独立的并发配置 ============
# 总请求并发限制
TOTAL_MAX_CONCURRENT = int(os.environ.get("TOTAL_MAX_CONCURRENT", 2000))
# SQL 查询线程池大小
SQL_MAX_WORKERS = int(os.environ.get("SQL_MAX_WORKERS", 500))
# Summary LLM 并发限制（匹配 SGLang 的容量）
SUMMARY_MAX_CONCURRENT = int(os.environ.get("SUMMARY_MAX_CONCURRENT", 1000))

# 清除代理环境变量，防止 OpenAI 客户端走代理（爬虫使用代码中的 PROXY 变量）
for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(key, None)

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

# ============ 日志配置 ============
LOG_DIR = "log_browse"
os.makedirs(LOG_DIR, exist_ok=True)

# 归档旧日志文件（只在主进程执行）
def archive_old_logs():
    """将 log_browse 目录下的旧日志文件移动到归档文件夹"""
    # 查找所有 .log 文件，排除 archive_ 开头的文件夹
    log_files = [f for f in glob.glob(os.path.join(LOG_DIR, "*.log")) if not os.path.basename(f).startswith("archive_")]
    
    if log_files:
        # 创建归档文件夹（使用时间戳命名）
        archive_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_dir = os.path.join(LOG_DIR, f"archive_{archive_timestamp}")
        os.makedirs(archive_dir, exist_ok=True)
        
        # 移动所有旧日志文件
        moved_count = 0
        for log_file_path in log_files:
            try:
                filename = os.path.basename(log_file_path)
                dest_path = os.path.join(archive_dir, filename)
                shutil.move(log_file_path, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"⚠️  移动日志文件失败: {log_file_path} - {e}")
        
        if moved_count > 0:
            print(f"📦 已归档 {moved_count} 个旧日志文件到: {archive_dir}")

# 执行归档操作
archive_old_logs()
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

# 普通后端日志文件
log_file = os.path.join(LOG_DIR, f"{timestamp_str}.log")
# 统计信息专用日志文件
stats_log_file = os.path.join(LOG_DIR, f"{timestamp_str}_stats.log")

# 日志格式
log_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# 主 logger（后端日志）
logger = logging.getLogger("browse_service")
logger.setLevel(logging.INFO)
logger.propagate = False  # 防止重复输出

# 后端日志 handlers：文件 + 控制台
backend_file_handler = logging.FileHandler(log_file)
backend_file_handler.setFormatter(log_format)
backend_console_handler = logging.StreamHandler()
backend_console_handler.setFormatter(log_format)
logger.addHandler(backend_file_handler)
logger.addHandler(backend_console_handler)

# 统计 logger（独立文件，不输出到控制台，避免淹没）
stats_logger = logging.getLogger("browse_stats")
stats_logger.setLevel(logging.INFO)
stats_logger.propagate = False

# 统计日志 handlers：专用文件 + 控制台（简洁格式）
stats_file_handler = logging.FileHandler(stats_log_file)
stats_file_handler.setFormatter(log_format)
stats_console_handler = logging.StreamHandler()
stats_console_handler.setFormatter(logging.Formatter("%(asctime)s | 📊 STATS | %(message)s"))
stats_logger.addHandler(stats_file_handler)
stats_logger.addHandler(stats_console_handler)

logger.info(f"📝 后端日志文件: {log_file}")
logger.info(f"📊 统计日志文件: {stats_log_file}")

app = FastAPI(title="Browse Service")

# ============ 各后端独立的线程池 ============
sql_executor = ThreadPoolExecutor(max_workers=SQL_MAX_WORKERS)
logger.info(f"🔧 线程池配置 | SQL Workers: {SQL_MAX_WORKERS}")

# 统计日志输出间隔（秒）
STATS_LOG_INTERVAL = int(os.environ.get("STATS_LOG_INTERVAL", 1))

# ============ 请求统计类 ============
class RequestStats:
    """线程安全的请求统计（3个维度：总请求、SQL、Summary）"""
    
    def __init__(self):
        self._lock = threading.Lock()
        # 总体统计（外层请求）
        self.total_active = 0      # 正在处理的总请求数
        self.total_queued = 0      # 排队中的总请求数
        self.total_completed = 0   # 已完成的总请求数
        # SQL 查询统计
        self.sql_active = 0        # 正在执行的 SQL 查询
        self.sql_queued = 0        # 排队中的 SQL 查询
        self.sql_completed = 0     # 已完成的 SQL 查询
        # Summary 统计
        self.summary_active = 0    # 正在执行的 Summary 任务
        self.summary_queued = 0    # 排队中的 Summary 任务（等待信号量）
        self.summary_completed = 0 # 已完成的 Summary 任务
        # 后台任务
        self._running = False
        self._task = None
    
    def log_stats(self, prefix: str = ""):
        """输出当前统计信息（不加锁版本，由调用方加锁）"""
        stats_logger.info(
            f"{prefix}| "
            f"Total[running={self.total_active}, queued={self.total_queued}, done={self.total_completed}] | "
            f"SQL[running={self.sql_active}, queued={self.sql_queued}, done={self.sql_completed}] | "
            f"Summary[running={self.summary_active}, queued={self.summary_queued}, done={self.summary_completed}]"
        )
    
    def log_stats_safe(self, prefix: str = ""):
        """输出当前统计信息（线程安全版本）"""
        with self._lock:
            self.log_stats(prefix)
    
    async def _stats_logger_task(self):
        """后台定时输出统计信息"""
        stats_logger.info(f"统计日志后台任务启动 | 间隔: {STATS_LOG_INTERVAL}s")
        while self._running:
            await asyncio.sleep(STATS_LOG_INTERVAL)
            if self._running:
                self.log_stats_safe("REALTIME ")
    
    def start_background_logger(self):
        """启动后台统计日志任务"""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._stats_logger_task())
            stats_logger.info("后台统计日志任务已启动")
    
    def stop_background_logger(self):
        """停止后台统计日志任务"""
        self._running = False
        if self._task:
            self._task.cancel()
            stats_logger.info("后台统计日志任务已停止")
    
    # 总体请求统计
    def request_queue(self):
        with self._lock:
            self.total_queued += 1
    
    def request_start(self):
        with self._lock:
            self.total_queued -= 1
            self.total_active += 1
    
    def request_end(self):
        with self._lock:
            self.total_active -= 1
            self.total_completed += 1
    
    # SQL 统计
    def sql_queue(self):
        with self._lock:
            self.sql_queued += 1
    
    def sql_start(self):
        with self._lock:
            self.sql_queued -= 1
            self.sql_active += 1
    
    def sql_end(self):
        with self._lock:
            self.sql_active -= 1
            self.sql_completed += 1
    
    # Summary 统计
    def summary_queue(self):
        with self._lock:
            self.summary_queued += 1
    
    def summary_start(self):
        with self._lock:
            self.summary_queued -= 1
            self.summary_active += 1
    
    def summary_end(self):
        with self._lock:
            self.summary_active -= 1
            self.summary_completed += 1
    
    def get_stats_dict(self):
        """返回统计字典"""
        with self._lock:
            return {
                "total": {
                    "running": self.total_active,
                    "queued": self.total_queued,
                    "max_concurrent": TOTAL_MAX_CONCURRENT
                },
                "sql": {
                    "running": self.sql_active,
                    "queued": self.sql_queued,
                    "max_workers": SQL_MAX_WORKERS
                },
                "summary": {
                    "running": self.summary_active,
                    "queued": self.summary_queued,
                    "max_concurrent": SUMMARY_MAX_CONCURRENT
                }
            }

# 全局统计实例
request_stats = RequestStats()

# FastAPI 启动/关闭事件
@app.on_event("startup")
async def startup_event():
    """服务启动时启动后台统计日志"""
    request_stats.start_background_logger()

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时停止后台统计日志"""
    request_stats.stop_background_logger()
    logger.info("✅ 服务关闭")

# ============ HTML转Markdown ============
try:
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md  # 导入 markdownify
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logger.warning("⚠️  未安装 BeautifulSoup 或 markdownify，HTML 转 Markdown 功能受限。请运行: pip install beautifulsoup4 markdownify")

def html_to_markdown(raw_html: str) -> str:
    """HTML 转 Markdown"""
    if not raw_html:
        return ""
    
    if HAS_BS4:
        try:
            # 使用 markdownify 库，可以保留更多格式
            markdown = md(raw_html, heading_style="ATX", strong_em_symbol="**")
            # 清理连续空行
            markdown = re.sub(r'\n\s*\n', '\n\n', markdown).strip()
            return markdown
        except Exception as e:
            logger.warning(f"⚠️  markdownify 转换失败: {e}，回退到 BeautifulSoup")
            try:
                soup = BeautifulSoup(raw_html, "html.parser")
                for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                return "\n".join(line.strip() for line in text.splitlines() if line.strip())
            except Exception as e:
                logger.warning(f"⚠️  BeautifulSoup 转换失败: {e}")
    
    # 回退方案 (最原始的正则替换)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.S|re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", raw_html, flags=re.S|re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

# ============ 数据库连接池配置 ============
DB_POOL_MIN_CONN = int(os.environ.get("DB_POOL_MIN_CONN", 10))
DB_POOL_MAX_CONN = int(os.environ.get("DB_POOL_MAX_CONN", 200))

# ============ 爬虫管理器类 ============
class CrawlerManager:
    """负责 SQL 缓存和记录未命中 URL"""
    
    def __init__(self):
        # 初始化连接池
        self.connection_pool = pool.ThreadedConnectionPool(
            minconn=DB_POOL_MIN_CONN,
            maxconn=DB_POOL_MAX_CONN,
            **DB_CONFIG
        )
        logger.info(f"🔧 数据库连接池初始化 | Min: {DB_POOL_MIN_CONN} | Max: {DB_POOL_MAX_CONN}")
        
        self.init_table()
        
        # ============ 重构：使用单个追加文件代替每请求一个 JSON 文件 ============
        # 旧方案：每次 cache miss 创建一个独立 JSON 文件 → 高并发下瞬间耗尽 FD
        # 新方案：使用一个全局 logging.FileHandler 追加写入，线程安全且 FD 固定为 1
        base_dir = os.path.join(os.path.dirname(__file__), "cache_miss_urls")
        os.makedirs(base_dir, exist_ok=True)
        miss_log_file = os.path.join(base_dir, f"cache_miss_{timestamp_str}.jsonl")
        
        # 创建专用的 cache miss logger（线程安全，单个 FileHandler）
        self._miss_logger = logging.getLogger("cache_miss")
        self._miss_logger.setLevel(logging.INFO)
        self._miss_logger.propagate = False  # 不传播到 root logger
        # 清除旧 handler（防止重复添加）
        self._miss_logger.handlers.clear()
        miss_handler = logging.FileHandler(miss_log_file, mode='a', encoding='utf-8')
        miss_handler.setFormatter(logging.Formatter('%(message)s'))  # 纯 JSON 行
        self._miss_logger.addHandler(miss_handler)
        
        logger.info(f"📝 未命中 URL 记录文件: {miss_log_file}")
    
    def log_cache_miss(self, url: str):
        """记录缓存未命中的 URL（线程安全，无需创建新线程/文件）
        
        使用 logging.FileHandler 追加写入单个 JSONL 文件，
        logging 模块内部有锁保证线程安全，FD 固定为 1。
        """
        try:
            record = json.dumps(
                {"url": url, "time": datetime.now().isoformat()},
                ensure_ascii=False
            )
            self._miss_logger.info(record)
        except Exception as e:
            logger.error(f"❌ 记录未命中 URL 失败: {e}")
    
    def get_db_conn(self):
        """从连接池获取数据库连接"""
        return self.connection_pool.getconn()
    
    def put_db_conn(self, conn):
        """归还连接到连接池"""
        self.connection_pool.putconn(conn)
    
    def init_table(self):
        """初始化表"""
        conn = self.get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                        id SERIAL PRIMARY KEY,
                        url VARCHAR(2048) NOT NULL UNIQUE,
                        markdown TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_url ON {TABLE_NAME}(url);
                """)
            conn.commit()
            logger.info(f"✅ 表 {TABLE_NAME} 初始化完成")
        finally:
            self.put_db_conn(conn)
    
    def get_markdown_sync(self, url: str) -> Optional[str]:
        """从数据库查询 markdown（同步，带统计）"""
        request_stats.sql_start()
        conn = None
        try:
            conn = self.get_db_conn()
            with conn.cursor() as cur:
                cur.execute(f"SELECT markdown FROM {TABLE_NAME} WHERE url = %s", (url,))
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            if conn:
                self.put_db_conn(conn)
            request_stats.sql_end()
    
    async def get_markdown(self, url: str) -> Optional[str]:
        """从数据库查询 markdown（异步，带统计）"""
        request_stats.sql_queue()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(sql_executor, self.get_markdown_sync, url)
    
    def save_markdown_sync(self, url: str, markdown: str):
        """同步保存到数据库"""
        conn = None
        try:
            conn = self.get_db_conn()
            # 清理NUL字符（\x00），PostgreSQL不支持在TEXT字段中存储NUL
            cleaned_markdown = markdown.replace('\x00', '') if markdown else ''
            
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {TABLE_NAME} (url, markdown, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (url) 
                    DO UPDATE SET 
                        markdown = EXCLUDED.markdown,
                        updated_at = CURRENT_TIMESTAMP
                """, (url, cleaned_markdown))
            conn.commit()
            md_len = len(cleaned_markdown)
            logger.info(f"💾 已存储 | URL: {url[:80]}... | 长度: {md_len}")
        except Exception as e:
            logger.error(f"❌ 存储失败 | {e}")
        finally:
            if conn:
                self.put_db_conn(conn)
    
    async def save_markdown_async(self, url: str, markdown: str):
        """异步保存到数据库"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(sql_executor, self.save_markdown_sync, url, markdown)
    
    def get_stats(self):
        """获取统计信息"""
        conn = None
        try:
            conn = self.get_db_conn()
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                count = cur.fetchone()[0]
            return {
                "total_urls": count,
                "table": TABLE_NAME
            }
        finally:
            if conn:
                self.put_db_conn(conn)

# ============ Token 截断工具 ============
def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """使用 tiktoken 截断文本到指定 token 数量"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = encoding.decode(tokens[:max_tokens])
    token_count = len(tokens)
    logger.warning(f"⚠️  内容截断 | {token_count} → {max_tokens} tokens")
    return truncated

# ============ Summary 服务类 ============
class SummaryService:
    """负责调用 LLM 进行摘要"""
    
    def __init__(self, api_base: str, model_name: str, api_key: str = "EMPTY"):
        self.api_base = api_base
        self.model_name = model_name
        self.api_key = api_key
        
        # 配置 httpx 连接池，与 SUMMARY_MAX_CONCURRENT 保持一致
        import httpx
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=SUMMARY_MAX_CONCURRENT,  # 最大连接数 = 并发数
                max_keepalive_connections=SUMMARY_MAX_CONCURRENT,  # 保持连接数 = 并发数
            ),
            timeout=httpx.Timeout(300.0, connect=10.0)  # 连接超时10秒，总超时300秒
        )
        
        self.client = AsyncOpenAI(
            api_key=api_key, 
            base_url=api_base,
            http_client=http_client
        )
        # 信号量限制并发请求数
        self.semaphore = asyncio.Semaphore(SUMMARY_MAX_CONCURRENT)
        logger.info(f"✅ Summary 服务初始化 | API: {api_base} | Model: {model_name} | 最大并发: {SUMMARY_MAX_CONCURRENT} | 连接池: {SUMMARY_MAX_CONCURRENT}")
    
    def clean_llm_response(self, content: str) -> str:
        """清理 LLM 响应"""
        if not content:
            return ""
        
        # 移除 <think> 标签
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # 提取 <answer> 标签内容
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if answer_match:
            content = answer_match.group(1).strip()
        
        return content
    
    async def summarize(self, content: str, goal: str, max_retries: int = 1) -> tuple[dict, str]:
        """调用 LLM 进行摘要（带并发限制和统计）
        
        Returns:
            tuple[dict, str]: (result_dict, error_message)
            - 成功: ({"evidence": ..., "summary": ...}, "")
            - 失败: ({"evidence": ..., "summary": ...}, "错误信息")
        """
        # 截断内容到 60k tokens（参考 browser_server.py 的 60k）
        MAX_CONTENT_TOKENS = 32000
        content = truncate_to_tokens(content, MAX_CONTENT_TOKENS)
        
        prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
        
        # 统计：进入排队（在获取信号量之前）
        request_stats.summary_queue()
        
        # 使用信号量限制并发
        async with self.semaphore:
            # 统计：开始执行（已获取信号量）
            request_stats.summary_start()
            
            try:
                last_error = ""
                for attempt in range(max_retries):
                    logger.info(f"🤖 Summary | 尝试 {attempt+1}/{max_retries}")
                    try:
                        resp = await self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=6000,
                            timeout=300,
                        )
                        raw = resp.choices[0].message.content or ""
                        
                        # 清理响应
                        raw = self.clean_llm_response(raw)
                        raw = raw.replace("```json", "").replace("```", "").strip()
                        
                        # 提取 JSON
                        left, right = raw.find("{"), raw.rfind("}")
                        if left != -1 and right != -1 and left <= right:
                            raw = raw[left:right + 1]
                        
                        result = json.loads(raw)
                        
                        evidence_len = len(str(result.get('evidence', '')))
                        summary_len = len(str(result.get('summary', '')))
                        logger.info(f"✅ Summary | 成功 | Evidence: {evidence_len} | Summary: {summary_len}")
                        
                        # 成功时返回结果，finally 会自动执行 summary_end()
                        return result, ""
                        
                    except json.JSONDecodeError as e:
                        last_error = f"JSON parse error: {str(e)[:100]}"
                        logger.error(f"❌ Summary | JSON解析失败 | 尝试 {attempt + 1}/{max_retries}")
                    except Exception as e:
                        last_error = f"Summary request error: {str(e)[:100]}"
                        logger.error(f"❌ Summary | 请求失败 | 尝试 {attempt + 1}/{max_retries} | {str(e)[:200]}")
                
                # 所有重试失败
                logger.error(f"❌ Summary | 所有重试失败")
                return {
                    "evidence": "The provided webpage content could not be accessed. Please check the URL or file format.",
                    "summary": "The webpage content could not be processed, and therefore, no information is available."
                }, f"Summary failed: {last_error}"
                
            finally:
                # 统计：执行结束（无论成功/失败/异常，都会执行）
                # 这里会减少 active 计数，增加 completed 计数
                # 同时信号量也会在 async with 退出时自动释放
                request_stats.summary_end()

# ============ API 端点 ============
class BrowseRequest(BaseModel):
    url: str
    goal: str = None  # 新格式
    question: str = None  # 旧格式，兼容 browse_utils.py

class BrowseResponse(BaseModel):
    success: bool
    result: str
    from_cache: bool
    error: str = ""

# 全局实例
crawler_manager = CrawlerManager()
summary_service = SummaryService(
    api_base=SUMMARY_API_BASE,
    model_name=SUMMARY_MODEL_NAME,
    api_key=SUMMARY_API_KEY
)

# 总请求并发信号量
total_semaphore = asyncio.Semaphore(TOTAL_MAX_CONCURRENT)
logger.info(f"🔧 总请求并发限制 | Max: {TOTAL_MAX_CONCURRENT}")

@app.post("/query", response_model=BrowseResponse)
async def browse(req: BrowseRequest):
    """
    主逻辑：
    1. 先查 SQL (markdown)
    2. 如果没有，爬取并异步存入 SQL
    3. 调用 Summary 服务
    4. 返回格式化结果
    
    支持两种参数格式：
    - 新格式：url + goal
    - 旧格式：url + question (兼容 browse_utils.py)
    """
    url = req.url
    # 兼容旧格式 (question) 和新格式 (goal)
    goal = req.goal if req.goal else req.question
    
    if not goal:
        return BrowseResponse(
            success=False,
            result="",
            from_cache=False,
            error="Missing 'goal' or 'question' parameter"
        )
    
    # 统计：进入排队
    request_stats.request_queue()
    
    # 使用信号量控制总并发
    async with total_semaphore:
        # 统计：开始处理
        request_stats.request_start()
        start = time.time()
        
        try:
            # 步骤1: 查询数据库
            logger.info(f"🔍 查询 SQL | {url[:80]}...")
            cached_markdown = await crawler_manager.get_markdown(url)
            
            from_cache = bool(cached_markdown)
            
            if cached_markdown:
                elapsed = time.time() - start
                md_len = len(cached_markdown)
                logger.info(f"💾 命中缓存 | {elapsed:.2f}s | {md_len} 字符")
                markdown = cached_markdown
            else:
                # 步骤2: 未命中缓存，记录 URL 并返回错误
                logger.warning(f"❌ 缓存未命中 | {url[:80]}...")
                crawler_manager.log_cache_miss(url)
                
                # 返回缓存未命中错误
                result = f"The useful information in {url} for user goal {goal} as follows: \n\n"
                result += "Evidence in page: \nThe provided webpage content could not be accessed. Please check the URL or file format.\n\n"
                result += "Summary: \nThe webpage content could not be processed, and therefore, no information is available.\n\n"
                
                # 统计：总请求结束（缓存未命中）
                request_stats.request_end()
                
                return BrowseResponse(
                    success=False,
                    result=result,
                    from_cache=False,
                    error="Cache miss: URL not in database"
                )
            
            # 步骤3: Summary
            parsed, summary_error = await summary_service.summarize(markdown, goal)
            
            evidence = str(parsed.get("evidence", "N/A"))
            summary = str(parsed.get("summary", "N/A"))
            
            # 步骤4: 格式化返回
            result = f"The useful information in {url} for user goal {goal} as follows: \n\n"
            result += "Evidence in page: \n" + evidence + "\n\n"
            result += "Summary: \n" + summary + "\n\n"
            
            elapsed = time.time() - start
            result_len = len(result)
            
            # 统计：总请求结束
            request_stats.request_end()
            
            # 如果 Summary 失败，返回 success=False
            if summary_error:
                logger.error(f"❌ Summary失败 | {elapsed:.2f}s | 缓存: {from_cache} | {summary_error}")
                return BrowseResponse(
                    success=False,
                    result=result.strip(),
                    from_cache=from_cache,
                    error=summary_error
                )
            
            logger.info(f"✅ 完成 | {elapsed:.2f}s | 缓存: {from_cache} | 结果: {result_len} 字符")
            return BrowseResponse(
                success=True,
                result=result.strip(),
                from_cache=from_cache
            )
            
        except asyncio.CancelledError:
            elapsed = time.time() - start
            logger.warning(f"⚠️ 请求被取消 | {elapsed:.2f}s | URL: {url[:60]}...")
            
            # 统计：总请求结束（取消）
            request_stats.request_end()
            
            return BrowseResponse(
                success=False,
                result="",
                from_cache=False,
                error="Request cancelled (server overloaded)"
            )
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"❌ 异常 | {elapsed:.2f}s | {type(e).__name__}: {str(e)}")
            
            # 统计：总请求结束（异常）
            request_stats.request_end()
            
            return BrowseResponse(
                success=False,
                result="",
                from_cache=False,
                error=f"{type(e).__name__}: {str(e)}"
            )

@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok", "service": "browse"}

@app.get("/stats")
def stats():
    """统计信息（包含请求统计和数据库统计）"""
    db_stats = crawler_manager.get_stats()
    req_stats = request_stats.get_stats_dict()
    return {
        "database": db_stats,
        "requests": req_stats
    }

# ============ 启动 ============
if __name__ == "__main__":
    import uvicorn
    
    # 再次确认 FD Limit（__main__ 入口保险）
    raise_fd_limit(65535)
    
    # 打印当前 FD 状态
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    logger.info(f"📊 当前 FD Limit: soft={soft}, hard={hard}")
    
    port = int(os.environ.get("BROWSE_PORT", 8010))
    limit_concurrency = int(os.environ.get("BROWSE_LIMIT_CONCURRENCY", 3000))
    backlog = int(os.environ.get("BROWSE_BACKLOG", 4096))
    
    logger.info(f"🚀 启动 Browse 服务 | 端口: {port} | 最大并发: {limit_concurrency} | Backlog: {backlog}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        limit_concurrency=limit_concurrency,
        backlog=backlog,
        timeout_keep_alive=60,
        limit_max_requests=None,
    )


