# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
# 注意：不要使用 pickle.FALSE，它是 bytes 类型，bool 值为 True！
import random
import re
import resource
import threading
from contextlib import ExitStack
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor

from verl.tools.utils.google_search_utils import perform_single_google_search_batch

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse


def _raise_fd_limit(target: int = 65535):
    """提升当前进程的 RLIMIT_NOFILE soft limit（适用于 Ray Worker 进程）"""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(target, hard)
        if soft < new_soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except Exception:
        pass  # Ray Worker 中静默处理，避免日志洪泛


# 模块加载时立即提升（覆盖 import 该模块的所有进程）
_raise_fd_limit(65535)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ============ URL Mask 日志（使用 logging.FileHandler 避免频繁 open/close FD） ============
URL_MASK_LOG_DIR = "/share/project/wanli/Search_Agent/verl/url_mask"
os.makedirs(URL_MASK_LOG_DIR, exist_ok=True)

_SESSION_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
_SESSION_LOG_FILE = os.path.join(URL_MASK_LOG_DIR, f"{_SESSION_TIMESTAMP}.log")

# 使用 logging 模块管理文件句柄（线程安全，FD 固定为 1）
_url_mask_logger = logging.getLogger("url_mask_logger")
_url_mask_logger.setLevel(logging.INFO)
_url_mask_logger.propagate = False
_url_mask_logger.handlers.clear()
_url_mask_handler = logging.FileHandler(_SESSION_LOG_FILE, mode='a', encoding='utf-8')
_url_mask_handler.setFormatter(logging.Formatter('%(message)s'))
_url_mask_logger.addHandler(_url_mask_handler)

# 模块加载时写入日志
_url_mask_logger.info(f"{datetime.now().strftime('%H:%M:%S')} | MODULE_LOADED | google_search_tool.py loaded by process {os.getpid()}")

# 线程锁保护写入（logging 已线程安全，此锁为额外保险）
_url_mask_lock = threading.Lock()

def log_url_mask(query: str, position: int, url: str):
    """记录 URL mask 事件到日志文件（线程安全，无需频繁 open/close）"""
    line = f"{datetime.now().strftime('%H:%M:%S')} | query=\"{query[:50]}\" | 第{position}个结果被mask | url={url}"
    _url_mask_logger.info(line)

T = TypeVar("T")


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """全局 token bucket worker for google search - 使用 threading.Semaphore 控制并发"""
    
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)
        logger.info(f"TokenBucketWorker initialized with rate_limit={rate_limit}")

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """同步获取 token"""
        self._semaphore.acquire()
        self.current_count += 1
        logger.debug(f"Token acquired, current count: {self.current_count}")

    @ray.method(concurrency_group="release")
    def release(self):
        """同步释放 token"""
        self._semaphore.release()
        self.current_count -= 1
        logger.debug(f"Token released, current count: {self.current_count}")

    def ping(self):
        return True


@ray.remote
class GoogleSearchExecutionWorker:
    """执行器：负责实际执行 google search 调用（同步方式，参考 search_tool.py）"""
    
    ACQUIRE_TIMEOUT = 300  # 获取 token 的超时时间（秒）- 从180增加到300秒(5分钟)
    
    def __init__(self, rate_limit_worker):
        # Ray Worker 进程不继承 shell 的 ulimit，必须在 __init__ 中提升
        _raise_fd_limit(65535)
        self.rate_limit_worker = rate_limit_worker

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """执行函数，使用 token bucket 控制并发（同步方式）"""
        from contextlib import ExitStack
        
        if self.rate_limit_worker:
            with ExitStack() as stack:
                # 确保 release 在退出时被调用
                stack.callback(self.rate_limit_worker.release.remote)
                # 获取 token（带超时）
                try:
                    ray.get(self.rate_limit_worker.acquire.remote(), timeout=self.ACQUIRE_TIMEOUT)
                except ray.exceptions.GetTimeoutError:
                    logger.error(f"Failed to acquire token within {self.ACQUIRE_TIMEOUT}s")
                    raise TimeoutError(f"Token acquisition timeout after {self.ACQUIRE_TIMEOUT}s")
                
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing search engine: {e}")
                    raise
        else:
            return fn(*fn_args, **fn_kwargs)


def init_google_search_token_bucket(rate_limit: int):
    """初始化全局唯一的 TokenBucketWorker（使用 Ray Named Actor）
    
    使用 get_if_exists=True 避免并发创建冲突（参考 search_tool.py）
    """
    actor_name = "google_search_token_bucket_singleton"
    return TokenBucketWorker.options(
        name=actor_name, 
        get_if_exists=True
    ).remote(rate_limit)


def init_google_search_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode):
    """初始化执行池和 token bucket（参考 search_tool.py）"""
    if mode == PoolMode.ThreadMode:
        # 创建全局唯一的 token bucket
        rate_limit_worker = init_google_search_token_bucket(rate_limit)
        
        # 创建全局唯一的 execution worker（使用 get_if_exists=True）
        actor_name = "google_search_execution_pool_singleton"
        return GoogleSearchExecutionWorker.options(
            name=actor_name,
            get_if_exists=True,
            max_concurrency=num_workers
        ).remote(rate_limit_worker)
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class GoogleSearchTool(BaseTool):
    """Search tool using RAG backend. Compatible with search_tool.py format."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)
        self.default_limit = config.get("default_limit", 10)
        self.enable_dedup = config.get("enable_dedup", True)  # 是否启用去重

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", False)
        self.execution_pool = init_google_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        self.search_service_url = config.get("search_service_url")
        assert self.search_service_url, "Configuration must include 'search_service_url'"

        logger.info(f"Initialized GoogleSearchTool with config: {config}")

    def _mask_url_in_search_result(self, search_result: str, masked_url: str, query: str = "") -> str:
        """Mask URL from search results."""
        # 调试：记录 mask_url 检查情况
        if masked_url:
            url_found = masked_url in search_result if isinstance(search_result, str) else False
            _url_mask_logger.info(f"{datetime.now().strftime('%H:%M:%S')} | MASK_CHECK | query=\"{query[:50]}\" | mask_url=\"{masked_url[:80]}\" | found_in_result={url_found}")
        
        if not masked_url or not isinstance(search_result, str) or masked_url not in search_result:
            return search_result

        result = search_result
        while masked_url in result:
            url_pos = result.find(masked_url)
            position = result[:url_pos].count('\n\n') // 2 + 1
            
            # 记录日志
            log_url_mask(query, position, masked_url)
            
            # 移除包含该 URL 的结果块
            start_pos = result.rfind('\n\n', 0, url_pos)
            start_pos = start_pos + 2 if start_pos != -1 else 0
            
            first_nn_after_url = result.find('\n\n', url_pos)
            if first_nn_after_url == -1:
                result = result[:start_pos]
                break
            
            search_from = first_nn_after_url + 2
            next_nn = result.find('\n\n', search_from)
            separator_pos = result.find('\n=======\n', search_from)
            
            if separator_pos != -1 and (next_nn == -1 or separator_pos < next_nn):
                end_pos = separator_pos
            elif next_nn != -1:
                end_pos = next_nn
            else:
                end_pos = len(result)
            
            result = result[:start_pos] + result[end_pos:]
        
        return result

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def _parse_search_result_block(self, block: str) -> list[dict]:
        """解析单个 query 的搜索结果块，提取每个结果的 title, link, snippet
        
        格式: {idx}. [{title}]({link})\n{snippet}\n\n
        """
        results = []
        # 匹配格式: 数字. [标题](链接)\n内容
        pattern = r'(\d+)\.\s*\[([^\]]*)\]\(([^)]+)\)\n(.*?)(?=\n\d+\.\s*\[|\Z)'
        matches = re.findall(pattern, block, re.DOTALL)
        
        for match in matches:
            idx, title, link, snippet = match
            results.append({
                'title': title.strip(),
                'link': link.strip(),
                'snippet': snippet.strip()
            })
        
        return results

    def _deduplicate_per_query(self, results_by_query: list[tuple[str, list[dict]]], masked_url: str = "") -> tuple[list[tuple[str, list[dict]]], int, int]:
        """对每个 query 的结果进行去重（相对于前面所有 query 的结果）
        
        Args:
            results_by_query: [(query1, [results1...]), (query2, [results2...]), ...]
            masked_url: 需要 mask 的 URL
        
        Returns:
            (去重后的结果列表, 原始总数量, 去重后总数量)
        """
        seen_links = set()
        deduped_by_query = []
        original_total = 0
        deduped_total = 0
        
        for query, results in results_by_query:
            original_total += len(results)
            deduped_results = []
            
            for idx, result in enumerate(results):
                link = result.get('link', '')
                
                # 跳过 masked URL
                if masked_url and masked_url in link:
                    # 记录 mask 日志（去重路径）
                    _url_mask_logger.info(f"{datetime.now().strftime('%H:%M:%S')} | MASKED_IN_DEDUP | query=\"{query[:50]}\" | position={idx+1} | url={link[:100]}")
                    continue
                
                # 跳过已出现的 URL
                if link and link not in seen_links:
                    seen_links.add(link)
                    deduped_results.append(result)
            
            deduped_by_query.append((query, deduped_results))
            deduped_total += len(deduped_results)
        
        return deduped_by_query, original_total, deduped_total

    def _format_results_per_query(self, deduped_by_query: list[tuple[str, list[dict]]]) -> str:
        """格式化每个 query 的去重结果（保留 ======= 分隔）
        
        Args:
            deduped_by_query: [(query1, [deduped_results1...]), ...]
        
        Returns:
            格式化的结果字符串
        """
        all_outputs = []
        
        for query, results in deduped_by_query:
            num_results = len(results)
            output = f"A Google search for '{query}' found {num_results} results:\n\n## Web Results\n"
            
            for idx, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                link = result.get('link', '')
                snippet = result.get('snippet', '')
                output += f"{idx}. [{title}]({link})\n{snippet}\n\n"
            
            all_outputs.append(output.strip())
        
        return "\n\n=======\n\n".join(all_outputs) if all_outputs else "No results found."

    async def create(self, instance_id: Optional[str] = None, create_kwargs: Optional[dict] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        
        create_kwargs = create_kwargs or {}
        masked_url = create_kwargs.get("url", "")
        
        # 调试：记录 create 调用
        _url_mask_logger.info(f"{datetime.now().strftime('%H:%M:%S')} | CREATE | instance={instance_id[:8]} | create_kwargs={create_kwargs} | masked_url=\"{masked_url[:80] if masked_url else ''}\"")

        
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
            "masked_url": masked_url,
        }
        return instance_id, ToolResponse()

    def execute_search(self, instance_id: str, query_list: list, search_service_url: str, limit: int, timeout: int):
        """Execute search for multiple queries - 并行版本，返回原始结果列表."""
        import concurrent.futures
        
        def search_single(query):
            return perform_single_google_search_batch(
                search_service_url, query, "hybrid", limit, None, timeout
            )
        
        # 使用线程池并行执行所有查询
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(query_list)) as executor:
            futures = [executor.submit(search_single, query) for query in query_list]
            results = [f.result() for f in futures]
        
        # 返回原始结果列表，供后续去重处理
        return results, {"query_count": len(query_list), "status": "success"}

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute search tool.
        
        Args:
            parameters: {"query": ["query1", "query2", ...]} - array of queries
        """
        import time
        
        # 支持 query 参数（array 格式）
        query_list = parameters.get("query")
        
        # 兼容单个字符串
        if isinstance(query_list, str):
            query_list = [query_list]
        
        if not query_list or not isinstance(query_list, list):
            error_msg = "Error: 'query' parameter must be an array of strings"
            logger.error(f"[GoogleSearchTool] {error_msg}. Received: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg}, ensure_ascii=False)), 0.0, {}

        try:
            start_time = time.time()
            results_list, metadata = await self.execution_pool.execute.remote(
                self.execute_search,
                instance_id,
                query_list,
                self.search_service_url,
                self.default_limit,
                self.timeout
            )
            elapsed = time.time() - start_time

            # 获取 masked_url
            masked_url = self._instance_dict[instance_id].get("masked_url", "")
            
            # 收集每个 query 的结果
            results_by_query = []
            for i, (result_text, result_metadata) in enumerate(results_list):
                query = query_list[i] if i < len(query_list) else f"query_{i}"
                
                # 解析 JSON 包装
                try:
                    parsed = json.loads(result_text)
                    content = parsed.get("result", result_text)
                except json.JSONDecodeError:
                    content = result_text
                
                # 解析搜索结果块
                parsed_results = self._parse_search_result_block(content)
                results_by_query.append((query, parsed_results))
            
            # 去重（每个 query 相对于前面的 query 去重）
            if self.enable_dedup and results_by_query:
                deduped_by_query, original_count, deduped_count = self._deduplicate_per_query(
                    results_by_query, masked_url
                )
                result_text = self._format_results_per_query(deduped_by_query)
                
                # 记录去重统计
                removed_count = original_count - deduped_count
                if removed_count > 0:
                    logger.info(f"[GoogleSearchTool] Dedup: {original_count} -> {deduped_count} (removed {removed_count} duplicates)")
            else:
                # 不去重时，合并原始结果
                all_results = []
                original_count = 0
                for result_text, result_metadata in results_list:
                    try:
                        parsed = json.loads(result_text)
                        content = parsed.get("result", result_text)
                    except json.JSONDecodeError:
                        content = result_text
                    all_results.append(content)
                    original_count += len(self._parse_search_result_block(content))
                
                result_text = "\n\n=======\n\n".join(all_results) if len(all_results) > 1 else (all_results[0] if all_results else "")
                
                # 应用 URL mask（不去重时仍需要 mask）
                if masked_url:
                    query_str = ", ".join(query_list) if query_list else ""
                    result_text = self._mask_url_in_search_result(result_text, masked_url, query_str)
                
                deduped_count = original_count

            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            metrics = {
                "query_count": len(query_list),
                "status": metadata.get("status", "unknown"),
                "url_masked": bool(masked_url),
                "tool_call_time": elapsed,
                "tool_name": "search",
                "original_result_count": original_count,
                "deduped_result_count": deduped_count,
                "dedup_enabled": self.enable_dedup,
            }

            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            logger.error(f"[GoogleSearchTool] Execution failed: {e}")
            return ToolResponse(text=json.dumps({"result": f"Search failed: {e}"}, ensure_ascii=False)), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> list:
        # 不需要工具计算reward，返回空列表
        # 最终reward由 custom_reward_function (llm_judge_async.py) 计算
        return []

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
