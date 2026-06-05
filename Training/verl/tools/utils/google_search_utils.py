import json
import logging
import threading
import time
from typing import Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1

logger = logging.getLogger(__name__)

# ============ 全局 Connection Pool (线程安全) ============
# 使用 requests.Session + HTTPAdapter 复用 TCP 连接，避免每次请求都创建新 socket
# pool_connections: 连接池中缓存的 host 数量
# pool_maxsize: 每个 host 最大保持的连接数（应 >= 并发线程数）
_session_lock = threading.Lock()
_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """获取全局复用的 requests.Session（线程安全，懒初始化）
    
    Connection Pool 配置说明：
    - pool_connections=20: 缓存 20 个不同 host 的连接池
    - pool_maxsize=3000: 每个 host 最多保持 3000 个持久连接
      （num_workers=500，每次 execute_search 可能并行 5+ query，峰值 ~2500 并发连接）
    - max_retries: urllib3 层面的重试（连接级别，非应用级别）
    """
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                session = requests.Session()
                # 配置连接池适配器
                # pool_maxsize 需要 >= 实际峰值并发连接数
                # 500 workers × ~5 queries/call = ~2500 峰值，设 3000 留余量
                retry_strategy = Retry(
                    total=0,  # 应用层已有重试逻辑，urllib3 层不再重试
                    backoff_factor=0,
                )
                adapter = HTTPAdapter(
                    pool_connections=20,
                    pool_maxsize=5000,
                    max_retries=retry_strategy,
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                # 预设通用 headers
                session.headers.update({"Content-Type": "application/json"})
                _session = session
                logger.info("✅ 全局 requests.Session 初始化完成 | pool_maxsize=3000")
    return _session


def _call_search_api(url: str, query: str, search_type: str, limit: int, timeout: int) -> Tuple[Optional[Dict], Optional[str]]:
    """调用 RAG 搜索 API，支持重试（复用全局 Session 连接池）"""
    payload = {"query": query, "search_type": search_type, "limit": limit}
    session = _get_session()
    
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(url, json=payload, timeout=timeout)
            
            if response.status_code in [500, 502, 503, 504]:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return None, f"Server error: {response.status_code}"
            
            response.raise_for_status()
            return response.json(), None
            
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            return None, "Request timeout"
        except requests.exceptions.RequestException as e:
            return None, f"Request error: {e}"
        except json.JSONDecodeError as e:
            return None, f"JSON decode error: {e}"
    
    return None, "Failed after retries"


def _format_search_results(query: str, results: list) -> str:
    """格式化搜索结果为指定格式"""
    num_results = len(results)
    output = f"A Google search for '{query}' found {num_results} results:\n\n## Web Results\n"
    
    for idx, result in enumerate(results, 1):
        title = result.get("title", "No title")
        link = result.get("link", "")
        snippet = result.get("snippet", "")
        
        output += f"{idx}. [{title}]({link})\n{snippet}\n\n"
    
    return output.strip()


def perform_single_google_search_batch(
    search_service_url: str, 
    query: str, 
    search_type: str = "hybrid", 
    limit: int = 10, 
    concurrent_semaphore: Optional = None, 
    timeout: int = DEFAULT_TIMEOUT
) -> Tuple[str, Dict]:
    """执行单次 RAG 搜索请求
    
    Args:
        search_service_url: 搜索服务 URL (例如: http://localhost:8018/search)
        query: 搜索查询
        search_type: 搜索类型 (hybrid/dense/sparse)
        limit: 返回结果数量
        concurrent_semaphore: 并发控制（Ray 已处理，此参数保留兼容）
        timeout: 超时时间
    
    Returns:
        (result_text, metadata)
    """
    metadata = {
        "query": query,
        "search_type": search_type,
        "limit": limit,
        "status": "unknown",
        "total_results": 0,
        "search_time": 0,
        "api_request_error": None
    }
    
    # 调用 API
    api_response, error_msg = _call_search_api(search_service_url, query, search_type, limit, timeout)
    
    if error_msg:
        metadata["status"] = "api_error"
        metadata["api_request_error"] = error_msg
        result_text = json.dumps({"result": f"Search error: {error_msg}"}, ensure_ascii=False)
        logger.error(f"Search failed: {error_msg}")
        return result_text, metadata
    
    # 解析结果
    results = api_response.get("results", [])
    total_results = api_response.get("total", 0)
    search_time = api_response.get("search_time", 0)
    
    if not results:
        metadata["status"] = "no_results"
        result_text = json.dumps({"result": f"A RAG search for '{query}' found 0 results."}, ensure_ascii=False)
        logger.info(f"No results for query: {query}")
        return result_text, metadata
    
    # 格式化输出
    formatted_result = _format_search_results(query, results)
    result_text = json.dumps({"result": formatted_result}, ensure_ascii=False)
    
    metadata["status"] = "success"
    metadata["total_results"] = total_results
    metadata["search_time"] = search_time
    
    logger.info(f"Search successful: {total_results} results in {search_time:.3f}s")
    
    return result_text, metadata
