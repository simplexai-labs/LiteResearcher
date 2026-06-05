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
import threading
import time
import traceback
import uuid
from typing import Any, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = 30  # Default browse request timeout
MAX_RETRIES = 1
INITIAL_RETRY_DELAY = 1

logger = logging.getLogger(__name__)

# ============ 全局 Connection Pool (线程安全) ============
# 复用 TCP 连接，避免每次 browse 请求创建新 socket → 减少 FD 消耗
_browse_session_lock = threading.Lock()
_browse_session: Optional[requests.Session] = None


def _get_browse_session() -> requests.Session:
    """获取全局复用的 requests.Session（线程安全，懒初始化）
    
    pool_maxsize=3000: num_workers=500，每次 execute 可能并行多个 URL，峰值 ~2500
    """
    global _browse_session
    if _browse_session is None:
        with _browse_session_lock:
            if _browse_session is None:
                session = requests.Session()
                adapter = HTTPAdapter(
                    pool_connections=20,
                    pool_maxsize=3000,
                    max_retries=Retry(total=0),  # 应用层已有重试
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                session.headers.update({
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                })
                _browse_session = session
                logger.info("✅ Browse 全局 requests.Session 初始化完成 | pool_maxsize=3000")
    return _browse_session


def call_browse_api(browse_service_url: str, link: str, what_to_find: str, timeout: int = DEFAULT_TIMEOUT) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calls the remote browse API to perform webpage browsing with retry logic for various errors,
    using increasing delay between retries. Logs internal calls with a unique ID.
    Uses a global requests.Session for connection pooling.

    Args:
        browse_service_url: The URL of the browse service API.
        link: The link to browse.
        what_to_find: Description of what information to extract.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Browse Request ID: {request_id}] "

    payload = {
        "url": link,  # 后端API仍然使用url参数名
        "question": what_to_find  # 适配后端API的参数名
    }

    session = _get_browse_session()
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling browse API at {browse_service_url}")
            response = session.post(
                browse_service_url,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt + 1}/{MAX_RETRIES}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}Browse API call successful on attempt {attempt + 1}")
            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Browse API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def perform_single_browse_batch(browse_service_url: str, link: str, what_to_find: str, concurrent_semaphore: Optional[threading.Semaphore] = None, timeout: int = DEFAULT_TIMEOUT) -> Tuple[str, Dict[str, Any]]:
    """
    Performs a single browse operation for a link and extraction task.

    Args:
        browse_service_url: The URL of the browse service API.
        link: The link to browse.
        what_to_find: Description of what information to extract.
        concurrent_semaphore: Optional semaphore for concurrency control.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (result_text, metadata).
        result_text: The browse result JSON string.
        metadata: Metadata dictionary for the browse operation.
    """
    logger.info(f"Starting browse operation for link: {link}")

    api_response = None
    error_msg = None

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_browse_api(browse_service_url=browse_service_url, link=link, what_to_find=what_to_find, timeout=timeout)
        else:
            api_response, error_msg = call_browse_api(browse_service_url=browse_service_url, link=link, what_to_find=what_to_find, timeout=timeout)
    except Exception as e:
        error_msg = f"API Request Exception during browse: {e}"
        logger.error(f"Browse operation: {error_msg}")
        traceback.print_exc()

    metadata = {
        "link": link,
        "what_to_find": what_to_find,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "extracted_info": None,
    }

    result_text = json.dumps({"result": "Browse request failed or timed out after retries."}, ensure_ascii=False)

    if error_msg:
        metadata["status"] = "api_error"
        result_text = json.dumps({"result": f"Browse error: {error_msg}"}, ensure_ascii=False)
        logger.error(f"Browse operation: API error occurred: {error_msg}")
    elif api_response:
        logger.debug(f"Browse operation: API Response: {api_response}")
        metadata["api_response"] = api_response

        try:
            # 适配后端返回格式：{"success": bool, "result": str, "from_cache": bool, "error": str}
            success = api_response.get("success", False)
            extracted_info = api_response.get("result", "")
            from_cache = api_response.get("from_cache", False)
            error_info = api_response.get("error", "")
            
            if success and extracted_info:
                # 成功：直接使用后端格式化好的结果
                result_text = extracted_info
                metadata["status"] = "success"
                metadata["extracted_info"] = extracted_info
                metadata["from_cache"] = from_cache
                logger.info(f"Browse operation: Successful (from_cache: {from_cache})")
            elif extracted_info:
                # 失败但有格式化结果（如缓存未命中）：使用后端格式化好的结果
                result_text = extracted_info
                metadata["status"] = "api_error"
                metadata["extracted_info"] = extracted_info
                metadata["from_cache"] = from_cache
                metadata["error_detail"] = error_info
                logger.error(f"Browse operation: API returned error: {error_info}")
            elif error_info:
                # 失败且无格式化结果：返回错误信息
                result_text = f"Browse error: {error_info}"
                metadata["status"] = "api_error"
                metadata["extracted_info"] = ""
                metadata["from_cache"] = from_cache
                metadata["error_detail"] = error_info
                logger.error(f"Browse operation: API returned error: {error_info}")
            else:
                result_text = "No relevant information found for the provided link."
                metadata["status"] = "success"
                metadata["extracted_info"] = ""
                metadata["from_cache"] = from_cache
                logger.info(f"Browse operation: No information found (from_cache: {from_cache})")

        except Exception as e:
            error_msg = f"Error processing browse response: {e}"
            result_text = json.dumps({"result": error_msg}, ensure_ascii=False)
            metadata["status"] = "processing_error"
            logger.error(f"Browse operation: {error_msg}")

    return result_text, metadata

