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
import re
import resource
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor

from verl.tools.utils.browse_utils import perform_single_browse_batch
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _raise_fd_limit(target: int = 65535):
    """提升当前进程的 RLIMIT_NOFILE soft limit（适用于 Ray Worker 进程）"""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(target, hard)
        if soft < new_soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except Exception:
        pass


# 模块加载时立即提升
_raise_fd_limit(65535)

T = TypeVar("T")


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class BrowseTokenBucketWorker:
    """全局 token bucket worker for browse - 使用 threading.Semaphore 控制并发"""
    
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)
        logger.info(f"BrowseTokenBucketWorker initialized with rate_limit={rate_limit}")

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
class BrowseExecutionWorker:
    """执行器：负责实际执行 browse 调用（同步方式，参考 search_tool.py）"""
    
    ACQUIRE_TIMEOUT = 300  # 获取 token 的超时时间（秒）- 从60增加到300秒(5分钟)
    
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
                    logger.warning(f"Error when executing browse: {e}")
                    raise
        else:
            return fn(*fn_args, **fn_kwargs)


def init_browse_token_bucket(rate_limit: int):
    """初始化全局唯一的 BrowseTokenBucketWorker（使用 Ray Named Actor）
    
    使用 get_if_exists=True 避免并发创建冲突（参考 search_tool.py）
    """
    actor_name = "browse_token_bucket_singleton"
    return BrowseTokenBucketWorker.options(
        name=actor_name, 
        get_if_exists=True
    ).remote(rate_limit)


def init_browse_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode):
    """初始化执行池和 token bucket（参考 search_tool.py）"""
    if mode == PoolMode.ThreadMode:
        # 创建全局唯一的 token bucket
        rate_limit_worker = init_browse_token_bucket(rate_limit)
        
        # 创建全局唯一的 execution worker（使用 get_if_exists=True）
        actor_name = "browse_execution_pool_singleton"
        return BrowseExecutionWorker.options(
            name=actor_name,
            get_if_exists=True,
            max_concurrency=num_workers
        ).remote(rate_limit_worker)
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class BrowseTool(BaseTool):
    """Browse tool for visiting webpages. Compatible with search_tool.py format."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        self.num_workers = config.get("num_workers", 60)
        self.rate_limit = config.get("rate_limit", 60)
        self.timeout = config.get("timeout", 30)
        self.strip_evidence = config.get("strip_evidence", False)  # 是否删除 Evidence，只保留 Summary

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_browse_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        self.browse_service_url = config.get("browse_service_url")
        assert self.browse_service_url, "Configuration must include 'browse_service_url'"

        # URL 存在性检查服务（用于统计 fabricated URL 比例）
        # url_check_service_url 指向 RAG 服务的 /batch_check_url 端点
        self.url_check_service_url = config.get("url_check_service_url", "")
        self.url_check_timeout = config.get("url_check_timeout", 30)

        logger.info(f"Initialized BrowseTool with config: {config}, strip_evidence={self.strip_evidence}, url_check={bool(self.url_check_service_url)}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def _strip_evidence_from_result(self, result: str) -> str:
        """从 browse 返回结果中删除 Evidence 部分，只保留 Summary
        
        原始格式:
            The useful information in {url} for user goal {goal} as follows: 
            
            Evidence in page: 
            {evidence}
            
            Summary: 
            {summary}
        
        处理后格式:
            The useful information in {url} for user goal {goal} as follows: 
            
            Summary: 
            {summary}
        """
        if not result or "Evidence in page:" not in result:
            return result
        
        # 匹配并删除 Evidence 部分
        # 格式: Evidence in page:\n{content}\n\nSummary:
        pattern = r'(Evidence in page:\s*\n).*?(\n\nSummary:)'
        result = re.sub(pattern, r'\2', result, flags=re.DOTALL)
        
        # 清理多余的换行
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result

    def _check_urls_exist(self, url_list: list[str]) -> dict[str, bool]:
        """批量检查 URL 是否存在于 RAG 索引中（同步调用，用于统计 fabricated URL）
        
        Returns:
            dict: {url: exists_bool} 映射
        """
        if not self.url_check_service_url or not url_list:
            return {}
        
        try:
            from verl.tools.utils.browse_utils import _get_browse_session
            session = _get_browse_session()
            resp = session.post(
                self.url_check_service_url,
                json={"urls": url_list},
                timeout=self.url_check_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                # 期望返回格式: {"results": [{"url": "...", "exists": true/false}, ...]}
                results = data.get("results", [])
                return {item["url"]: item["exists"] for item in results if "url" in item and "exists" in item}
        except Exception as e:
            logger.warning(f"[BrowseTool] URL existence check failed (non-fatal): {e}")
        
        return {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    def execute_browse(self, instance_id: str, url: str, goal: str, browse_service_url: str, timeout: int):
        """Execute browse for a single URL."""
        url = url.strip() if url else url
        result_text, metadata = perform_single_browse_batch(
            browse_service_url=browse_service_url,
            link=url,
            what_to_find=goal,
            concurrent_semaphore=None,
            timeout=timeout,
        )
        return result_text, metadata

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute browse tool.
        
        Args:
            parameters: {"url": ["url1", "url2", ...] or "url", "goal": "description"}
        """
        import time
        
        # 获取 URL 参数（支持 array 和 string）
        url_param = parameters.get("url")
        goal = parameters.get("goal", "")

        # 转换为列表
        if isinstance(url_param, str):
            url_list = [url_param]
        elif isinstance(url_param, list):
            url_list = url_param
        else:
            url_list = []

        if not url_list:
            error_msg = "Error: 'url' parameter must be a string or array of URLs"
            logger.error(f"[BrowseTool] {error_msg}. Received: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg}, ensure_ascii=False)), 0.0, {}

        if not goal or not isinstance(goal, str):
            error_msg = "Error: 'goal' parameter is required"
            logger.error(f"[BrowseTool] {error_msg}. Received: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg}, ensure_ascii=False)), 0.0, {}

        try:
            import asyncio
            start_time = time.time()
            
            # 并行执行所有 URL 请求
            tasks = [
                self.execution_pool.execute.remote(
                    self.execute_browse, instance_id, url, goal, self.browse_service_url, self.timeout
                )
                for url in url_list
            ]
            results = await asyncio.gather(*tasks)
            
            # 收集结果和统计信息
            all_results = []
            cache_hit_count = 0           # 缓存命中且成功
            crawler_success_count = 0     # 爬虫成功（非缓存）
            cache_miss_count = 0          # 缓存未命中（后端返回 Cache miss 错误）
            connection_error_count = 0    # 连接错误（超时、重置等，没收到后端响应）
            summary_error_count = 0       # Summary 失败（缓存命中但 Summary 处理失败）
            
            for result_text, metadata in results:
                all_results.append(str(result_text))
                
                status = metadata.get("status")
                from_cache = metadata.get("from_cache")
                error_detail = metadata.get("error_detail", "")
                
                if status == "success":
                    if from_cache:
                        cache_hit_count += 1
                    else:
                        crawler_success_count += 1
                elif status == "api_error":
                    if "from_cache" not in metadata:
                        # 没有 from_cache 字段 = 连接错误（没收到后端响应）
                        connection_error_count += 1
                    elif "Cache miss" in error_detail:
                        # 缓存未命中
                        cache_miss_count += 1
                    elif from_cache:
                        # 缓存命中但 Summary 失败
                        summary_error_count += 1
                    else:
                        # 其他错误（爬虫相关）
                        connection_error_count += 1
                elif status == "processing_error":
                    # 处理错误
                    connection_error_count += 1
            
            elapsed = time.time() - start_time

            # ============ URL 存在性检查（统计 fabricated URL 比例） ============
            # 已禁用：设置 url_check_service_url 为非空字符串可重新启用
            fabricated_url_count = 0
            checked_url_count = 0
            if False and self.url_check_service_url and url_list:
                try:
                    url_exists_map = await asyncio.get_event_loop().run_in_executor(
                        None, self._check_urls_exist, url_list
                    )
                    if url_exists_map:
                        checked_url_count = len(url_exists_map)
                        fabricated_url_count = sum(1 for exists in url_exists_map.values() if not exists)
                except Exception as e:
                    logger.warning(f"[BrowseTool] URL check executor failed (non-fatal): {e}")

            # 合并结果
            combined_result = "\n\n=======\n\n".join(all_results) if len(all_results) > 1 else (all_results[0] if all_results else "")
            
            # 如果启用 strip_evidence，删除 Evidence 部分
            if self.strip_evidence:
                combined_result = self._strip_evidence_from_result(combined_result)
            
            self._instance_dict[instance_id]["reward"].append(combined_result.strip())

            # 计算总错误数
            total_errors = connection_error_count + cache_miss_count + summary_error_count
            
            metrics = {
                "url_count": len(url_list),
                "goal": goal[:100],
                "status": "success" if total_errors == 0 else "partial_error",
                "tool_call_time": elapsed,
                "tool_name": "visit",
                # 详细统计
                "cache_hit_count": cache_hit_count,
                "crawler_success_count": crawler_success_count,
                "cache_miss_count": cache_miss_count,
                "connection_error_count": connection_error_count,
                "summary_error_count": summary_error_count,
                # fabricated URL 统计（URL 不存在于 RAG 索引 = 模型编造的）
                "fabricated_url_count": fabricated_url_count,
                "checked_url_count": checked_url_count,
                # 兼容旧字段
                "cache_count": cache_hit_count,
                "crawler_count": crawler_success_count,
                "strip_evidence": self.strip_evidence,
            }

            return ToolResponse(text=combined_result), 0.0, metrics

        except Exception as e:
            logger.error(f"[BrowseTool] Execution failed: {e}")
            return ToolResponse(text=json.dumps({"result": f"Browse failed: {e}"}, ensure_ascii=False)), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> list:
        # 不需要工具计算reward，返回空列表
        # 最终reward由 custom_reward_function (llm_judge_async.py) 计算
        return []

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
