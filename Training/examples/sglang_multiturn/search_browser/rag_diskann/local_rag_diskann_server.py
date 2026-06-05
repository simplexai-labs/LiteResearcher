#!/usr/bin/env python3
"""
🚀 DISKANN RAG服务器 - 端口8018
微服务架构：RAG搜索 + 独立Embedding服务
FP32向量 + DISKANN索引（千万级数据）
"""

import time
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import numpy as np
import asyncio
import aiohttp
from scipy.sparse import csr_array
import logging
from datetime import datetime
import os
import threading
from dataclasses import dataclass, field
import uuid

from pymilvus import connections, Collection

from diskann_config import (
    SEARCH_COLLECTION_NAME, API_MILVUS_URI,
    DISKANN_SEARCH_LIST, DISKANN_EMBEDDING_PORT,
    TOTAL_CONCURRENCY_LIMIT, EMBEDDING_CONCURRENCY_LIMIT, MILVUS_CONCURRENCY_LIMIT,
    RAG_BATCH_ENABLED, RAG_MAX_BATCH_SIZE, RAG_MAX_WAIT_TIME, RAG_MIN_BATCH_SIZE, RAG_BATCH_PROCESSORS
)

# 配置日志系统
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

# 创建请求日志记录器
request_logger = logging.getLogger("rag_requests")
request_logger.setLevel(logging.INFO)

# 请求日志文件（带时分秒，每次启动新文件）
request_log_file = os.path.join(log_dir, f"rag_requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
request_handler = logging.FileHandler(request_log_file, encoding='utf-8')
request_handler.setFormatter(logging.Formatter(
    '%(asctime)s | IP:%(client_ip)s | Endpoint:%(endpoint)s | Method:%(method)s | Status:%(status)s | Time:%(duration).3fs | %(message)s'
))
request_logger.addHandler(request_handler)

# 同时输出到控制台
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '🔍 [%(asctime)s] %(client_ip)s → %(endpoint)s (%(duration).3fs)'
))
request_logger.addHandler(console_handler)

# 创建详细时间日志记录器（用于embedding、milvus时间统计）
timing_logger = logging.getLogger("rag_timing")
timing_logger.setLevel(logging.INFO)
timing_handler = logging.FileHandler(request_log_file, encoding='utf-8')  # 写入同一个文件
timing_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
timing_logger.addHandler(timing_handler)
# 只输出到控制台一次
timing_console = logging.StreamHandler()
timing_console.setFormatter(logging.Formatter('%(message)s'))
timing_logger.addHandler(timing_console)
# 禁止传播到父logger，避免重复
timing_logger.propagate = False

# 配置
EMBEDDING_SERVICE_URL = f"http://10.160.199.231:8028/embed"  # 本地调用
EMBEDDING_TIMEOUT = 10  # 增加到 30 秒，避免高并发时超时
STATS_LOG_INTERVAL = 5  # 统计日志输出间隔（秒）

# 全局变量
collection = None
http_session = None

# 并发控制信号量
total_semaphore: Optional[asyncio.Semaphore] = None
embedding_semaphore: Optional[asyncio.Semaphore] = None
milvus_semaphore: Optional[asyncio.Semaphore] = None

# ===================== RAG端批处理队列（新增） =====================

from collections import deque

class EmbeddingRequest:
    """单个embedding请求"""
    def __init__(self, query: str, request_id: str):
        self.query = query
        self.request_id = request_id
        self.future = asyncio.Future()
        self.enter_time = time.time()

# 全局请求队列
embedding_queue = deque()
queue_lock: Optional[asyncio.Lock] = None
batch_processor_tasks: List[asyncio.Task] = []  # 多个并发批处理器

async def embedding_batch_processor():
    """RAG端批处理器 - 聚合请求减少HTTP连接"""
    global embedding_queue, http_session

    while True:
        try:
            # 等待第一个请求
            while True:
                async with queue_lock:
                    if embedding_queue:
                        break
                await asyncio.sleep(0.001)

            # 收集batch
            batch_items = []
            wait_start = time.time()

            while len(batch_items) < RAG_MAX_BATCH_SIZE:
                async with queue_lock:
                    while embedding_queue and len(batch_items) < RAG_MAX_BATCH_SIZE:
                        batch_items.append(embedding_queue.popleft())

                # 达到最大batch，立即处理
                if len(batch_items) >= RAG_MAX_BATCH_SIZE:
                    break

                # 超时检查
                elapsed = time.time() - wait_start
                if elapsed > RAG_MAX_WAIT_TIME:
                    if len(batch_items) >= RAG_MIN_BATCH_SIZE or elapsed > RAG_MAX_WAIT_TIME * 3:
                        break

                await asyncio.sleep(0.001)

            if not batch_items:
                continue

            # 批量调用embedding服务
            queries = [item.query for item in batch_items]
            request_ids = [item.request_id for item in batch_items]

            try:
                batch_start = time.time()

                # 调用批量embedding接口
                embeddings_result, embedding_time = await get_embeddings_batch(queries)

                # 分发结果
                for i, item in enumerate(batch_items):
                    queue_wait = batch_start - item.enter_time

                    # 提取单个结果（包装成列表格式，保持和直接调用一致）
                    result = {
                        "dense": [embeddings_result["dense"][i]],  # 包装成列表
                        "sparse": embeddings_result["sparse"][[i]] if embeddings_result["sparse"] is not None else None,
                        "queue_wait_time": queue_wait,
                        "model_inference_time": embeddings_result.get("model_inference_time", 0)
                    }

                    # 构造timing信息
                    total_time = time.time() - item.enter_time
                    timing_info = {
                        "http_connect": 0,  # batch模式没有单独连接时间
                        "http_wait": embedding_time * 1000,
                        "total_http": total_time * 1000,
                        "queue_time": queue_wait * 1000,
                        "inference_time": result["model_inference_time"] * 1000,
                        "blind_time": 0,  # batch模式无盲区
                        "batch_size": len(batch_items)
                    }

                    item.future.set_result((result, total_time, timing_info))

            except Exception as e:
                # 批量失败，通知所有请求
                for item in batch_items:
                    if not item.future.done():
                        item.future.set_exception(e)

        except Exception as e:
            print(f"❌ Batch processor错误: {e}")
            await asyncio.sleep(0.1)

# ===================== 请求统计器 =====================

class RequestStats:
    """请求统计器 - 线程安全，支持分阶段统计和排队监控"""
    
    def __init__(self):
        self.lock = threading.Lock()
        # 总体统计
        self.active_requests = 0          # 当前正在处理的请求数
        self.total_queued = 0             # 总排队数（等待进入）
        self.total_requests = 0           # 总请求数
        self.total_embedding_time = 0.0   # 总 embedding 时间
        self.total_milvus_time = 0.0      # 总 milvus 时间
        self.total_response_time = 0.0    # 总响应时间
        self.success_count = 0            # 成功请求数
        self.error_count = 0              # 失败请求数
        self.start_time = time.time()
        
        # 分阶段统计 - running 和 queued
        self.embedding_running = 0        # 正在执行 Embedding 的请求
        self.embedding_queued = 0         # 等待 Embedding 的请求
        self.embedding_done = 0           # Embedding 完成数
        self.milvus_running = 0           # 正在执行 Milvus 搜索的请求
        self.milvus_queued = 0            # 等待 Milvus 的请求
        self.milvus_done = 0              # Milvus 搜索完成数
        
    def request_start(self):
        """请求开始（进入总队列）"""
        with self.lock:
            self.total_queued += 1
            self.total_requests += 1
            
    def request_acquired(self):
        """请求获得总信号量，开始处理"""
        with self.lock:
            self.total_queued = max(0, self.total_queued - 1)
            self.active_requests += 1
    
    def embedding_queue(self):
        """进入 Embedding 队列"""
        with self.lock:
            self.embedding_queued += 1
            
    def embedding_start(self):
        """Embedding 开始（获得信号量）"""
        with self.lock:
            self.embedding_queued = max(0, self.embedding_queued - 1)
            self.embedding_running += 1
            
    def embedding_end(self):
        """Embedding 结束"""
        with self.lock:
            self.embedding_running = max(0, self.embedding_running - 1)
            self.embedding_done += 1
    
    def milvus_queue(self):
        """进入 Milvus 队列"""
        with self.lock:
            self.milvus_queued += 1
            
    def milvus_start(self):
        """Milvus 搜索开始（获得信号量）"""
        with self.lock:
            self.milvus_queued = max(0, self.milvus_queued - 1)
            self.milvus_running += 1
            
    def milvus_end(self):
        """Milvus 搜索结束"""
        with self.lock:
            self.milvus_running = max(0, self.milvus_running - 1)
            self.milvus_done += 1
            
    def request_end(self, success: bool, embedding_time: float = 0, milvus_time: float = 0, total_time: float = 0):
        """请求结束"""
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)
            if success:
                self.success_count += 1
                self.total_embedding_time += embedding_time
                self.total_milvus_time += milvus_time
                self.total_response_time += total_time
            else:
                self.error_count += 1
                
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self.lock:
            uptime = time.time() - self.start_time
            avg_embedding = self.total_embedding_time / max(self.success_count, 1) * 1000
            avg_milvus = self.total_milvus_time / max(self.success_count, 1) * 1000
            avg_response = self.total_response_time / max(self.success_count, 1) * 1000
            
            return {
                "active_requests": self.active_requests,
                "total_queued": self.total_queued,
                "total_requests": self.total_requests,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "uptime": uptime,
                "requests_per_second": self.total_requests / max(uptime, 1),
                "avg_embedding_ms": avg_embedding,
                "avg_milvus_ms": avg_milvus,
                "avg_response_ms": avg_response,
                # 分阶段统计
                "embedding_running": self.embedding_running,
                "embedding_queued": self.embedding_queued,
                "embedding_done": self.embedding_done,
                "milvus_running": self.milvus_running,
                "milvus_queued": self.milvus_queued,
                "milvus_done": self.milvus_done,
            }

# 全局统计器
request_stats = RequestStats()

# ===================== 统计日志记录器（精简版） =====================

class RAGStatsLogger:
    """RAG 服务统计日志记录器（仅保留结构，不做控制台打印）"""

    def __init__(self, stats: RequestStats, interval: float = 5.0):
        self.stats = stats

    def start(self):
        """启动（空实现）"""
        pass

    def stop(self):
        """停止（空实现）"""
        pass

# 全局日志记录器
stats_logger: Optional[RAGStatsLogger] = None

# 数据模型
class SearchRequest(BaseModel):
    query: str = Field(..., description="查询问题")
    limit: int = Field(10, description="返回结果数量", ge=1, le=50)
    search_type: str = Field("hybrid", description="搜索类型: hybrid/dense/sparse")
    sparse_weight: float = Field(0.7, description="稀疏向量权重", ge=0.0, le=2.0)
    dense_weight: float = Field(1.0, description="密集向量权重", ge=0.0, le=2.0)

class SearchResult(BaseModel):
    link: str = Field(..., description="网页链接")
    title: str = Field(..., description="网页标题")
    snippet: str = Field(..., description="网页摘要片段")
    score: float = Field(..., description="相关性评分")

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    search_time: float
    embedding_time: float = Field(..., description="Embedding生成时间")
    milvus_time: float = Field(..., description="Milvus搜索时间")
    search_type: str = Field(..., description="实际使用的搜索类型")
    sparse_weight: float = Field(..., description="稀疏向量权重")
    dense_weight: float = Field(..., description="密集向量权重")
    server_id: str = "local_rag_diskann_server"
    vector_dtype: str = "FP32"
    index_type: str = "DISKANN"

# ==================== 批量查询数据模型 ====================

class BatchSearchRequest(BaseModel):
    """批量搜索请求"""
    queries: List[str] = Field(..., description="查询问题列表", min_length=1, max_length=100)
    limit: int = Field(10, description="每个查询返回结果数量", ge=1, le=50)
    search_type: str = Field("hybrid", description="搜索类型: hybrid/dense/sparse")
    sparse_weight: float = Field(0.7, description="稀疏向量权重", ge=0.0, le=2.0)
    dense_weight: float = Field(1.0, description="密集向量权重", ge=0.0, le=2.0)

class SingleQueryResult(BaseModel):
    """单个查询的结果"""
    query: str = Field(..., description="原始查询")
    results: List[SearchResult] = Field(..., description="搜索结果")
    total: int = Field(..., description="结果数量")

class BatchSearchResponse(BaseModel):
    """批量搜索响应"""
    query_results: List[SingleQueryResult] = Field(..., description="每个查询的结果")
    total_queries: int = Field(..., description="查询总数")
    search_time: float = Field(..., description="总搜索时间")
    embedding_time: float = Field(..., description="Embedding生成时间")
    milvus_time: float = Field(..., description="Milvus搜索时间")
    search_type: str = Field(..., description="搜索类型")
    sparse_weight: float = Field(..., description="稀疏向量权重")
    dense_weight: float = Field(..., description="密集向量权重")
    server_id: str = "local_rag_diskann_server"
    vector_dtype: str = "FP32"
    index_type: str = "DISKANN"

# ==================== URL检查数据模型 ====================

class UrlCheckRequest(BaseModel):
    """URL检查请求"""
    url: str = Field(..., description="要检查的URL", min_length=1)

class UrlCheckResponse(BaseModel):
    """URL检查响应"""
    exists: bool = Field(..., description="URL是否存在")
    url: str = Field(..., description="查询的URL")

class BatchUrlCheckRequest(BaseModel):
    """批量URL检查请求"""
    urls: List[str] = Field(..., description="要检查的URL列表", min_length=1, max_length=1000)

class BatchUrlCheckResponse(BaseModel):
    """批量URL检查响应"""
    results: dict = Field(..., description="URL -> 是否存在 的映射")
    total: int = Field(..., description="查询URL总数")
    found: int = Field(..., description="存在的URL数量")
    query_time: float = Field(..., description="查询耗时（秒）")

# 工具函数
def ensure_vector_dtype_fp32(vector):
    """确保向量类型为FP32（DISKANN要求）"""
    if hasattr(vector, 'dtype') and vector.dtype != np.float32:
        print(f"🔄 转换向量类型: {vector.dtype} → FP32")
        return vector.astype(np.float32)
    return vector

def safe_sparse_check(sparse_data):
    """检查sparse向量是否为空或无效"""
    if sparse_data is None:
        return True
    if hasattr(sparse_data, '__len__') and len(sparse_data) == 0:
        return True
    return False

def deserialize_sparse_matrix(sparse_dict):
    """从字典反序列化为scipy稀疏矩阵"""
    if sparse_dict is None:
        return None
    try:
        indices = np.array(sparse_dict["indices"])
        values = np.array(sparse_dict["values"])
        cols = np.array(sparse_dict["cols"])
        shape = tuple(sparse_dict["shape"])

        from scipy.sparse import coo_array
        coo = coo_array((values, (indices, cols)), shape=shape)
        return coo.tocsr()
    except Exception as e:
        print(f"❌ Sparse反序列化失败: {e}")
        return None

def format_google_snippet(text: str, min_words: int = 20, max_words: int = 30) -> str:
    """格式化snippet，清理多余空白"""
    import re
    import random

    if not text or not text.strip():
        return ""

    # 只处理明确的Unicode转义序列（如 \\uXXXX 字符串形式）
    # 不要对正常UTF-8文本做unicode_escape解码！
    def replace_unicode_escape(match):
        try:
            code = match.group(1)
            return chr(int(code, 16))
        except:
            return match.group(0)
    
    # 仅处理字符串中的 \uXXXX 和 \UXXXXXXXX 转义（注意是4个反斜杠或2个）
    if '\\u' in text or '\\U' in text:
        text = re.sub(r'\\U([0-9A-Fa-f]{8})', replace_unicode_escape, text)
        text = re.sub(r'\\u([0-9A-Fa-f]{4})', replace_unicode_escape, text)

    # 清理多余空白
    text = re.sub(r'\s+', ' ', text.strip())
    words = text.split()
    if len(words) <= min_words:
        return text

    target_length = random.randint(min_words, max_words)
    if len(words) <= target_length:
        return text

    snippet_words = words[:target_length]
    snippet = ' '.join(snippet_words)

    if snippet.endswith(('.', '?', '!')):
        return snippet

    if snippet.endswith(',') and random.random() < 0.3:
        return snippet

    for i in range(len(snippet_words) - 1, max(0, len(snippet_words) - 8), -1):
        word = snippet_words[i]
        if word.endswith(('.', '?', '!')):
            return ' '.join(snippet_words[:i+1])
        elif word.endswith(',') and random.random() < 0.4:
            return ' '.join(snippet_words[:i+1])

    if snippet.endswith((',', '.', '?', '!')):
        snippet = snippet.rstrip(',.?!')

    return snippet + "..."

async def get_embeddings_batch(queries: List[str], retry_count=3):
    """批量获取embedding - 一次请求多条文本（高效）"""
    global http_session
    
    for attempt in range(retry_count):
        start_time = time.time()
        
        try:
            async with http_session.post(
                EMBEDDING_SERVICE_URL,
                json={"texts": queries, "return_dense": True, "return_sparse": True},
                timeout=aiohttp.ClientTimeout(total=EMBEDDING_TIMEOUT * 2)  # 批量请求给更多时间
            ) as response:
                if response.status != 200:
                    error_detail = f"Embedding服务错误: HTTP {response.status}"
                    if attempt < retry_count - 1:
                        print(f"⚠️  {error_detail}, 重试 {attempt+1}/{retry_count-1}")
                        await asyncio.sleep(0.5)
                        continue
                    raise HTTPException(status_code=503, detail=error_detail)

                data = await response.json()
                embedding_time = time.time() - start_time
                
                # 处理 dense 向量
                dense_vectors = []
                if data.get("dense") is not None:
                    for vec in data["dense"]:
                        dense_vectors.append(np.array(vec, dtype=np.float32))
                
                # 处理 sparse 向量
                sparse_matrix = deserialize_sparse_matrix(data.get("sparse"))
                
                result = {
                    "dense": dense_vectors,
                    "sparse": sparse_matrix,
                    "queue_wait_time": data.get("queue_wait_time", 0),
                    "model_inference_time": data.get("model_inference_time", 0)
                }

                timing_logger.info(f"⏱️  批量Embedding ({len(queries)}条): 总={embedding_time*1000:.1f}ms | "
                      f"平均={embedding_time/len(queries)*1000:.1f}ms/条")

                return result, embedding_time

        except asyncio.TimeoutError:
            if attempt < retry_count - 1:
                print(f"⚠️  批量Embedding超时, 重试 {attempt+1}/{retry_count-1}")
                await asyncio.sleep(0.5)
                continue
            raise HTTPException(status_code=504, detail="Embedding服务超时")
        
        except (aiohttp.ClientConnectionError, aiohttp.ClientOSError) as e:
            if attempt < retry_count - 1:
                print(f"⚠️  连接错误: {str(e)[:50]}, 重试 {attempt+1}/{retry_count-1}")
                await asyncio.sleep(1)
                continue
            raise HTTPException(status_code=503, detail=f"Embedding服务连接失败: {str(e)}")
        
        except Exception as e:
            if attempt < retry_count - 1:
                print(f"⚠️  请求失败: {str(e)[:50]}, 重试 {attempt+1}/{retry_count-1}")
                await asyncio.sleep(0.5)
                continue
            raise HTTPException(status_code=503, detail=f"Embedding服务不可用: {str(e)}")


async def get_embeddings_fast(query: str, retry_count=3, request_id=None):
    """获取embedding - 根据配置选择批处理或直接调用"""

    if request_id is None:
        request_id = str(uuid.uuid4())[:8]

    # 如果启用RAG端批处理
    if RAG_BATCH_ENABLED:
        return await get_embeddings_via_batch_queue(query, request_id)
    else:
        return await get_embeddings_direct(query, retry_count, request_id)

async def get_embeddings_via_batch_queue(query: str, request_id: str):
    """通过批处理队列获取embedding"""
    global embedding_queue, queue_lock

    # 创建请求对象
    req = EmbeddingRequest(query, request_id)

    # 加入队列
    async with queue_lock:
        embedding_queue.append(req)

    # 等待结果（最多30秒）
    try:
        result = await asyncio.wait_for(req.future, timeout=30.0)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Embedding批处理超时")

async def get_embeddings_direct(query: str, retry_count=3, request_id=None):
    """直接调用embedding服务（原逻辑）"""
    global http_session

    if request_id is None:
        request_id = str(uuid.uuid4())[:8]

    for attempt in range(retry_count):
        overall_start = time.time()

        try:
            # 阶段1: 准备HTTP请求
            post_start = time.time()
            async with http_session.post(
                EMBEDDING_SERVICE_URL,
                json={"texts": [query], "return_dense": True, "return_sparse": True, "request_id": request_id},
                timeout=aiohttp.ClientTimeout(total=EMBEDDING_TIMEOUT)
            ) as response:
                # 阶段2: 连接建立
                connect_time = time.time()

                if response.status != 200:
                    error_detail = f"Embedding服务错误: HTTP {response.status}"
                    if attempt < retry_count - 1:
                        print(f"⚠️  [{request_id}] {error_detail}, 重试 {attempt+1}/{retry_count-1}")
                        await asyncio.sleep(0.5)
                        continue
                    raise HTTPException(status_code=503, detail=error_detail)

                # 阶段3: 读取响应
                data = await response.json()
                json_done = time.time()

                result = {
                    "dense": np.array(data["dense"], dtype=np.float32),
                    "sparse": deserialize_sparse_matrix(data.get("sparse")),
                    "queue_wait_time": data.get("queue_wait_time", 0),
                    "model_inference_time": data.get("model_inference_time", 0)
                }

                # 计算各阶段耗时
                http_connect = (connect_time - post_start) * 1000
                http_wait = (json_done - connect_time) * 1000
                total_http = (json_done - overall_start) * 1000
                queue_time = result.get('queue_wait_time', 0) * 1000
                inference_time = result.get('model_inference_time', 0) * 1000
                blind_time = total_http - queue_time - inference_time

                # 返回结果和详细timing
                timing_info = {
                    "http_connect": http_connect,
                    "http_wait": http_wait,
                    "total_http": total_http,
                    "queue_time": queue_time,
                    "inference_time": inference_time,
                    "blind_time": blind_time
                }

                return result, (json_done - overall_start), timing_info

        except asyncio.TimeoutError:
            if attempt < retry_count - 1:
                print(f"⚠️  Embedding超时, 重试 {attempt+1}/{retry_count-1}")
                await asyncio.sleep(0.5)
                continue
            raise HTTPException(status_code=504, detail="Embedding服务超时")
        
        except (aiohttp.ClientConnectionError, aiohttp.ClientOSError) as e:
            if attempt < retry_count - 1:
                print(f"⚠️  连接错误: {str(e)[:50]}, 重试 {attempt+1}/{retry_count-1}")
                await asyncio.sleep(1)  # 连接错误等待久一点
                continue
            raise HTTPException(status_code=503, detail=f"Embedding服务连接失败: {str(e)}")
        
        except Exception as e:
            if attempt < retry_count - 1:
                print(f"⚠️  请求失败: {str(e)[:50]}, 重试 {attempt+1}/{retry_count-1}")
                await asyncio.sleep(0.5)
                continue
            raise HTTPException(status_code=503, detail=f"Embedding服务不可用: {str(e)}")

async def milvus_search_async_diskann(query: str, limit: int, search_type: str = "hybrid",
                                       sparse_weight: float = 0.7, dense_weight: float = 1.0):
    """异步Milvus混合搜索 - DISKANN专用版本，带并发控制"""
    global collection, request_stats, total_semaphore, embedding_semaphore, milvus_semaphore

    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    # 记录请求开始（进入总队列）
    request_stats.request_start()
    
    # 获取总并发信号量
    async with total_semaphore:
        request_stats.request_acquired()

        for attempt in range(MAX_RETRIES + 1):
            total_start = time.time()
            embedding_time = 0
            milvus_time = 0

            try:
                # 生成请求ID
                request_id = str(uuid.uuid4())[:8]

                # Embedding 阶段 - 进入队列，获取信号量，执行
                request_stats.embedding_queue()
                async with embedding_semaphore:
                    request_stats.embedding_start()
                    try:
                        embeddings, embedding_time, embed_timing = await get_embeddings_fast(query, request_id=request_id)
                    finally:
                        request_stats.embedding_end()  # 确保总是调用，即使异常

                query_dense = embeddings["dense"][0]
                query_dense = ensure_vector_dtype_fp32(query_dense)

                try:
                    if not safe_sparse_check(embeddings["sparse"]):
                        query_sparse = embeddings["sparse"][[0]]
                    else:
                        query_sparse = []
                except (IndexError, TypeError):
                    query_sparse = []

                search_params = {
                    "metric_type": "IP",
                    "params": {"search_list": DISKANN_SEARCH_LIST}
                }
                output_fields = ["url", "title", "doc"]

                def _execute_hybrid_search_diskann():
                    from pymilvus import AnnSearchRequest, WeightedRanker

                    if search_type == "dense":
                        return collection.search(
                            [query_dense],
                            anns_field="dense_vector",
                            limit=limit,
                            output_fields=output_fields,
                            param=search_params,
                        )[0]
                    elif search_type == "sparse":
                        if safe_sparse_check(query_sparse):
                            return []
                        else:
                            return collection.search(
                                query_sparse,
                                anns_field="sparse_vector",
                                limit=limit,
                                output_fields=output_fields,
                                param={"metric_type": "IP", "params": {}},
                            )[0]
                    else:  # hybrid
                        if safe_sparse_check(query_sparse):
                            return collection.search(
                                [query_dense],
                                anns_field="dense_vector",
                                limit=limit,
                                output_fields=output_fields,
                                param=search_params,
                            )[0]
                        else:
                            dense_req = AnnSearchRequest(
                                [query_dense], "dense_vector", search_params, limit=limit
                            )
                            sparse_req = AnnSearchRequest(
                                query_sparse, "sparse_vector", {"metric_type": "IP", "params": {}}, limit=limit
                            )
                            rerank = WeightedRanker(sparse_weight, dense_weight)

                            return collection.hybrid_search(
                                [sparse_req, dense_req],
                                rerank=rerank,
                                limit=limit,
                                output_fields=output_fields
                            )[0]

                # Milvus 阶段 - 进入队列，获取信号量，执行
                request_stats.milvus_queue()
                async with milvus_semaphore:
                    request_stats.milvus_start()
                    try:
                        milvus_start = time.time()
                        results = _execute_hybrid_search_diskann()
                        milvus_time = time.time() - milvus_start
                    finally:
                        request_stats.milvus_end()  # 确保总是调用，即使异常

                search_results = []
                for hit in results:
                    raw_doc = hit.get("doc", "")
                    formatted_snippet = format_google_snippet(raw_doc)

                    search_results.append(SearchResult(
                        link=hit.get("url", ""),
                        title=hit.get("title", ""),
                        snippet=formatted_snippet,
                        score=float(hit.distance)
                    ))

                total_time = time.time() - total_start

                # 整合所有时间信息到一行
                timing_logger.info(
                    f"⏱️  [{request_id}] 总={total_time*1000:.1f}ms | "
                    f"HTTP调用Embed={embed_timing['total_http']:.1f}ms ("
                    f"连接={embed_timing['http_connect']:.1f}ms "
                    f"等待={embed_timing['http_wait']:.1f}ms "
                    f"服务端排队={embed_timing['queue_time']:.1f}ms "
                    f"推理={embed_timing['inference_time']:.1f}ms "
                    f"盲区={embed_timing['blind_time']:.1f}ms) | "
                    f"Milvus={milvus_time*1000:.1f}ms"
                )

                # 记录请求成功
                request_stats.request_end(
                    success=True,
                    embedding_time=embedding_time,
                    milvus_time=milvus_time,
                    total_time=total_time
                )
                
                return search_results, total_time, embedding_time, milvus_time

            except Exception as e:
                # 清理可能泄漏的队列计数（防止计数器永远不减）
                with request_stats.lock:
                    request_stats.embedding_queued = max(0, request_stats.embedding_queued - 1)
                    request_stats.milvus_queued = max(0, request_stats.milvus_queued - 1)
                
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                else:
                    import traceback
                    traceback.print_exc()
                    # 记录请求失败
                    request_stats.request_end(success=False)
                    raise HTTPException(status_code=500, detail=f"DISKANN搜索失败: {str(e)}")


async def milvus_batch_search_async_diskann(
    queries: List[str], 
    limit: int, 
    search_type: str = "hybrid",
    sparse_weight: float = 0.7, 
    dense_weight: float = 1.0
):
    """
    批量异步Milvus混合搜索 - 一次请求多个查询
    
    优势：
    1. Embedding 一次请求，减少网络开销
    2. Milvus 批量搜索，提高吞吐量
    """
    global collection, request_stats
    
    total_start = time.time()
    num_queries = len(queries)
    
    # 记录请求开始（批量算作一个请求）
    request_stats.request_start()
    
    try:
        # 1. 批量获取 Embedding
        embeddings, embedding_time = await get_embeddings_batch(queries)
        
        # 2. 准备所有查询的向量
        all_dense_vectors = []
        all_sparse_vectors = []
        
        for i in range(num_queries):
            # Dense 向量
            dense_vec = embeddings["dense"][i]
            dense_vec = ensure_vector_dtype_fp32(dense_vec)
            all_dense_vectors.append(dense_vec)
            
            # Sparse 向量
            try:
                if not safe_sparse_check(embeddings["sparse"]):
                    sparse_vec = embeddings["sparse"][[i]]
                else:
                    sparse_vec = None
            except (IndexError, TypeError):
                sparse_vec = None
            all_sparse_vectors.append(sparse_vec)
        
        # 3. 批量 Milvus 搜索
        search_params = {
            "metric_type": "IP",
            "params": {"search_list": DISKANN_SEARCH_LIST}
        }
        output_fields = ["url", "title", "doc"]
        
        milvus_start = time.time()
        
        all_results = []
        
        if search_type == "dense":
            # Dense 搜索 - 批量
            batch_results = collection.search(
                all_dense_vectors,
                anns_field="dense_vector",
                limit=limit,
                output_fields=output_fields,
                param=search_params,
            )
            all_results = batch_results
            
        elif search_type == "sparse":
            # Sparse 搜索 - 逐个（因为稀疏向量格式不同）
            for sparse_vec in all_sparse_vectors:
                if sparse_vec is not None and not safe_sparse_check(sparse_vec):
                    result = collection.search(
                        sparse_vec,
                        anns_field="sparse_vector",
                        limit=limit,
                        output_fields=output_fields,
                        param={"metric_type": "IP", "params": {}},
                    )[0]
                    all_results.append(result)
                else:
                    all_results.append([])
                    
        else:  # hybrid
            from pymilvus import AnnSearchRequest, WeightedRanker
            
            # Hybrid 搜索 - 逐个执行
            for i in range(num_queries):
                dense_vec = all_dense_vectors[i]
                sparse_vec = all_sparse_vectors[i]
                
                if sparse_vec is None or safe_sparse_check(sparse_vec):
                    # 只有 dense
                    result = collection.search(
                        [dense_vec],
                        anns_field="dense_vector",
                        limit=limit,
                        output_fields=output_fields,
                        param=search_params,
                    )[0]
                else:
                    # 混合搜索
                    dense_req = AnnSearchRequest(
                        [dense_vec], "dense_vector", search_params, limit=limit
                    )
                    sparse_req = AnnSearchRequest(
                        sparse_vec, "sparse_vector", {"metric_type": "IP", "params": {}}, limit=limit
                    )
                    rerank = WeightedRanker(sparse_weight, dense_weight)
                    
                    result = collection.hybrid_search(
                        [sparse_req, dense_req],
                        rerank=rerank,
                        limit=limit,
                        output_fields=output_fields
                    )[0]
                    
                all_results.append(result)
        
        milvus_time = time.time() - milvus_start
        
        # 4. 格式化结果
        query_results = []
        for i, (query, results) in enumerate(zip(queries, all_results)):
            search_results = []
            for hit in results:
                raw_doc = hit.get("doc", "")
                formatted_snippet = format_google_snippet(raw_doc)
                
                search_results.append(SearchResult(
                    link=hit.get("url", ""),
                    title=hit.get("title", ""),
                    snippet=formatted_snippet,
                    score=float(hit.distance)
                ))
            
            query_results.append(SingleQueryResult(
                query=query,
                results=search_results,
                total=len(search_results)
            ))
        
        total_time = time.time() - total_start

        # 打印总结
        timing_logger.info(f"📊 批量搜索总结 ({num_queries}条): 总时间={total_time*1000:.1f}ms | "
              f"Embedding={embedding_time*1000:.1f}ms | Milvus={milvus_time*1000:.1f}ms | "
              f"平均={total_time/num_queries*1000:.1f}ms/条")

        # 记录请求成功
        request_stats.request_end(
            success=True,
            embedding_time=embedding_time,
            milvus_time=milvus_time,
            total_time=total_time
        )
        
        return query_results, total_time, embedding_time, milvus_time
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        request_stats.request_end(success=False)
        raise HTTPException(status_code=500, detail=f"批量搜索失败: {str(e)}")


async def initialize_system():
    """初始化DISKANN RAG系统"""
    global collection, http_session, stats_logger, request_stats
    global total_semaphore, embedding_semaphore, milvus_semaphore
    global queue_lock, batch_processor_tasks

    print("🚀 初始化DISKANN RAG服务器（微服务架构）...")

    try:
        # 初始化并发控制信号量
        total_semaphore = asyncio.Semaphore(TOTAL_CONCURRENCY_LIMIT)
        embedding_semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY_LIMIT)
        milvus_semaphore = asyncio.Semaphore(MILVUS_CONCURRENCY_LIMIT)
        print(f"🔒 并发限制: Total={TOTAL_CONCURRENCY_LIMIT}, Embedding={EMBEDDING_CONCURRENCY_LIMIT}, Milvus={MILVUS_CONCURRENCY_LIMIT}")

        # 初始化RAG端批处理队列（多个并发处理器）
        if RAG_BATCH_ENABLED:
            queue_lock = asyncio.Lock()
            batch_processor_tasks.clear()  # 清空旧任务
            for i in range(RAG_BATCH_PROCESSORS):
                task = asyncio.create_task(embedding_batch_processor())
                batch_processor_tasks.append(task)
            print(f"📦 RAG端批处理已启用: batch={RAG_MAX_BATCH_SIZE}, wait={RAG_MAX_WAIT_TIME*1000:.0f}ms, processors={RAG_BATCH_PROCESSORS}")

        # 优化连接器配置 - 支持高并发
        connector = aiohttp.TCPConnector(
            limit=2000,             # 总连接数（支持高并发）
            limit_per_host=1000,    # 单主机连接数（关键！支持1000并发到Embedding服务）
            ttl_dns_cache=300,      # DNS缓存
            force_close=False,      # 不强制关闭
            enable_cleanup_closed=True,
            keepalive_timeout=60    # Keep-alive超时（增加）
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=15)
        http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        print(f"🤖 Embedding服务: {EMBEDDING_SERVICE_URL}")
        print(f"⚠️  跳过初始化健康检查（防火墙限制），运行时再验证")

        print("🔗 连接Milvus...")
        connections.connect(uri=API_MILVUS_URI)

        print(f"📚 加载集合: {SEARCH_COLLECTION_NAME}")
        collection = Collection(SEARCH_COLLECTION_NAME)
        collection.load()
        
        # 启动统计日志记录器（控制台 5 秒，Worker日志 1 秒）
        stats_logger = RAGStatsLogger(request_stats, interval=STATS_LOG_INTERVAL)
        stats_logger.start()

        print("✅ DISKANN RAG服务器初始化完成！")
        print(f"   📊 集合: {SEARCH_COLLECTION_NAME}")
        print(f"   📄 文档数量: {collection.num_entities:,}")
        print(f"   🔢 向量精度: FP32")
        print(f"   📈 索引类型: DISKANN")
        print(f"   📊 统计日志间隔: {STATS_LOG_INTERVAL}s")

        return True

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def cleanup_system():
    """清理系统资源"""
    global http_session, stats_logger, batch_processor_tasks

    # 停止统计日志记录器
    if stats_logger:
        stats_logger.stop()

    # 取消所有批处理器任务
    for task in batch_processor_tasks:
        task.cancel()

    # 等待所有任务完成
    if batch_processor_tasks:
        await asyncio.gather(*batch_processor_tasks, return_exceptions=True)

    if http_session:
        await http_session.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    success = await initialize_system()
    if not success:
        raise RuntimeError("系统初始化失败")
    yield
    await cleanup_system()

app = FastAPI(
    title="DISKANN RAG Service",
    description="DISKANN-based RAG search (FP32, 10M+ docs)",
    version="2.0.0",
    lifespan=lifespan
)

# 配置JSON响应，确保中文不被ASCII转义
import json
from starlette.responses import JSONResponse as StarletteJSONResponse

class UnicodeJSONResponse(StarletteJSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,  # 关键：不将Unicode转义为ASCII
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

app.router.default_response_class = UnicodeJSONResponse

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有HTTP请求"""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"

    # 处理请求
    response = await call_next(request)

    # 计算请求耗时
    duration = time.time() - start_time

    # 记录日志
    query_info = getattr(request.state, 'query_info', '')
    request_logger.info(
        query_info,
        extra={
            'client_ip': client_ip,
            'endpoint': request.url.path,
            'method': request.method,
            'status': response.status_code,
            'duration': duration
        }
    )

    return response

@app.get("/")
async def root():
    return {
        "message": "DISKANN RAG服务器 - 微服务版",
        "version": "2.0.0-microservice",
        "server_id": "local_rag_diskann_server",
        "architecture": "microservice",
        "vector_precision": "FP32",
        "index_type": "DISKANN",
        "embedding_service": EMBEDDING_SERVICE_URL
    }

@app.get("/health")
async def health_check():
    global collection, http_session, request_stats

    milvus_loaded = collection is not None
    http_ready = http_session is not None

    embedding_service_ok = False
    if http_ready:
        try:
            async with http_session.get(
                EMBEDDING_SERVICE_URL.replace("/embed", "/health"),
                timeout=aiohttp.ClientTimeout(total=3)
            ) as response:
                embedding_service_ok = response.status == 200
        except:
            pass

    status = "healthy" if (milvus_loaded and http_ready and embedding_service_ok) else "unhealthy"
    
    # 获取统计信息
    stats = request_stats.get_stats()

    return {
        "status": status,
        "milvus_loaded": milvus_loaded,
        "collection_entities": collection.num_entities if collection else 0,
        "embedding_service_connected": embedding_service_ok,
        "vector_precision": "FP32",
        "index_type": "DISKANN",
        "stats": {
            "active_requests": stats["active_requests"],
            "total_requests": stats["total_requests"],
            "success_count": stats["success_count"],
            "error_count": stats["error_count"],
            "requests_per_second": round(stats["requests_per_second"], 2),
            "avg_response_ms": round(stats["avg_response_ms"], 1)
        }
    }

@app.get("/stats")
async def get_stats():
    """获取详细统计信息"""
    global request_stats
    
    stats = request_stats.get_stats()
    
    return {
        "uptime_seconds": round(stats["uptime"], 1),
        "active_requests": stats["active_requests"],
        "total_requests": stats["total_requests"],
        "success_count": stats["success_count"],
        "error_count": stats["error_count"],
        "throughput": {
            "requests_per_second": round(stats["requests_per_second"], 2)
        },
        "latency": {
            "avg_embedding_ms": round(stats["avg_embedding_ms"], 1),
            "avg_milvus_ms": round(stats["avg_milvus_ms"], 1),
            "avg_total_ms": round(stats["avg_response_ms"], 1)
        }
    }

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request_data: SearchRequest, request: Request):
    try:
        if request_data.search_type not in ["hybrid", "dense", "sparse"]:
            raise HTTPException(status_code=400, detail="search_type必须是: hybrid, dense, 或 sparse")

        # 记录查询信息到request.state，供中间件使用
        request.state.query_info = f"Query:'{request_data.query[:50]}...' Limit:{request_data.limit} Type:{request_data.search_type}"

        results, total_time, embedding_time, milvus_time = await milvus_search_async_diskann(
            query=request_data.query,
            limit=request_data.limit,
            search_type=request_data.search_type,
            sparse_weight=request_data.sparse_weight,
            dense_weight=request_data.dense_weight
        )

        return SearchResponse(
            results=results,
            total=len(results),
            search_time=total_time,
            embedding_time=embedding_time,
            milvus_time=milvus_time,
            search_type=request_data.search_type,
            sparse_weight=request_data.sparse_weight,
            dense_weight=request_data.dense_weight,
            server_id="local_rag_diskann_server",
            vector_dtype="FP32",
            index_type="DISKANN"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@app.post("/batch_search", response_model=BatchSearchResponse)
async def batch_search_endpoint(request_data: BatchSearchRequest, request: Request):
    """
    批量搜索端点 - 一次请求多个查询
    
    优势：
    - Embedding 批量处理，减少网络延迟
    - 适合需要同时搜索多个问题的场景
    - 比多次调用 /search 更高效
    
    示例请求：
    {
        "queries": ["问题1", "问题2", "问题3"],
        "limit": 10,
        "search_type": "hybrid"
    }
    """
    try:
        if request_data.search_type not in ["hybrid", "dense", "sparse"]:
            raise HTTPException(status_code=400, detail="search_type必须是: hybrid, dense, 或 sparse")
        
        if len(request_data.queries) > 100:
            raise HTTPException(status_code=400, detail="单次批量查询最多支持100条")

        # 记录查询信息到request.state
        request.state.query_info = f"BatchQuery: {len(request_data.queries)}条 Limit:{request_data.limit} Type:{request_data.search_type}"

        query_results, total_time, embedding_time, milvus_time = await milvus_batch_search_async_diskann(
            queries=request_data.queries,
            limit=request_data.limit,
            search_type=request_data.search_type,
            sparse_weight=request_data.sparse_weight,
            dense_weight=request_data.dense_weight
        )

        return BatchSearchResponse(
            query_results=query_results,
            total_queries=len(request_data.queries),
            search_time=total_time,
            embedding_time=embedding_time,
            milvus_time=milvus_time,
            search_type=request_data.search_type,
            sparse_weight=request_data.sparse_weight,
            dense_weight=request_data.dense_weight,
            server_id="local_rag_diskann_server",
            vector_dtype="FP32",
            index_type="DISKANN"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量搜索失败: {str(e)}")


@app.post("/batch_check_url", response_model=BatchUrlCheckResponse)
async def batch_check_url_endpoint(request_data: BatchUrlCheckRequest, request: Request):
    """
    批量检查URL是否存在于Milvus集合中
    
    使用 Milvus `in` 表达式一次查询所有URL，一次网络请求搞定。
    """
    global collection

    try:
        request.state.query_info = f"BatchCheckURL: {len(request_data.urls)}条"

        start_time = time.time()

        # 用 in 表达式一次查出所有存在的URL
        url_list = '", "'.join(u.replace('"', '\\"') for u in request_data.urls)
        expr = f'url in ["{url_list}"]'

        results = collection.query(
            expr=expr,
            output_fields=["url"],
            limit=len(request_data.urls)
        )

        found_urls = {r["url"] for r in results}
        url_results = {url: (url in found_urls) for url in request_data.urls}

        query_time = time.time() - start_time
        timing_logger.info(f"🔍 批量URL检查: {len(request_data.urls)}条 | 存在={len(found_urls)} | 耗时={query_time*1000:.1f}ms")

        return BatchUrlCheckResponse(
            results=url_results,
            total=len(request_data.urls),
            found=len(found_urls),
            query_time=query_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量URL检查失败: {str(e)}")


@app.post("/check_url", response_model=UrlCheckResponse)
async def check_url_endpoint(request_data: UrlCheckRequest, request: Request):
    """
    检查URL是否存在于Milvus集合中
    
    简单方案：使用Milvus的query功能，通过expr表达式精确匹配URL
    """
    global collection
    
    try:
        # 记录查询信息
        request.state.query_info = f"CheckURL: {request_data.url[:50]}..."
        
        start_time = time.time()
        
        # 使用Milvus query功能，通过expr表达式查询URL
        # 注意：需要转义URL中的特殊字符（如单引号、双引号）
        escaped_url = request_data.url.replace("'", "\\'").replace('"', '\\"')
        expr = f'url == "{escaped_url}"'
        
        results = collection.query(
            expr=expr,
            output_fields=["url"],
            limit=1
        )
        
        exists = len(results) > 0
        query_time = time.time() - start_time
        
        timing_logger.info(f"🔍 URL检查: {request_data.url[:50]}... | 存在={exists} | 耗时={query_time*1000:.1f}ms")
        
        return UrlCheckResponse(
            exists=exists,
            url=request_data.url
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL检查失败: {str(e)}")


if __name__ == "__main__":
    print("🚀 启动DISKANN RAG服务器（微服务架构）")
    print("📡 端口: 8018")
    print("🎯 FP32向量 + DISKANN索引（千万级数据）")
    print("⚠️  请先启动: python embedding_server_diskann.py")

    uvicorn.run(
        "local_rag_diskann_server:app",
        host="0.0.0.0",
        port=8010,
        reload=False,
        workers=1,  # 单进程异步模式！利用 asyncio 处理大量并发，统计准确
        log_level="info"
    )
