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
import logging
from datetime import datetime
import os
import threading

from pymilvus import connections, Collection

from diskann_config import (
    SEARCH_COLLECTION_NAME, API_MILVUS_URI,
    DISKANN_SEARCH_LIST
)

# 配置日志系统
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

# 并发监控日志目录
concurrency_log_dir = os.path.join(os.path.dirname(__file__), "logs_rag")
os.makedirs(concurrency_log_dir, exist_ok=True)

# 创建请求日志记录器
request_logger = logging.getLogger("rag_requests")
request_logger.setLevel(logging.INFO)

# 请求日志文件（按天轮转）
request_log_file = os.path.join(log_dir, f"rag_requests_{datetime.now().strftime('%Y%m%d')}.log")
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

# ==================== Redis配置 ====================
import redis.asyncio as aioredis
import msgpack
import uuid

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")  # Redis服务器地址
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
INPUT_QUEUE = os.environ.get("INPUT_QUEUE", "embed_queue")  # 请求队列名
EMBEDDING_TIMEOUT = 30         # 等待结果超时(秒)

# 🔥 并发控制：限制同时发送到 Embedding 的请求数
TOTAL_CONCURRENCY_LIMIT = 5000      # 总并发限制
EMBEDDING_CONCURRENCY_LIMIT = 3000  # Embedding 并发限制
MILVUS_CONCURRENCY_LIMIT = 2000     # Milvus 并发限制

# 全局变量
collection = None
redis_client = None  # 异步Redis客户端
total_semaphore = None      # 总并发控制
embedding_semaphore = None  # Embedding并发控制
milvus_semaphore = None     # Milvus 并发控制

# ===================== 并发状态跟踪器 =====================

class ConcurrencyTracker:
    """并发状态跟踪器 - 跟踪running/waiting数量"""
    
    def __init__(self):
        self.lock = threading.Lock()
        # 总并发
        self.total_running = 0
        self.total_waiting = 0
        # Embedding worker
        self.embed_running = 0
        self.embed_waiting = 0
        # Milvus worker  
        self.milvus_running = 0
        self.milvus_waiting = 0
        
    def total_enter_wait(self):
        with self.lock:
            self.total_waiting += 1
    
    def total_enter_run(self):
        with self.lock:
            self.total_waiting -= 1
            self.total_running += 1
    
    def total_exit(self):
        with self.lock:
            self.total_running -= 1
            
    def embed_enter_wait(self):
        with self.lock:
            self.embed_waiting += 1
    
    def embed_enter_run(self):
        with self.lock:
            self.embed_waiting -= 1
            self.embed_running += 1
    
    def embed_exit(self):
        with self.lock:
            self.embed_running -= 1
            
    def milvus_enter_wait(self):
        with self.lock:
            self.milvus_waiting += 1
    
    def milvus_enter_run(self):
        with self.lock:
            self.milvus_waiting -= 1
            self.milvus_running += 1
    
    def milvus_exit(self):
        with self.lock:
            self.milvus_running -= 1
    
    def get_state(self) -> dict:
        with self.lock:
            return {
                "total": {"running": self.total_running, "waiting": self.total_waiting},
                "embed": {"running": self.embed_running, "waiting": self.embed_waiting},
                "milvus": {"running": self.milvus_running, "waiting": self.milvus_waiting},
            }

# 全局并发跟踪器
concurrency_tracker = ConcurrencyTracker()

# ===================== 请求统计器 =====================

class RequestStats:
    """请求统计器 - 线程安全"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.active_requests = 0          # 当前正在处理的请求数
        self.total_requests = 0           # 总请求数
        self.total_embedding_time = 0.0   # 总 embedding 时间
        self.total_milvus_time = 0.0      # 总 milvus 时间
        self.total_response_time = 0.0    # 总响应时间
        self.success_count = 0            # 成功请求数
        self.error_count = 0              # 失败请求数
        self.start_time = time.time()
        
    def request_start(self):
        """请求开始"""
        with self.lock:
            self.active_requests += 1
            self.total_requests += 1
            
    def request_end(self, success: bool, embedding_time: float = 0, milvus_time: float = 0, total_time: float = 0):
        """请求结束"""
        with self.lock:
            self.active_requests -= 1
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
                "total_requests": self.total_requests,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "uptime": uptime,
                "requests_per_second": self.total_requests / max(uptime, 1),
                "avg_embedding_ms": avg_embedding,
                "avg_milvus_ms": avg_milvus,
                "avg_response_ms": avg_response,
            }

# 全局统计器
request_stats = RequestStats()

# ===================== 并发日志记录器 =====================

class ConcurrencyLogger:
    """并发状态日志记录器 - 每3秒写入日志"""
    
    def __init__(self, tracker: ConcurrencyTracker, interval: float = 3.0):
        self.tracker = tracker
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.log_file = None
        
    def start(self):
        """启动日志记录"""
        if self.running:
            return
        # 以启动时间命名日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(concurrency_log_dir, f"{timestamp}.log")
        self.running = True
        self.thread = threading.Thread(target=self._log_loop, daemon=True)
        self.thread.start()
        print(f"📊 并发日志启动: {self.log_file}")
        
    def stop(self):
        """停止日志记录"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            
    def _log_loop(self):
        """日志输出主循环"""
        while self.running:
            self._write_log()
            time.sleep(self.interval)
            
    def _write_log(self):
        """写入并发状态日志"""
        state = self.tracker.get_state()
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = (f"{ts} | "
                f"Total[R:{state['total']['running']:4d} W:{state['total']['waiting']:4d}] | "
                f"Embed[R:{state['embed']['running']:4d} W:{state['embed']['waiting']:4d}] | "
                f"Milvus[R:{state['milvus']['running']:4d} W:{state['milvus']['waiting']:4d}]\n")
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception:
            pass

# 全局日志记录器
concurrency_logger: Optional[ConcurrencyLogger] = None

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

# ==================== 全文取数数据模型 ====================

class FulltextRequest(BaseModel):
    """按 URL 取全文请求"""
    url: str = Field(..., description="要取全文的 URL", min_length=1)

class FulltextResponse(BaseModel):
    """全文取数响应"""
    found: bool = Field(..., description="是否在 PostgreSQL 中找到全文")
    url: str = Field(..., description="查询的 URL")
    title: str = Field("", description="文档标题")
    text: str = Field("", description="整篇正文（来自 PostgreSQL）")

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

async def get_embeddings_batch(queries: List[str]):
    """批量获取embedding - 通过Redis队列"""
    global redis_client
    
    start_time = time.time()
    
    # 生成请求ID并发送到Redis
    req_ids = [str(uuid.uuid4()) for _ in queries]
    
    # 批量写入请求队列
    pipe = redis_client.pipeline()
    for req_id, text in zip(req_ids, queries):
        payload = msgpack.packb({'id': req_id, 'text': text})
        pipe.rpush(INPUT_QUEUE, payload)
    await pipe.execute()
    
    # 等待所有结果
    dense_vectors = []
    sparse_list = []
    total_infer_time = 0
    
    for req_id in req_ids:
        result_key = f"res:{req_id}"
        # 阻塞等待结果
        resp = await redis_client.blpop(result_key, timeout=EMBEDDING_TIMEOUT)
        
        if not resp:
            raise HTTPException(status_code=504, detail="Embedding超时")
        
        # 解析结果
        data = msgpack.unpackb(resp[1])
        
        # Dense向量
        dense_bytes = data[b'dense'] if b'dense' in data else data['dense']
        shape = data[b'dense_shape'] if b'dense_shape' in data else data['dense_shape']
        dense_vec = np.frombuffer(dense_bytes, dtype=np.float32).reshape(shape)
        dense_vectors.append(dense_vec)
        
        # Sparse向量
        sparse_data = data.get(b'sparse') or data.get('sparse')
        if sparse_data:
            sparse_dict = msgpack.unpackb(sparse_data)
            sparse_list.append(sparse_dict)
        
        total_infer_time += data.get(b'infer_time', 0) or data.get('infer_time', 0)
    
    embedding_time = time.time() - start_time
    
    # 合并sparse矩阵
    sparse_matrix = None
    if sparse_list:
        sparse_matrix = deserialize_sparse_matrix(sparse_list[0]) if sparse_list else None
    
    result = {
        "dense": dense_vectors,
        "sparse": sparse_matrix,
        "queue_wait_time": 0,
        "model_inference_time": total_infer_time
    }
    
    print(f"⏱️  批量Embedding ({len(queries)}条): 总={embedding_time*1000:.1f}ms | "
          f"平均={embedding_time/len(queries)*1000:.1f}ms/条")
    
    return result, embedding_time


async def get_embeddings_fast(query: str):
    """获取embedding - 通过Redis队列（带并发控制）"""
    global redis_client, embedding_semaphore, concurrency_tracker
    
    # 🔥 并发控制
    concurrency_tracker.embed_enter_wait()
    async with embedding_semaphore:
        concurrency_tracker.embed_enter_run()
        try:
            start_time = time.time()
            req_id = str(uuid.uuid4())
            
            # 1. 发送请求到Redis队列
            payload = msgpack.packb({'id': req_id, 'text': query})
            await redis_client.rpush(INPUT_QUEUE, payload)
            
            # 2. 等待结果
            result_key = f"res:{req_id}"
            resp = await redis_client.blpop(result_key, timeout=EMBEDDING_TIMEOUT)
            
            if not resp:
                raise HTTPException(status_code=504, detail="Embedding超时")
            
            # 3. 解析结果
            data = msgpack.unpackb(resp[1])
            
            # Dense向量 - 确保是二维 (1, dim)
            dense_bytes = data[b'dense'] if b'dense' in data else data['dense']
            shape = data[b'dense_shape'] if b'dense_shape' in data else data['dense_shape']
            dense_vec = np.frombuffer(dense_bytes, dtype=np.float32).reshape(shape)
            if dense_vec.ndim == 1:
                dense_vec = dense_vec.reshape(1, -1)
            
            # Sparse向量
            sparse_matrix = None
            sparse_data = data.get(b'sparse') or data.get('sparse')
            if sparse_data:
                sparse_dict = msgpack.unpackb(sparse_data)
                sparse_matrix = deserialize_sparse_matrix(sparse_dict)
            
            embedding_time = time.time() - start_time
            infer_time = data.get(b'infer_time', 0) or data.get('infer_time', 0)
            
            result = {
                "dense": dense_vec,
                "sparse": sparse_matrix,
                "queue_wait_time": 0,
                "model_inference_time": infer_time
            }
            
            print(f"⏱️  Embedding: 总={embedding_time*1000:.1f}ms | 推理={infer_time*1000:.1f}ms")
            
            return result, embedding_time
        finally:
            concurrency_tracker.embed_exit()

async def milvus_search_async_diskann(query: str, limit: int, search_type: str = "hybrid",
                                       sparse_weight: float = 0.7, dense_weight: float = 1.0):
    """异步Milvus混合搜索 - DISKANN专用版本"""
    global collection, request_stats, milvus_semaphore, total_semaphore, concurrency_tracker

    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    # 记录请求开始
    request_stats.request_start()
    
    # 🔥 总并发控制
    concurrency_tracker.total_enter_wait()
    async with total_semaphore:
        concurrency_tracker.total_enter_run()
        try:
            for attempt in range(MAX_RETRIES + 1):
                total_start = time.time()
                embedding_time = 0
                milvus_time = 0

                try:
                    embeddings, embedding_time = await get_embeddings_fast(query)

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

                    # 🔥 Milvus 并发控制 + 异步执行
                    concurrency_tracker.milvus_enter_wait()
                    async with milvus_semaphore:
                        concurrency_tracker.milvus_enter_run()
                        try:
                            milvus_start = time.time()
                            loop = asyncio.get_event_loop()
                            results = await loop.run_in_executor(None, _execute_hybrid_search_diskann)
                            milvus_time = time.time() - milvus_start
                        finally:
                            concurrency_tracker.milvus_exit()
            
                    # 打印Milvus检索时间
                    print(f"🔍 Milvus检索: {milvus_time*1000:.1f}ms")

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
                    
                    # 打印总结
                    print(f"📊 请求总结: 总时间={total_time*1000:.1f}ms | "
                          f"Embedding={embedding_time*1000:.1f}ms ({embedding_time/total_time*100:.1f}%) | "
                          f"Milvus={milvus_time*1000:.1f}ms ({milvus_time/total_time*100:.1f}%)")
                    
                    # 记录请求成功
                    request_stats.request_end(
                        success=True,
                        embedding_time=embedding_time,
                        milvus_time=milvus_time,
                        total_time=total_time
                    )
                    
                    return search_results, total_time, embedding_time, milvus_time

                except Exception as e:
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    else:
                        import traceback
                        traceback.print_exc()
                        request_stats.request_end(success=False)
                        raise HTTPException(status_code=500, detail=f"DISKANN搜索失败: {str(e)}")
        finally:
            concurrency_tracker.total_exit()


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
        print(f"📊 批量搜索总结 ({num_queries}条): 总时间={total_time*1000:.1f}ms | "
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
    global collection, redis_client, concurrency_logger, total_semaphore, embedding_semaphore, milvus_semaphore

    print("🚀 初始化DISKANN RAG服务器（Redis队列模式）...")

    try:
        # 初始化并发控制信号量
        total_semaphore = asyncio.Semaphore(TOTAL_CONCURRENCY_LIMIT)
        embedding_semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY_LIMIT)
        milvus_semaphore = asyncio.Semaphore(MILVUS_CONCURRENCY_LIMIT)
        print(f"🔒 并发限制: Total={TOTAL_CONCURRENCY_LIMIT}, Embed={EMBEDDING_CONCURRENCY_LIMIT}, Milvus={MILVUS_CONCURRENCY_LIMIT}")
        
        # 连接Redis
        print(f"📡 连接Redis: {REDIS_HOST}:{REDIS_PORT}")
        redis_client = await aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}"
        )
        await redis_client.ping()
        print(f"✅ Redis连接成功，队列: {INPUT_QUEUE}")

        print("🔗 连接Milvus...")
        connections.connect(uri=API_MILVUS_URI)

        print(f"📚 加载集合: {SEARCH_COLLECTION_NAME}")
        collection = Collection(SEARCH_COLLECTION_NAME)
        collection.load()
        
        # 启动并发日志记录器（每3秒写入）
        concurrency_logger = ConcurrencyLogger(concurrency_tracker, interval=3.0)
        concurrency_logger.start()

        # 初始化 PostgreSQL 全文连接池（可选，SQL 未启用时自动跳过）
        import sql_fulltext
        sql_ready = sql_fulltext.init_pool()

        print("✅ DISKANN RAG服务器初始化完成！")
        print(f"   📊 集合: {SEARCH_COLLECTION_NAME}")
        print(f"   📄 文档数量: {collection.num_entities:,}")
        print(f"   🔢 向量精度: FP32")
        print(f"   📈 索引类型: DISKANN")
        print(f"   📡 Embedding: Redis队列模式")
        print(f"   🗄️  全文取数(/web_parser): {'启用' if sql_ready else '未启用'}")

        return True

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def cleanup_system():
    """清理系统资源"""
    global redis_client, concurrency_logger

    if concurrency_logger:
        concurrency_logger.stop()

    if redis_client:
        await redis_client.close()

    import sql_fulltext
    sql_fulltext.close_pool()

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
        "message": "DISKANN RAG服务器 - Redis队列版",
        "version": "2.0.0-redis",
        "server_id": "local_rag_diskann_server",
        "architecture": "redis_queue",
        "vector_precision": "FP32",
        "index_type": "DISKANN",
        "embedding_queue": INPUT_QUEUE
    }

@app.get("/health")
async def health_check():
    global collection, redis_client, request_stats

    milvus_loaded = collection is not None
    redis_ready = redis_client is not None

    redis_ok = False
    queue_size = 0
    if redis_ready:
        try:
            await redis_client.ping()
            redis_ok = True
            queue_size = await redis_client.llen(INPUT_QUEUE)
        except:
            pass

    status = "healthy" if (milvus_loaded and redis_ok) else "unhealthy"
    
    # 获取统计信息
    stats = request_stats.get_stats()

    return {
        "status": status,
        "milvus_loaded": milvus_loaded,
        "collection_entities": collection.num_entities if collection else 0,
        "redis_connected": redis_ok,
        "embedding_queue_size": queue_size,
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
    使用 Milvus `in` 表达式，一次查询所有URL，性能极好。
    """
    global collection

    try:
        request.state.query_info = f"BatchCheckURL: {len(request_data.urls)}条"

        start_time = time.time()

        # 用 in 表达式一次查出所有存在的URL
        url_list = '", "'.join(u.replace('"', '\\"') for u in request_data.urls)
        expr = f'url in ["{url_list}"]'

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(expr=expr, output_fields=["url"], limit=len(request_data.urls))
        )

        found_urls = {r["url"] for r in results}
        url_results = {url: (url in found_urls) for url in request_data.urls}

        query_time = time.time() - start_time
        print(f"🔍 批量URL检查: {len(request_data.urls)}条 | 存在={len(found_urls)} | 耗时={query_time*1000:.1f}ms")

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
    检查单个URL是否存在于Milvus集合中
    """
    global collection

    try:
        request.state.query_info = f"CheckURL: {request_data.url[:50]}..."

        start_time = time.time()

        escaped_url = request_data.url.replace('"', '\\"')
        expr = f'url == "{escaped_url}"'

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(expr=expr, output_fields=["url"], limit=1)
        )

        exists = len(results) > 0
        query_time = time.time() - start_time
        print(f"🔍 URL检查: {request_data.url[:50]}... | 存在={exists} | 耗时={query_time*1000:.1f}ms")

        return UrlCheckResponse(exists=exists, url=request_data.url)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL检查失败: {str(e)}")


@app.post("/web_parser", response_model=FulltextResponse)
async def web_parser_endpoint(request_data: FulltextRequest, request: Request):
    """
    按 URL 取整篇原文（从 PostgreSQL）。

    向量检索只返回截断的 snippet；当导入时启用了 ENABLE_SQL_STORAGE，
    全文会存入 PostgreSQL。本接口据此返回整篇正文，供研究 agent 阅读全文。

    需要：检索后端启用 ENABLE_SQL_FULLTEXT=true 且 URL 已入库。
    """
    import sql_fulltext

    request.state.query_info = f"WebParser: {request_data.url[:50]}..."

    if not sql_fulltext.is_ready():
        raise HTTPException(
            status_code=503,
            detail="全文取数未启用：请在 diskann_config 设置 ENABLE_SQL_FULLTEXT=true 并确保 PostgreSQL 可达"
        )

    loop = asyncio.get_event_loop()
    row = await loop.run_in_executor(None, sql_fulltext.get_fulltext_by_url, request_data.url)

    if not row:
        return FulltextResponse(found=False, url=request_data.url)

    return FulltextResponse(
        found=True,
        url=row.get("url", request_data.url),
        title=row.get("title", "") or "",
        text=row.get("text", "") or "",
    )


if __name__ == "__main__":
    print("🚀 启动 LiteResearcher 检索后端")
    print("📡 端口: 8018")
    print("🎯 FP32向量 + DISKANN索引（千万级数据）")
    print("⚠️  请先启动 embedding worker: python embedding_worker.py")

    uvicorn.run(
        "local_rag_server:app",  # ✅ 加载自己
        host="0.0.0.0",
        port=8018,
        reload=False,
        workers=4,  # 异步模式，利用 asyncio 处理大量并发
        log_level="info"
    )
