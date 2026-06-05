#!/usr/bin/env python3
"""
🤖 DISKANN Embedding服务 - 端口8028
FP32精度 + 动态batching
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
from collections import deque
import logging
from datetime import datetime
import os

from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from diskann_config import (
    API_BGE_MODEL_PATH, API_BGE_MAX_LENGTH, API_BATCH_SIZE,
    API_DEVICE, MAX_BATCH_SIZE, MAX_WAIT_TIME, MIN_BATCH_SIZE,
    EMBEDDING_WORKERS
)
import fcntl
import tempfile

# 配置日志系统
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

# 创建两个日志记录器：一个用于请求日志，一个用于系统日志
request_logger = logging.getLogger("embedding_requests")
request_logger.setLevel(logging.INFO)

# 请求日志文件（按天轮转）
request_log_file = os.path.join(log_dir, f"embedding_requests_{datetime.now().strftime('%Y%m%d')}.log")
request_handler = logging.FileHandler(request_log_file, encoding='utf-8')
request_handler.setFormatter(logging.Formatter(
    '%(asctime)s | IP:%(client_ip)s | Endpoint:%(endpoint)s | Method:%(method)s | Status:%(status)s | Time:%(duration).3fs | %(message)s'
))
request_logger.addHandler(request_handler)

# 同时输出到控制台
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '📝 [%(asctime)s] %(client_ip)s → %(endpoint)s (%(duration).3fs)'
))
request_logger.addHandler(console_handler)

# 稀疏矩阵序列化
def serialize_sparse_matrix(sparse_matrix):
    if sparse_matrix is None:
        return None
    try:
        coo = sparse_matrix.tocoo()
        return {
            "indices": coo.row.tolist(),
            "values": coo.data.tolist(),
            "shape": list(coo.shape),
            "cols": coo.col.tolist()
        }
    except Exception as e:
        print(f"❌ Sparse序列化失败: {e}")
        return None

# 全局变量
embedding_function = None
request_queue = deque()
queue_lock = asyncio.Lock()
batch_processor_task = None
current_gpu_device = None  # 当前worker使用的GPU


def get_worker_gpu():
    """
    为当前 worker 分配 GPU（基于文件锁的原子计数器）
    每个 worker 启动时获取一个唯一的 GPU ID
    """
    gpu_count = len(API_DEVICE) if isinstance(API_DEVICE, list) else 1
    if gpu_count == 1:
        return API_DEVICE[0] if isinstance(API_DEVICE, list) else API_DEVICE
    
    counter_file = os.path.join(tempfile.gettempdir(), "embedding_server_gpu_counter")
    lock_file = counter_file + ".lock"
    
    # 使用文件锁确保原子操作
    with open(lock_file, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            # 读取当前计数
            if os.path.exists(counter_file):
                with open(counter_file, 'r') as f:
                    content = f.read().strip()
                    count = int(content) if content else 0
            else:
                count = 0
            
            # 分配 GPU（轮询）
            gpu_idx = count % gpu_count
            device = API_DEVICE[gpu_idx]
            
            # 更新计数
            with open(counter_file, 'w') as f:
                f.write(str(count + 1))
            
            return device
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)

# 数据模型
class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)
    return_dense: bool = Field(True)
    return_sparse: bool = Field(True)

class EmbedResponse(BaseModel):
    dense: Optional[List[List[float]]] = None
    sparse: Optional[dict] = None
    embedding_time: float
    text_count: int
    vector_dtype: str
    batch_size: int
    queue_wait_time: float = 0.0  # 队列等待时间
    model_inference_time: float = 0.0  # 模型推理时间

class RequestItem:
    def __init__(self, texts: List[str], return_dense: bool, return_sparse: bool):
        self.texts = texts
        self.return_dense = return_dense
        self.return_sparse = return_sparse
        self.future = asyncio.Future()
        self.timestamp = time.time()

async def batch_processor():
    """后台批处理任务 - 带等待合并逻辑"""
    global request_queue, embedding_function

    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()

    while True:
        try:
            # 等待第一个请求
            while True:
                async with queue_lock:
                    if request_queue:
                        break
                await asyncio.sleep(0.001)
            
            # 收集批次：尽可能多地收集请求，充分利用GPU
            batch_items = []
            wait_start = time.time()
            
            while len(batch_items) < MAX_BATCH_SIZE:
                async with queue_lock:
                    while request_queue and len(batch_items) < MAX_BATCH_SIZE:
                        batch_items.append(request_queue.popleft())
                
                # 达到最大batch，立即处理
                if len(batch_items) >= MAX_BATCH_SIZE:
                    break
                
                # 等待时间超过阈值，处理当前收集的（至少要有MIN_BATCH_SIZE个，除非超时）
                elapsed = time.time() - wait_start
                if elapsed > MAX_WAIT_TIME:
                    if len(batch_items) >= MIN_BATCH_SIZE or elapsed > MAX_WAIT_TIME * 3:
                        break
                
                # 短暂等待更多请求
                await asyncio.sleep(0.001)

            if not batch_items:
                continue

            all_texts = []
            text_counts = []
            for item in batch_items:
                text_counts.append(len(item.texts))
                all_texts.extend(item.texts)

            try:
                inference_start = time.time()

                def gpu_inference():
                    return embedding_function(all_texts)

                result = await loop.run_in_executor(executor, gpu_inference)
                inference_time = time.time() - inference_start

                offset = 0
                for i, item in enumerate(batch_items):
                    count = text_counts[i]
                    queue_wait = inference_start - item.timestamp  # 队列等待时间

                    response_data = {
                        "embedding_time": inference_time / len(batch_items),
                        "model_inference_time": inference_time / len(batch_items),
                        "queue_wait_time": queue_wait,
                        "text_count": count,
                        "vector_dtype": "FP32",
                        "batch_size": len(all_texts)
                    }

                    if item.return_dense and result.get("dense") is not None:
                        dense_slice = result["dense"][offset:offset+count]
                        if hasattr(dense_slice, 'tolist'):
                            response_data["dense"] = dense_slice.tolist()
                        else:
                            response_data["dense"] = [v.tolist() if hasattr(v, 'tolist') else list(v) for v in dense_slice]

                    if item.return_sparse and result.get("sparse") is not None:
                        sparse_slice = result["sparse"][offset:offset+count]
                        response_data["sparse"] = serialize_sparse_matrix(sparse_slice)

                    offset += count

                    if not item.future.done():
                        item.future.set_result(response_data)

            except Exception as e:
                print(f"❌ Batch处理失败: {e}")
                for item in batch_items:
                    if not item.future.done():
                        item.future.set_exception(e)

        except Exception as e:
            print(f"❌ 批处理器异常: {e}")
            await asyncio.sleep(1)

async def initialize_embedding_service():
    """初始化Embedding服务"""
    global embedding_function, batch_processor_task, current_gpu_device

    # 获取当前 worker 分配的 GPU
    current_gpu_device = get_worker_gpu()
    
    print(f"🚀 初始化DISKANN Embedding服务...")
    print(f"🎮 当前Worker使用GPU: {current_gpu_device}")
    print(f"🔢 精度: FP32 (DISKANN要求)")

    try:
        embedding_function = BGEM3EmbeddingFunction(
            model_name=API_BGE_MODEL_PATH,
            batch_size=MAX_BATCH_SIZE,
            device=current_gpu_device,
            use_fp16=False,  # DISKANN使用FP32
            max_length=API_BGE_MAX_LENGTH,
            normalize_embeddings=True
        )

        test_result = embedding_function(["warmup"])
        test_dense = test_result["dense"][0]

        print(f"✅ 模型加载完成 (GPU: {current_gpu_device})")
        print(f"   输出类型: {test_dense.dtype if hasattr(test_dense, 'dtype') else 'list'}")

        if hasattr(test_dense, 'dtype'):
            if test_dense.dtype == np.float32:
                print(f"   🎯 FP32输出 ✅")
            else:
                print(f"   ⚠️  输出为{test_dense.dtype}，DISKANN需要FP32")

        batch_processor_task = asyncio.create_task(batch_processor())
        return True

    except Exception as e:
        print(f"❌ 初始化失败 (GPU: {current_gpu_device}): {e}")
        import traceback
        traceback.print_exc()
        return False

async def cleanup_service():
    """清理资源"""
    global batch_processor_task
    if batch_processor_task:
        batch_processor_task.cancel()
        try:
            await batch_processor_task
        except asyncio.CancelledError:
            pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    success = await initialize_embedding_service()
    if not success:
        raise RuntimeError("初始化失败")
    yield
    await cleanup_service()

app = FastAPI(
    title="DISKANN Embedding Service",
    description="FP32 embedding for DISKANN",
    version="2.0.0",
    lifespan=lifespan
)

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
    request_logger.info(
        f"Texts:{getattr(request.state, 'text_count', 'N/A')}",
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
        "message": "DISKANN Embedding服务",
        "version": "2.0.0",
        "vector_precision": "FP32",
        "max_batch_size": MAX_BATCH_SIZE
    }

@app.get("/health")
async def health_check():
    global embedding_function, batch_processor_task, current_gpu_device

    model_loaded = embedding_function is not None
    processor_running = batch_processor_task is not None and not batch_processor_task.done()

    return {
        "status": "healthy" if (model_loaded and processor_running) else "unhealthy",
        "model_loaded": model_loaded,
        "batch_processor_running": processor_running,
        "model_path": API_BGE_MODEL_PATH,
        "vector_dtype": "FP32",
        "queue_size": len(request_queue),
        "gpu_device": current_gpu_device,
        "pid": os.getpid()
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed_endpoint(request_data: EmbedRequest, request: Request):
    global request_queue, embedding_function

    if not embedding_function:
        raise HTTPException(status_code=503, detail="模型未加载")

    # 记录请求详情到request.state，供中间件使用
    request.state.text_count = len(request_data.texts)

    request_item = RequestItem(request_data.texts, request_data.return_dense, request_data.return_sparse)

    async with queue_lock:
        request_queue.append(request_item)

    try:
        response_data = await asyncio.wait_for(request_item.future, timeout=30)
        return EmbedResponse(**response_data)

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"失败: {str(e)}")

if __name__ == "__main__":
    # 清理上次的 GPU 计数器（确保从 GPU 0 开始分配）
    counter_file = os.path.join(tempfile.gettempdir(), "embedding_server_gpu_counter")
    if os.path.exists(counter_file):
        os.remove(counter_file)
    
    # 使用配置文件中的 workers 数量（和 RAG 服务一样是32）
    gpu_count = len(API_DEVICE) if isinstance(API_DEVICE, list) else 1
    workers_count = EMBEDDING_WORKERS  # 使用配置：32个workers
    workers_per_gpu = workers_count // gpu_count
    
    print()
    print("=" * 70)
    print("🤖 启动DISKANN Embedding服务（四卡高并发）")
    print("=" * 70)
    print(f"📡 端口: 8028")
    print(f"🎮 GPU设备: {API_DEVICE} ({gpu_count}张卡)")
    print(f"👥 Workers: {workers_count} (每张卡{workers_per_gpu}个Worker)")
    print(f"🎯 FP32精度")
    print()
    print("⚡ 架构说明:")
    print(f"   - {workers_count}个 Uvicorn Worker 进程")
    print(f"   - 每个 Worker 分配到一张 GPU（轮询分配）")
    print(f"   - 每张卡约 {workers_per_gpu} 个 Worker 共享")
    print(f"   - Worker内部有动态 Batching")
    print("   - RAG服务连接: http://localhost:8028")
    print("=" * 70)
    print()

    uvicorn.run(
        "embedding_server_diskann:app",
        host="0.0.0.0",
        port=8028,
        reload=False,
        workers=workers_count,
        log_level="info"
    )
