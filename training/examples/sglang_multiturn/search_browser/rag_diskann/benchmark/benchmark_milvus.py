#!/usr/bin/env python3
"""
🔥 Milvus 数据库压测脚本
模拟 agentloop 中大量并发向量搜索的场景

使用方式:
    python benchmark_milvus.py --concurrency 100 --total 1000
    python benchmark_milvus.py --concurrency 200 --total 5000 --search-type hybrid
    
注意: 此脚本会先从 Embedding 服务获取向量，然后压测 Milvus 搜索
"""

import asyncio
import aiohttp
import argparse
import time
import random
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import defaultdict
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

# 添加父目录到 path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diskann_config import (
    SEARCH_COLLECTION_NAME, API_MILVUS_URI,
    DISKANN_SEARCH_LIST
)

from pymilvus import connections, Collection

# ==================== 配置 ====================

# Embedding 服务地址（用于预生成向量）
EMBEDDING_SERVICE_URL = "http://10.160.199.231:8028/embed"

# 测试查询池
TEST_QUERIES = [
    "What is machine learning?",
    "深度学习的基本原理是什么",
    "How does transformer architecture work?",
    "自然语言处理的最新进展",
    "Explain the concept of neural networks",
    "人工智能在医疗领域的应用",
    "What is reinforcement learning?",
    "计算机视觉技术的发展历程",
    "How to train a large language model?",
    "知识图谱的构建方法",
    "Explain attention mechanism in deep learning",
    "推荐系统的算法原理",
    "What is federated learning?",
    "自动驾驶技术的挑战",
    "How does GPT-4 work?",
    "语音识别技术的演进",
    "What is transfer learning?",
    "图神经网络的应用场景",
    "Explain the BERT model architecture",
    "多模态学习的研究方向",
]


@dataclass
class RequestResult:
    """单个请求的结果"""
    success: bool
    latency: float  # 秒
    num_results: int = 0
    error: Optional[str] = None


@dataclass
class BenchmarkStats:
    """压测统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0
    latencies: List[float] = field(default_factory=list)
    result_counts: List[int] = field(default_factory=list)
    errors: dict = field(default_factory=lambda: defaultdict(int))
    
    def add_result(self, result: RequestResult):
        self.total_requests += 1
        if result.success:
            self.successful_requests += 1
            self.latencies.append(result.latency)
            self.result_counts.append(result.num_results)
        else:
            self.failed_requests += 1
            self.errors[result.error or "unknown"] += 1
    
    def get_percentile(self, data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print("📊 Milvus 数据库压测结果")
        print("=" * 80)
        
        # 基本统计
        qps = self.successful_requests / self.total_time if self.total_time > 0 else 0
        success_rate = self.successful_requests / max(self.total_requests, 1) * 100
        
        print(f"\n📈 总体统计:")
        print(f"   总请求数: {self.total_requests}")
        print(f"   成功请求: {self.successful_requests}")
        print(f"   失败请求: {self.failed_requests}")
        print(f"   成功率: {success_rate:.2f}%")
        print(f"   总耗时: {self.total_time:.2f}s")
        print(f"   QPS: {qps:.2f} req/s")
        
        if self.latencies:
            print(f"\n⏱️  搜索延迟 (ms):")
            print(f"   平均: {statistics.mean(self.latencies) * 1000:.2f}")
            print(f"   中位数: {statistics.median(self.latencies) * 1000:.2f}")
            print(f"   P50: {self.get_percentile(self.latencies, 50) * 1000:.2f}")
            print(f"   P95: {self.get_percentile(self.latencies, 95) * 1000:.2f}")
            print(f"   P99: {self.get_percentile(self.latencies, 99) * 1000:.2f}")
            print(f"   最小: {min(self.latencies) * 1000:.2f}")
            print(f"   最大: {max(self.latencies) * 1000:.2f}")
        
        if self.result_counts:
            print(f"\n📄 返回结果数:")
            print(f"   平均: {statistics.mean(self.result_counts):.1f}")
        
        if self.errors:
            print(f"\n❌ 错误统计:")
            for error, count in sorted(self.errors.items(), key=lambda x: -x[1]):
                print(f"   {error}: {count}")
        
        print("=" * 80)


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


class MilvusBenchmark:
    """Milvus 数据库压测器"""
    
    def __init__(
        self,
        concurrency: int = 100,
        total_requests: int = 1000,
        search_type: str = "hybrid",
        limit: int = 10,
        sparse_weight: float = 0.7,
        dense_weight: float = 1.0,
    ):
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.search_type = search_type
        self.limit = limit
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.stats = BenchmarkStats()
        self.collection: Optional[Collection] = None
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.progress_lock = asyncio.Lock()
        self.completed = 0
        self.start_time = 0
        
        # 预生成的向量缓存
        self.vector_cache: List[Dict[str, Any]] = []
    
    async def prepare_vectors(self, count: int = 100):
        """预先从 Embedding 服务获取向量"""
        print(f"\n🔄 预生成 {count} 个测试向量...")
        
        connector = aiohttp.TCPConnector(limit=50)
        async with aiohttp.ClientSession(connector=connector) as session:
            # 批量获取向量
            batch_size = 20
            for i in range(0, count, batch_size):
                queries = [random.choice(TEST_QUERIES) for _ in range(min(batch_size, count - i))]
                
                try:
                    async with session.post(
                        EMBEDDING_SERVICE_URL,
                        json={"texts": queries, "return_dense": True, "return_sparse": True},
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # 解析向量
                            dense_vectors = data.get("dense", [])
                            sparse_matrix = deserialize_sparse_matrix(data.get("sparse"))
                            
                            for j, query in enumerate(queries):
                                vector_data = {
                                    "query": query,
                                    "dense": np.array(dense_vectors[j], dtype=np.float32) if dense_vectors else None,
                                    "sparse": sparse_matrix[[j]] if sparse_matrix is not None else None,
                                }
                                self.vector_cache.append(vector_data)
                        else:
                            print(f"   ⚠️ 批次 {i//batch_size} 失败: HTTP {response.status}")
                except Exception as e:
                    print(f"   ⚠️ 批次 {i//batch_size} 异常: {e}")
                
                print(f"   进度: {len(self.vector_cache)}/{count}")
        
        print(f"   ✅ 预生成完成: {len(self.vector_cache)} 个向量")
    
    def get_random_vector(self) -> Dict[str, Any]:
        """获取随机向量"""
        return random.choice(self.vector_cache)
    
    def _execute_search(self, vector_data: Dict[str, Any]) -> RequestResult:
        """执行 Milvus 搜索（同步，在线程池中执行）"""
        from pymilvus import AnnSearchRequest, WeightedRanker
        
        start_time = time.time()
        
        try:
            query_dense = vector_data["dense"]
            query_sparse = vector_data["sparse"]
            
            search_params = {
                "metric_type": "IP",
                "params": {"search_list": DISKANN_SEARCH_LIST}
            }
            output_fields = ["url", "title", "doc"]
            
            def safe_sparse_check(sparse_data):
                if sparse_data is None:
                    return True
                if hasattr(sparse_data, '__len__') and len(sparse_data) == 0:
                    return True
                return False
            
            if self.search_type == "dense":
                results = self.collection.search(
                    [query_dense],
                    anns_field="dense_vector",
                    limit=self.limit,
                    output_fields=output_fields,
                    param=search_params,
                )[0]
            elif self.search_type == "sparse":
                if safe_sparse_check(query_sparse):
                    results = []
                else:
                    results = self.collection.search(
                        query_sparse,
                        anns_field="sparse_vector",
                        limit=self.limit,
                        output_fields=output_fields,
                        param={"metric_type": "IP", "params": {}},
                    )[0]
            else:  # hybrid
                if safe_sparse_check(query_sparse):
                    results = self.collection.search(
                        [query_dense],
                        anns_field="dense_vector",
                        limit=self.limit,
                        output_fields=output_fields,
                        param=search_params,
                    )[0]
                else:
                    dense_req = AnnSearchRequest(
                        [query_dense], "dense_vector", search_params, limit=self.limit
                    )
                    sparse_req = AnnSearchRequest(
                        query_sparse, "sparse_vector", {"metric_type": "IP", "params": {}}, limit=self.limit
                    )
                    rerank = WeightedRanker(self.sparse_weight, self.dense_weight)
                    
                    results = self.collection.hybrid_search(
                        [sparse_req, dense_req],
                        rerank=rerank,
                        limit=self.limit,
                        output_fields=output_fields
                    )[0]
            
            latency = time.time() - start_time
            return RequestResult(
                success=True,
                latency=latency,
                num_results=len(results) if hasattr(results, '__len__') else 0,
            )
        
        except Exception as e:
            return RequestResult(
                success=False,
                latency=time.time() - start_time,
                error=str(e)[:100],
            )
    
    async def single_request(self, request_id: int) -> RequestResult:
        """发送单个 Milvus 搜索请求"""
        loop = asyncio.get_event_loop()
        
        try:
            async with self.semaphore:
                vector_data = self.get_random_vector()
                result = await loop.run_in_executor(
                    self.executor,
                    self._execute_search,
                    vector_data,
                )
                return result
        
        except Exception as e:
            return RequestResult(
                success=False,
                latency=0,
                error=f"Exception: {type(e).__name__}",
            )
        finally:
            async with self.progress_lock:
                self.completed += 1
                if self.completed % 100 == 0 or self.completed == self.total_requests:
                    elapsed = time.time() - self.start_time
                    qps = self.completed / elapsed if elapsed > 0 else 0
                    print(f"   进度: {self.completed}/{self.total_requests} ({self.completed/self.total_requests*100:.1f}%) | QPS: {qps:.1f}")
    
    async def run(self):
        """
        运行压测 - 全异步模式
        
        所有请求同时发出，仅靠信号量控制实际并发数。
        """
        print("\n" + "=" * 80)
        print("🔥 Milvus 数据库压测开始 (全异步模式)")
        print("=" * 80)
        print(f"📊 集合: {SEARCH_COLLECTION_NAME}")
        print(f"🔍 搜索类型: {self.search_type}")
        print(f"🔢 总请求数: {self.total_requests}")
        print(f"⚡ 并发数: {self.concurrency}")
        print(f"📄 每次返回: {self.limit} 条")
        print("-" * 80)
        
        # 连接 Milvus
        print("\n🔗 连接 Milvus...")
        connections.connect(uri=API_MILVUS_URI)
        self.collection = Collection(SEARCH_COLLECTION_NAME)
        self.collection.load()
        print(f"   ✅ 集合加载完成: {self.collection.num_entities:,} 条文档")
        
        # 预生成向量
        num_vectors = min(100, self.total_requests)
        await self.prepare_vectors(num_vectors)
        
        if not self.vector_cache:
            print("❌ 无法预生成向量，请检查 Embedding 服务")
            return
        
        # 初始化
        self.semaphore = asyncio.Semaphore(self.concurrency)
        self.executor = ThreadPoolExecutor(max_workers=self.concurrency)
        
        try:
            # 预热
            print("\n🔥 预热中...")
            warmup_result = await self.single_request(-1)
            if warmup_result.success:
                print(f"   ✅ 预热成功 ({warmup_result.latency*1000:.1f}ms, {warmup_result.num_results} 结果)")
            else:
                print(f"   ⚠️ 预热失败: {warmup_result.error}")
            
            # 重置计数
            self.completed = 0
            
            # 开始压测 - 全异步模式
            print(f"\n🚀 开始全异步压测 ({self.total_requests} 请求, {self.concurrency} 并发)...")
            self.start_time = time.time()
            
            # 🔥 全异步：一次性创建所有任务
            tasks = [
                asyncio.create_task(self.single_request(i))
                for i in range(self.total_requests)
            ]
            
            # 等待所有任务完成
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            self.stats.total_time = time.time() - self.start_time
            
            # 统计结果
            for result in all_results:
                if isinstance(result, Exception):
                    self.stats.add_result(RequestResult(
                        success=False,
                        latency=0,
                        error=f"Exception: {type(result).__name__}",
                    ))
                else:
                    self.stats.add_result(result)
            
            # 打印结果
            self.stats.print_summary()
            
        finally:
            self.executor.shutdown(wait=False)


async def main():
    parser = argparse.ArgumentParser(description="Milvus 数据库压测")
    parser.add_argument("--concurrency", "-c", type=int, default=100, help="并发数")
    parser.add_argument("--total", "-n", type=int, default=1000, help="总请求数")
    parser.add_argument("--search-type", "-s", type=str, default="hybrid", 
                        choices=["hybrid", "dense", "sparse"], help="搜索类型")
    parser.add_argument("--limit", "-l", type=int, default=10, help="每次返回结果数")
    parser.add_argument("--sparse-weight", type=float, default=0.7, help="稀疏向量权重")
    parser.add_argument("--dense-weight", type=float, default=1.0, help="密集向量权重")
    
    args = parser.parse_args()
    
    benchmark = MilvusBenchmark(
        concurrency=args.concurrency,
        total_requests=args.total,
        search_type=args.search_type,
        limit=args.limit,
        sparse_weight=args.sparse_weight,
        dense_weight=args.dense_weight,
    )
    
    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())

