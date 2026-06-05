#!/usr/bin/env python3
"""
🔥 Embedding 服务压测脚本
模拟 agentloop 中大量并发 embedding 请求的场景

使用方式:
    python benchmark_embedding.py --concurrency 100 --total 1000
    python benchmark_embedding.py --concurrency 200 --total 5000 --batch-size 8
"""

import asyncio
import aiohttp
import argparse
import time
import random
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict
import statistics

# ==================== 配置 ====================

# Embedding 服务地址
EMBEDDING_SERVICE_URL = "http://10.160.199.231:8028/embed"

# 测试查询池（模拟真实场景的多样性）
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
    error: Optional[str] = None
    queue_wait_time: float = 0.0
    model_inference_time: float = 0.0
    batch_size: int = 1


@dataclass
class BenchmarkStats:
    """压测统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0
    latencies: List[float] = field(default_factory=list)
    queue_wait_times: List[float] = field(default_factory=list)
    inference_times: List[float] = field(default_factory=list)
    errors: dict = field(default_factory=lambda: defaultdict(int))
    
    def add_result(self, result: RequestResult):
        self.total_requests += 1
        if result.success:
            self.successful_requests += 1
            self.latencies.append(result.latency)
            self.queue_wait_times.append(result.queue_wait_time)
            self.inference_times.append(result.model_inference_time)
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
        print("📊 Embedding 服务压测结果")
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
            print(f"\n⏱️  端到端延迟 (ms):")
            print(f"   平均: {statistics.mean(self.latencies) * 1000:.2f}")
            print(f"   中位数: {statistics.median(self.latencies) * 1000:.2f}")
            print(f"   P50: {self.get_percentile(self.latencies, 50) * 1000:.2f}")
            print(f"   P95: {self.get_percentile(self.latencies, 95) * 1000:.2f}")
            print(f"   P99: {self.get_percentile(self.latencies, 99) * 1000:.2f}")
            print(f"   最小: {min(self.latencies) * 1000:.2f}")
            print(f"   最大: {max(self.latencies) * 1000:.2f}")
        
        if self.queue_wait_times and any(t > 0 for t in self.queue_wait_times):
            print(f"\n⏳ 队列等待时间 (ms):")
            print(f"   平均: {statistics.mean(self.queue_wait_times) * 1000:.2f}")
            print(f"   P95: {self.get_percentile(self.queue_wait_times, 95) * 1000:.2f}")
            print(f"   P99: {self.get_percentile(self.queue_wait_times, 99) * 1000:.2f}")
        
        if self.inference_times and any(t > 0 for t in self.inference_times):
            print(f"\n🤖 模型推理时间 (ms):")
            print(f"   平均: {statistics.mean(self.inference_times) * 1000:.2f}")
            print(f"   P95: {self.get_percentile(self.inference_times, 95) * 1000:.2f}")
            print(f"   P99: {self.get_percentile(self.inference_times, 99) * 1000:.2f}")
        
        if self.errors:
            print(f"\n❌ 错误统计:")
            for error, count in sorted(self.errors.items(), key=lambda x: -x[1]):
                print(f"   {error}: {count}")
        
        print("=" * 80)


class EmbeddingBenchmark:
    """Embedding 服务压测器"""
    
    def __init__(
        self,
        url: str = EMBEDDING_SERVICE_URL,
        concurrency: int = 100,
        total_requests: int = 1000,
        batch_size: int = 1,
        timeout: float = 30.0,
    ):
        self.url = url
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.batch_size = batch_size
        self.timeout = timeout
        self.stats = BenchmarkStats()
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.progress_lock = asyncio.Lock()
        self.completed = 0
    
    def get_random_queries(self, count: int = 1) -> List[str]:
        """获取随机查询"""
        return [random.choice(TEST_QUERIES) for _ in range(count)]
    
    async def single_request(self, request_id: int) -> RequestResult:
        """发送单个 embedding 请求（模拟 agentloop 中的工具调用）"""
        start_time = time.time()
        
        try:
            # 获取信号量（控制并发）
            async with self.semaphore:
                queries = self.get_random_queries(self.batch_size)
                
                async with self.session.post(
                    self.url,
                    json={
                        "texts": queries,
                        "return_dense": True,
                        "return_sparse": True,
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    latency = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        return RequestResult(
                            success=True,
                            latency=latency,
                            queue_wait_time=data.get("queue_wait_time", 0),
                            model_inference_time=data.get("model_inference_time", 0),
                            batch_size=len(queries),
                        )
                    else:
                        return RequestResult(
                            success=False,
                            latency=latency,
                            error=f"HTTP_{response.status}",
                        )
        
        except asyncio.TimeoutError:
            return RequestResult(
                success=False,
                latency=time.time() - start_time,
                error="Timeout",
            )
        except aiohttp.ClientError as e:
            return RequestResult(
                success=False,
                latency=time.time() - start_time,
                error=f"ClientError: {type(e).__name__}",
            )
        except Exception as e:
            return RequestResult(
                success=False,
                latency=time.time() - start_time,
                error=f"Exception: {type(e).__name__}",
            )
        finally:
            # 更新进度
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
        这样可以真正模拟 agentloop 中源源不断发请求的场景。
        """
        print("\n" + "=" * 80)
        print("🔥 Embedding 服务压测开始 (全异步模式)")
        print("=" * 80)
        print(f"📡 目标服务: {self.url}")
        print(f"🔢 总请求数: {self.total_requests}")
        print(f"⚡ 并发数: {self.concurrency}")
        print(f"📦 每请求文本数: {self.batch_size}")
        print(f"⏱️  超时: {self.timeout}s")
        print("-" * 80)
        
        # 初始化
        self.semaphore = asyncio.Semaphore(self.concurrency)
        connector = aiohttp.TCPConnector(
            limit=self.concurrency * 2,
            limit_per_host=self.concurrency * 2,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True,
        )
        self.session = aiohttp.ClientSession(connector=connector)
        
        try:
            # 预热
            print("\n🔥 预热中...")
            warmup_result = await self.single_request(-1)
            if warmup_result.success:
                print(f"   ✅ 预热成功 ({warmup_result.latency*1000:.1f}ms)")
            else:
                print(f"   ⚠️ 预热失败: {warmup_result.error}")
            
            # 重置计数
            self.completed = 0
            
            # 开始压测 - 全异步模式
            print(f"\n🚀 开始全异步压测 ({self.total_requests} 请求, {self.concurrency} 并发)...")
            self.start_time = time.time()
            
            # 🔥 全异步：一次性创建所有任务，让 asyncio 调度
            # 信号量会自动控制同时运行的请求数量
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
            await self.session.close()


async def main():
    parser = argparse.ArgumentParser(description="Embedding 服务压测")
    parser.add_argument("--url", type=str, default=EMBEDDING_SERVICE_URL, help="Embedding 服务地址")
    parser.add_argument("--concurrency", "-c", type=int, default=100, help="并发数")
    parser.add_argument("--total", "-n", type=int, default=1000, help="总请求数")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="每个请求包含的文本数")
    parser.add_argument("--timeout", "-t", type=float, default=30.0, help="请求超时时间(秒)")
    
    args = parser.parse_args()
    
    benchmark = EmbeddingBenchmark(
        url=args.url,
        concurrency=args.concurrency,
        total_requests=args.total,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    
    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())

