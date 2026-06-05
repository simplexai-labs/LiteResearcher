#!/usr/bin/env python3
"""
🔥 完整查询流程压测脚本（Embedding + Milvus）
模拟 agentloop 中 search 工具的真实调用场景

使用方式:
    python benchmark_query.py --concurrency 100 --total 1000
    python benchmark_query.py --concurrency 200 --total 5000 --search-type hybrid
    
    # 模拟 rollout 批量并发模式（8 workers，每 worker 16 samples，每 sample 2 个工具调用）
    python benchmark_query.py --rollout-mode --workers 8 --samples-per-worker 16 --tools-per-sample 2
    
这个脚本最接近实际 rollout 中的调用模式：
1. 使用 asyncio.gather 并发执行请求
2. 支持分层并发（workers -> samples -> tool calls）
3. 统计端到端延迟和各阶段耗时
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

# ==================== 配置 ====================

# RAG 服务地址（完整查询）
RAG_SERVICE_URL = "http://localhost:8018/search"

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
    "What are the advantages of convolutional neural networks?",
    "循环神经网络的工作原理",
    "How does backpropagation work?",
    "梯度下降算法的优化方法",
    "What is the difference between AI and ML?",
    "监督学习和无监督学习的区别",
    "How to prevent overfitting in deep learning?",
    "数据增强技术在图像分类中的应用",
    "What is semantic segmentation?",
    "目标检测算法的发展历程",
]


@dataclass
class RequestResult:
    """单个请求的结果"""
    success: bool
    total_latency: float  # 端到端延迟（秒）
    embedding_time: float = 0.0  # Embedding 耗时
    milvus_time: float = 0.0  # Milvus 搜索耗时
    num_results: int = 0
    error: Optional[str] = None
    query: str = ""


@dataclass
class BenchmarkStats:
    """压测统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0
    total_latencies: List[float] = field(default_factory=list)
    embedding_times: List[float] = field(default_factory=list)
    milvus_times: List[float] = field(default_factory=list)
    result_counts: List[int] = field(default_factory=list)
    errors: dict = field(default_factory=lambda: defaultdict(int))
    
    def add_result(self, result: RequestResult):
        self.total_requests += 1
        if result.success:
            self.successful_requests += 1
            self.total_latencies.append(result.total_latency)
            self.embedding_times.append(result.embedding_time)
            self.milvus_times.append(result.milvus_time)
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
        print("📊 完整查询流程压测结果")
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
        
        if self.total_latencies:
            print(f"\n⏱️  端到端延迟 (ms):")
            print(f"   平均: {statistics.mean(self.total_latencies) * 1000:.2f}")
            print(f"   中位数: {statistics.median(self.total_latencies) * 1000:.2f}")
            print(f"   P50: {self.get_percentile(self.total_latencies, 50) * 1000:.2f}")
            print(f"   P95: {self.get_percentile(self.total_latencies, 95) * 1000:.2f}")
            print(f"   P99: {self.get_percentile(self.total_latencies, 99) * 1000:.2f}")
            print(f"   最小: {min(self.total_latencies) * 1000:.2f}")
            print(f"   最大: {max(self.total_latencies) * 1000:.2f}")
        
        if self.embedding_times:
            print(f"\n🤖 Embedding 阶段 (ms):")
            print(f"   平均: {statistics.mean(self.embedding_times) * 1000:.2f}")
            print(f"   P95: {self.get_percentile(self.embedding_times, 95) * 1000:.2f}")
            print(f"   P99: {self.get_percentile(self.embedding_times, 99) * 1000:.2f}")
        
        if self.milvus_times:
            print(f"\n🔍 Milvus 搜索阶段 (ms):")
            print(f"   平均: {statistics.mean(self.milvus_times) * 1000:.2f}")
            print(f"   P95: {self.get_percentile(self.milvus_times, 95) * 1000:.2f}")
            print(f"   P99: {self.get_percentile(self.milvus_times, 99) * 1000:.2f}")
        
        # 时间占比分析
        if self.embedding_times and self.milvus_times and self.total_latencies:
            avg_total = statistics.mean(self.total_latencies)
            avg_embedding = statistics.mean(self.embedding_times)
            avg_milvus = statistics.mean(self.milvus_times)
            avg_other = max(0, avg_total - avg_embedding - avg_milvus)
            
            print(f"\n📊 时间占比分析:")
            print(f"   Embedding: {avg_embedding/avg_total*100:.1f}%")
            print(f"   Milvus: {avg_milvus/avg_total*100:.1f}%")
            print(f"   其他(网络等): {avg_other/avg_total*100:.1f}%")
        
        if self.result_counts:
            print(f"\n📄 返回结果数:")
            print(f"   平均: {statistics.mean(self.result_counts):.1f}")
        
        if self.errors:
            print(f"\n❌ 错误统计:")
            for error, count in sorted(self.errors.items(), key=lambda x: -x[1]):
                print(f"   {error}: {count}")
        
        print("=" * 80)


class QueryBenchmark:
    """完整查询流程压测器"""
    
    def __init__(
        self,
        url: str = RAG_SERVICE_URL,
        concurrency: int = 100,
        total_requests: int = 1000,
        search_type: str = "hybrid",
        limit: int = 10,
        sparse_weight: float = 0.7,
        dense_weight: float = 1.0,
        timeout: float = 60.0,
    ):
        self.url = url
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.search_type = search_type
        self.limit = limit
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.timeout = timeout
        self.stats = BenchmarkStats()
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.progress_lock = asyncio.Lock()
        self.completed = 0
        self.start_time = 0
    
    def get_random_query(self) -> str:
        """获取随机查询"""
        return random.choice(TEST_QUERIES)
    
    async def single_request(self, request_id: int) -> RequestResult:
        """发送单个查询请求（模拟 agentloop 中的 search 工具调用）"""
        start_time = time.time()
        query = self.get_random_query()
        
        try:
            async with self.semaphore:
                async with self.session.post(
                    self.url,
                    json={
                        "query": query,
                        "limit": self.limit,
                        "search_type": self.search_type,
                        "sparse_weight": self.sparse_weight,
                        "dense_weight": self.dense_weight,
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    total_latency = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        return RequestResult(
                            success=True,
                            total_latency=total_latency,
                            embedding_time=data.get("embedding_time", 0),
                            milvus_time=data.get("milvus_time", 0),
                            num_results=data.get("total", 0),
                            query=query,
                        )
                    else:
                        return RequestResult(
                            success=False,
                            total_latency=total_latency,
                            error=f"HTTP_{response.status}",
                            query=query,
                        )
        
        except asyncio.TimeoutError:
            return RequestResult(
                success=False,
                total_latency=time.time() - start_time,
                error="Timeout",
                query=query,
            )
        except aiohttp.ClientError as e:
            return RequestResult(
                success=False,
                total_latency=time.time() - start_time,
                error=f"ClientError: {type(e).__name__}",
                query=query,
            )
        except Exception as e:
            return RequestResult(
                success=False,
                total_latency=time.time() - start_time,
                error=f"Exception: {type(e).__name__}",
                query=query,
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
        print("🔥 完整查询流程压测开始 (全异步模式)")
        print("=" * 80)
        print(f"📡 目标服务: {self.url}")
        print(f"🔍 搜索类型: {self.search_type}")
        print(f"🔢 总请求数: {self.total_requests}")
        print(f"⚡ 并发数: {self.concurrency}")
        print(f"📄 每次返回: {self.limit} 条")
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
                print(f"   ✅ 预热成功 (总延迟: {warmup_result.total_latency*1000:.1f}ms, "
                      f"Embedding: {warmup_result.embedding_time*1000:.1f}ms, "
                      f"Milvus: {warmup_result.milvus_time*1000:.1f}ms)")
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
                        total_latency=0,
                        error=f"Exception: {type(result).__name__}",
                    ))
                else:
                    self.stats.add_result(result)
            
            # 打印结果
            self.stats.print_summary()
            
        finally:
            await self.session.close()


class RolloutModeBenchmark:
    """
    模拟 Rollout 批量并发模式的压测器
    
    这个类模拟了真实 rollout 中的分层并发模式:
    - Layer 1: Workers 并发 (num_workers)
    - Layer 2: Samples 并发 (samples_per_worker)
    - Layer 3: Tool calls 并发 (tools_per_sample)
    
    类似于 agent_loop.py 中的:
        tasks = []
        for i in range(len(batch)):
            tasks.append(asyncio.create_task(self._run_agent_loop(...)))
        outputs = await asyncio.gather(*tasks)
    """
    
    def __init__(
        self,
        url: str = RAG_SERVICE_URL,
        num_workers: int = 8,
        samples_per_worker: int = 16,
        tools_per_sample: int = 2,
        num_turns: int = 5,
        search_type: str = "hybrid",
        limit: int = 10,
        timeout: float = 60.0,
    ):
        self.url = url
        self.num_workers = num_workers
        self.samples_per_worker = samples_per_worker
        self.tools_per_sample = tools_per_sample
        self.num_turns = num_turns
        self.search_type = search_type
        self.limit = limit
        self.timeout = timeout
        
        self.total_requests = num_workers * samples_per_worker * tools_per_sample * num_turns
        self.stats = BenchmarkStats()
        self.session: Optional[aiohttp.ClientSession] = None
        self.progress_lock = asyncio.Lock()
        self.completed = 0
        self.start_time = 0
    
    async def single_tool_call(self, worker_id: int, sample_id: int, turn: int, tool_idx: int) -> RequestResult:
        """模拟单个工具调用"""
        start_time = time.time()
        query = random.choice(TEST_QUERIES)
        
        try:
            async with self.session.post(
                self.url,
                json={
                    "query": query,
                    "limit": self.limit,
                    "search_type": self.search_type,
                    "sparse_weight": 0.7,
                    "dense_weight": 1.0,
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                total_latency = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    return RequestResult(
                        success=True,
                        total_latency=total_latency,
                        embedding_time=data.get("embedding_time", 0),
                        milvus_time=data.get("milvus_time", 0),
                        num_results=data.get("total", 0),
                    )
                else:
                    return RequestResult(
                        success=False,
                        total_latency=total_latency,
                        error=f"HTTP_{response.status}",
                    )
        except Exception as e:
            return RequestResult(
                success=False,
                total_latency=time.time() - start_time,
                error=f"{type(e).__name__}",
            )
        finally:
            async with self.progress_lock:
                self.completed += 1
    
    async def simulate_sample_turn(self, worker_id: int, sample_id: int, turn: int) -> List[RequestResult]:
        """
        模拟单个 sample 的一个 turn（并发调用多个工具）
        
        对应 tool_agent_loop.py 中的:
            tasks = []
            for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
                tasks.append(self._call_tool(...))
            responses = await asyncio.gather(*tasks)
        """
        tasks = [
            self.single_tool_call(worker_id, sample_id, turn, i)
            for i in range(self.tools_per_sample)
        ]
        return await asyncio.gather(*tasks)
    
    async def simulate_sample(self, worker_id: int, sample_id: int) -> List[RequestResult]:
        """模拟单个 sample 的完整对话（多个 turns）"""
        all_results = []
        for turn in range(self.num_turns):
            results = await self.simulate_sample_turn(worker_id, sample_id, turn)
            all_results.extend(results)
        return all_results
    
    async def simulate_worker(self, worker_id: int) -> List[RequestResult]:
        """
        模拟单个 worker 处理多个 samples
        
        对应 agent_loop.py 中的:
            tasks = []
            for i in range(len(batch)):
                tasks.append(asyncio.create_task(self._run_agent_loop(...)))
            outputs = await asyncio.gather(*tasks)
        """
        tasks = [
            self.simulate_sample(worker_id, i)
            for i in range(self.samples_per_worker)
        ]
        results = await asyncio.gather(*tasks)
        return [r for sublist in results for r in sublist]
    
    async def run(self):
        """运行 Rollout 模式压测"""
        print("\n" + "=" * 80)
        print("🔥 Rollout 模式压测开始")
        print("=" * 80)
        print(f"📡 目标服务: {self.url}")
        print(f"\n🏗️  Rollout 配置:")
        print(f"   Workers: {self.num_workers}")
        print(f"   Samples/Worker: {self.samples_per_worker}")
        print(f"   Tools/Sample/Turn: {self.tools_per_sample}")
        print(f"   Turns: {self.num_turns}")
        print(f"   总请求数: {self.total_requests}")
        print(f"\n🔢 并发层级:")
        print(f"   Layer 1 (Workers): {self.num_workers} 并发")
        print(f"   Layer 2 (Samples): {self.samples_per_worker} 并发/worker")
        print(f"   Layer 3 (Tools): {self.tools_per_sample} 并发/sample/turn")
        print(f"   理论最大并发: {self.num_workers * self.samples_per_worker * self.tools_per_sample}")
        print("-" * 80)
        
        # 初始化
        connector = aiohttp.TCPConnector(
            limit=self.num_workers * self.samples_per_worker * self.tools_per_sample * 2,
            limit_per_host=self.num_workers * self.samples_per_worker * self.tools_per_sample * 2,
            ttl_dns_cache=300,
        )
        self.session = aiohttp.ClientSession(connector=connector)
        
        try:
            # 预热
            print("\n🔥 预热中...")
            warmup_result = await self.single_tool_call(0, 0, 0, 0)
            if warmup_result.success:
                print(f"   ✅ 预热成功 ({warmup_result.total_latency*1000:.1f}ms)")
            else:
                print(f"   ⚠️ 预热失败: {warmup_result.error}")
            self.completed = 0
            
            # 开始压测
            print(f"\n🚀 开始 Rollout 模式压测...")
            self.start_time = time.time()
            
            # 模拟所有 workers 并发执行
            tasks = [
                self.simulate_worker(i)
                for i in range(self.num_workers)
            ]
            
            # 进度监控
            async def progress_monitor():
                while self.completed < self.total_requests:
                    elapsed = time.time() - self.start_time
                    qps = self.completed / elapsed if elapsed > 0 else 0
                    print(f"   进度: {self.completed}/{self.total_requests} ({self.completed/self.total_requests*100:.1f}%) | QPS: {qps:.1f}")
                    await asyncio.sleep(2)
            
            monitor_task = asyncio.create_task(progress_monitor())
            
            try:
                results = await asyncio.gather(*tasks)
                all_results = [r for sublist in results for r in sublist]
            finally:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            
            self.stats.total_time = time.time() - self.start_time
            
            # 统计结果
            for result in all_results:
                self.stats.add_result(result)
            
            # 最终进度
            elapsed = time.time() - self.start_time
            qps = self.completed / elapsed if elapsed > 0 else 0
            print(f"   进度: {self.completed}/{self.total_requests} (100.0%) | QPS: {qps:.1f}")
            
            # 打印结果
            self.stats.print_summary()
            
        finally:
            await self.session.close()


async def main():
    parser = argparse.ArgumentParser(description="完整查询流程压测")
    parser.add_argument("--url", type=str, default=RAG_SERVICE_URL, help="RAG 服务地址")
    
    # 普通模式参数
    parser.add_argument("--concurrency", "-c", type=int, default=100, help="并发数")
    parser.add_argument("--total", "-n", type=int, default=1000, help="总请求数")
    
    # Rollout 模式参数
    parser.add_argument("--rollout-mode", "-r", action="store_true", help="使用 Rollout 模式")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Worker 数量")
    parser.add_argument("--samples-per-worker", type=int, default=16, help="每 Worker 的 Sample 数")
    parser.add_argument("--tools-per-sample", type=int, default=2, help="每 Sample 每 Turn 的工具调用数")
    parser.add_argument("--turns", type=int, default=5, help="每 Sample 的 Turn 数")
    
    # 搜索参数
    parser.add_argument("--search-type", "-s", type=str, default="hybrid",
                        choices=["hybrid", "dense", "sparse"], help="搜索类型")
    parser.add_argument("--limit", "-l", type=int, default=10, help="每次返回结果数")
    parser.add_argument("--timeout", "-t", type=float, default=60.0, help="请求超时时间(秒)")
    
    args = parser.parse_args()
    
    if args.rollout_mode:
        # Rollout 模式
        benchmark = RolloutModeBenchmark(
            url=args.url,
            num_workers=args.workers,
            samples_per_worker=args.samples_per_worker,
            tools_per_sample=args.tools_per_sample,
            num_turns=args.turns,
            search_type=args.search_type,
            limit=args.limit,
            timeout=args.timeout,
        )
    else:
        # 普通模式
        benchmark = QueryBenchmark(
            url=args.url,
            concurrency=args.concurrency,
            total_requests=args.total,
            search_type=args.search_type,
            limit=args.limit,
            timeout=args.timeout,
        )
    
    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())

