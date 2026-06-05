#!/usr/bin/env python3
"""
Browse 服务压测脚本

流程：
1. 调用 Search API 获取搜索结果
2. 提取 URL 列表
3. 并发请求 Browse 服务
4. 统计各阶段（SQL、爬虫、Summary）的时间

使用方法：
    python browse_benchmark.py --queries "What is AI" "Python programming" --concurrency 500
    python browse_benchmark.py --query-file queries.txt --concurrency 20
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

import aiohttp
import requests

# ============ 配置 ============
DEFAULT_SEARCH_URL = "http://localhost:8017/search"  # Search 服务地址
DEFAULT_BROWSE_URL = "http://localhost:8010/query"   # Browse 服务地址
DEFAULT_CONCURRENCY = 10  # 默认并发数
DEFAULT_SEARCH_LIMIT = 5  # 每个搜索返回的结果数

# ============ 日志配置 ============
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(LOG_DIR, f"benchmark_{timestamp_str}.log")
result_file = os.path.join(LOG_DIR, f"benchmark_{timestamp_str}_result.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BrowseResult:
    """单个 Browse 请求的结果"""
    url: str
    goal: str
    success: bool
    from_cache: bool
    total_time: float  # 总耗时
    error: str = ""
    response_length: int = 0


@dataclass
class BenchmarkStats:
    """压测统计结果"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # 时间统计（秒）
    total_times: List[float] = field(default_factory=list)
    cache_hit_times: List[float] = field(default_factory=list)
    cache_miss_times: List[float] = field(default_factory=list)
    
    # 按阶段统计（从 Browse 服务日志推断）
    search_time: float = 0  # Search 阶段总时间
    browse_start_time: float = 0  # Browse 开始时间
    browse_end_time: float = 0  # Browse 结束时间
    
    def add_result(self, result: BrowseResult):
        self.total_requests += 1
        if result.success:
            self.successful_requests += 1
            self.total_times.append(result.total_time)
            if result.from_cache:
                self.cache_hits += 1
                self.cache_hit_times.append(result.total_time)
            else:
                self.cache_misses += 1
                self.cache_miss_times.append(result.total_time)
        else:
            self.failed_requests += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        def calc_stats(times: List[float]) -> Dict[str, float]:
            if not times:
                return {"count": 0, "avg": 0, "min": 0, "max": 0, "median": 0, "p95": 0, "p99": 0}
            sorted_times = sorted(times)
            return {
                "count": len(times),
                "avg": statistics.mean(times),
                "min": min(times),
                "max": max(times),
                "median": statistics.median(times),
                "p95": sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 1 else sorted_times[0],
                "p99": sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 1 else sorted_times[0],
            }
        
        return {
            "overview": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": f"{self.successful_requests / self.total_requests * 100:.2f}%" if self.total_requests > 0 else "0%",
                "cache_hit_rate": f"{self.cache_hits / self.successful_requests * 100:.2f}%" if self.successful_requests > 0 else "0%",
            },
            "timing": {
                "search_phase": f"{self.search_time:.2f}s",
                "browse_phase": f"{self.browse_end_time - self.browse_start_time:.2f}s" if self.browse_end_time else "N/A",
                "total_benchmark": f"{self.browse_end_time - self.search_time + self.browse_start_time:.2f}s" if self.browse_end_time else "N/A",
            },
            "all_requests": calc_stats(self.total_times),
            "cache_hits": calc_stats(self.cache_hit_times),
            "cache_misses": calc_stats(self.cache_miss_times),
        }


class BrowseBenchmark:
    """Browse 服务压测器"""
    
    def __init__(
        self,
        search_url: str = DEFAULT_SEARCH_URL,
        browse_url: str = DEFAULT_BROWSE_URL,
        concurrency: int = DEFAULT_CONCURRENCY,
        search_limit: int = DEFAULT_SEARCH_LIMIT,
    ):
        self.search_url = search_url
        self.browse_url = browse_url
        self.concurrency = concurrency
        self.search_limit = search_limit
        self.stats = BenchmarkStats()
        self.results: List[BrowseResult] = []
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """调用 Search API 获取搜索结果"""
        logger.info(f"🔍 搜索: {query}")
        
        payload = {
            "query": query,
            "search_type": "hybrid",
            "limit": self.search_limit
        }
        
        try:
            resp = requests.post(
                self.search_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            
            # 提取结果
            results = data.get("results", [])
            logger.info(f"✅ 搜索完成 | 查询: {query[:50]}... | 结果数: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"❌ 搜索失败 | 查询: {query[:50]}... | 错误: {e}")
            return []
    
    def search_batch(self, queries: List[str]) -> List[Dict[str, str]]:
        """
        批量搜索，返回 URL 和对应的 goal 列表
        
        Returns:
            List[Dict]: [{"url": "...", "goal": "..."}]
        """
        logger.info(f"📊 开始批量搜索 | 查询数: {len(queries)}")
        search_start = time.time()
        
        url_goal_pairs = []
        seen_urls = set()  # 去重
        
        for query in queries:
            results = self.search(query)
            for result in results:
                url = result.get("link", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    url_goal_pairs.append({
                        "url": url,
                        "goal": query,
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", "")
                    })
        
        search_elapsed = time.time() - search_start
        self.stats.search_time = search_elapsed
        logger.info(f"✅ 批量搜索完成 | 耗时: {search_elapsed:.2f}s | URL 数: {len(url_goal_pairs)}")
        
        return url_goal_pairs
    
    async def browse_single(
        self,
        session: aiohttp.ClientSession,
        url: str,
        goal: str,
        semaphore: asyncio.Semaphore,
    ) -> BrowseResult:
        """单个 Browse 请求"""
        async with semaphore:
            start_time = time.time()
            
            try:
                payload = {"url": url, "goal": goal}
                async with session.post(
                    self.browse_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    elapsed = time.time() - start_time
                    
                    if resp.status == 200:
                        data = await resp.json()
                        result = BrowseResult(
                            url=url,
                            goal=goal,
                            success=data.get("success", False),
                            from_cache=data.get("from_cache", False),
                            total_time=elapsed,
                            response_length=len(data.get("result", "")),
                            error=data.get("error", "")
                        )
                        
                        cache_status = "💾 缓存" if result.from_cache else "🌐 爬取"
                        logger.info(f"{cache_status} | {elapsed:.2f}s | {url[:60]}...")
                        return result
                    else:
                        error_text = await resp.text()
                        logger.error(f"❌ HTTP {resp.status} | {url[:60]}... | {error_text[:100]}")
                        return BrowseResult(
                            url=url,
                            goal=goal,
                            success=False,
                            from_cache=False,
                            total_time=elapsed,
                            error=f"HTTP {resp.status}: {error_text[:100]}"
                        )
                        
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.error(f"❌ 超时 | {elapsed:.2f}s | {url[:60]}...")
                return BrowseResult(
                    url=url,
                    goal=goal,
                    success=False,
                    from_cache=False,
                    total_time=elapsed,
                    error="Timeout"
                )
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ 异常 | {elapsed:.2f}s | {url[:60]}... | {e}")
                return BrowseResult(
                    url=url,
                    goal=goal,
                    success=False,
                    from_cache=False,
                    total_time=elapsed,
                    error=str(e)
                )
    
    async def browse_batch(self, url_goal_pairs: List[Dict[str, str]]) -> List[BrowseResult]:
        """批量 Browse 请求"""
        logger.info(f"📊 开始批量 Browse | URL 数: {len(url_goal_pairs)} | 并发: {self.concurrency}")
        self.stats.browse_start_time = time.time()
        
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.browse_single(session, pair["url"], pair["goal"], semaphore)
                for pair in url_goal_pairs
            ]
            results = await asyncio.gather(*tasks)
        
        self.stats.browse_end_time = time.time()
        browse_elapsed = self.stats.browse_end_time - self.stats.browse_start_time
        
        # 统计结果
        for result in results:
            self.stats.add_result(result)
            self.results.append(result)
        
        logger.info(f"✅ 批量 Browse 完成 | 耗时: {browse_elapsed:.2f}s")
        return results
    
    def run(self, queries: List[str]) -> Dict[str, Any]:
        """
        运行完整压测流程
        
        Args:
            queries: 搜索查询列表
            
        Returns:
            统计结果字典
        """
        logger.info("=" * 60)
        logger.info(f"🚀 开始压测")
        logger.info(f"   Search URL: {self.search_url}")
        logger.info(f"   Browse URL: {self.browse_url}")
        logger.info(f"   并发数: {self.concurrency}")
        logger.info(f"   查询数: {len(queries)}")
        logger.info("=" * 60)
        
        total_start = time.time()
        
        # 1. 批量搜索
        url_goal_pairs = self.search_batch(queries)
        
        if not url_goal_pairs:
            logger.error("❌ 没有获取到任何 URL，压测终止")
            return {"error": "No URLs found"}
        
        # 2. 批量 Browse
        asyncio.run(self.browse_batch(url_goal_pairs))
        
        total_elapsed = time.time() - total_start
        
        # 3. 生成报告
        summary = self.stats.get_summary()
        summary["benchmark_config"] = {
            "search_url": self.search_url,
            "browse_url": self.browse_url,
            "concurrency": self.concurrency,
            "search_limit": self.search_limit,
            "queries": queries,
            "total_urls": len(url_goal_pairs),
        }
        summary["total_benchmark_time"] = f"{total_elapsed:.2f}s"
        
        # 打印报告
        self.print_report(summary)
        
        # 保存结果
        self.save_results(summary)
        
        return summary
    
    def print_report(self, summary: Dict[str, Any]):
        """打印压测报告"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("📊 压测报告")
        logger.info("=" * 60)
        
        overview = summary["overview"]
        logger.info(f"总请求数: {overview['total_requests']}")
        logger.info(f"成功请求: {overview['successful_requests']}")
        logger.info(f"失败请求: {overview['failed_requests']}")
        logger.info(f"成功率: {overview['success_rate']}")
        logger.info(f"缓存命中率: {overview['cache_hit_rate']}")
        
        logger.info("")
        logger.info("⏱️  时间统计")
        logger.info("-" * 40)
        
        timing = summary["timing"]
        logger.info(f"Search 阶段: {timing['search_phase']}")
        logger.info(f"Browse 阶段: {timing['browse_phase']}")
        
        logger.info("")
        logger.info("📈 响应时间统计（秒）")
        logger.info("-" * 40)
        
        all_stats = summary["all_requests"]
        logger.info(f"全部请求:")
        logger.info(f"  数量: {all_stats['count']}")
        logger.info(f"  平均: {all_stats['avg']:.2f}s")
        logger.info(f"  中位: {all_stats['median']:.2f}s")
        logger.info(f"  最小: {all_stats['min']:.2f}s")
        logger.info(f"  最大: {all_stats['max']:.2f}s")
        logger.info(f"  P95: {all_stats['p95']:.2f}s")
        logger.info(f"  P99: {all_stats['p99']:.2f}s")
        
        cache_hit_stats = summary["cache_hits"]
        if cache_hit_stats["count"] > 0:
            logger.info(f"\n缓存命中（SQL 阶段）:")
            logger.info(f"  数量: {cache_hit_stats['count']}")
            logger.info(f"  平均: {cache_hit_stats['avg']:.2f}s")
            logger.info(f"  中位: {cache_hit_stats['median']:.2f}s")
        
        cache_miss_stats = summary["cache_misses"]
        if cache_miss_stats["count"] > 0:
            logger.info(f"\n缓存未命中（爬虫 + Summary）:")
            logger.info(f"  数量: {cache_miss_stats['count']}")
            logger.info(f"  平均: {cache_miss_stats['avg']:.2f}s")
            logger.info(f"  中位: {cache_miss_stats['median']:.2f}s")
        
        logger.info("")
        logger.info(f"📁 详细结果已保存到: {result_file}")
        logger.info("=" * 60)
    
    def save_results(self, summary: Dict[str, Any]):
        """保存结果到 JSON 文件"""
        output = {
            "summary": summary,
            "results": [
                {
                    "url": r.url,
                    "goal": r.goal,
                    "success": r.success,
                    "from_cache": r.from_cache,
                    "total_time": r.total_time,
                    "response_length": r.response_length,
                    "error": r.error,
                }
                for r in self.results
            ]
        }
        
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 结果已保存到: {result_file}")


def main():
    parser = argparse.ArgumentParser(description="Browse 服务压测脚本")
    
    parser.add_argument(
        "--queries",
        nargs="+",
        help="搜索查询列表"
    )
    parser.add_argument(
        "--query-file",
        type=str,
        help="查询文件路径（每行一个查询）"
    )
    parser.add_argument(
        "--search-url",
        type=str,
        default=DEFAULT_SEARCH_URL,
        help=f"Search 服务地址（默认: {DEFAULT_SEARCH_URL}）"
    )
    parser.add_argument(
        "--browse-url",
        type=str,
        default=DEFAULT_BROWSE_URL,
        help=f"Browse 服务地址（默认: {DEFAULT_BROWSE_URL}）"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"并发请求数（默认: {DEFAULT_CONCURRENCY}）"
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"每个搜索返回的结果数（默认: {DEFAULT_SEARCH_LIMIT}）"
    )
    
    args = parser.parse_args()
    
    # 获取查询列表
    queries = []
    
    if args.queries:
        queries = args.queries
    elif args.query_file:
        if os.path.exists(args.query_file):
            with open(args.query_file, "r", encoding="utf-8") as f:
                queries = [line.strip() for line in f if line.strip()]
        else:
            logger.error(f"❌ 查询文件不存在: {args.query_file}")
            sys.exit(1)
    else:
        # 默认测试查询
        queries = [
            "What is artificial intelligence?",
            "Python programming language",
            "Machine learning algorithms",
            "Deep learning neural networks",
            "Natural language processing",
        ]
        logger.info(f"⚠️  未指定查询，使用默认测试查询 ({len(queries)} 个)")
    
    # 运行压测
    benchmark = BrowseBenchmark(
        search_url=args.search_url,
        browse_url=args.browse_url,
        concurrency=args.concurrency,
        search_limit=args.search_limit,
    )
    
    benchmark.run(queries)


if __name__ == "__main__":
    main()

