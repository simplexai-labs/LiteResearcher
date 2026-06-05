#!/usr/bin/env python3
"""
Browse 服务详细压测脚本

特点：
1. 支持从 Search API 获取 URL，或直接提供 URL 列表
2. 统计各阶段时间：SQL 查询、爬虫、Summary
3. 支持实时监控 Browse 服务的 /stats 接口
4. 生成详细的统计报告

使用方法：
    # 从 Search 获取 URL 进行压测
    python browse_benchmark_detailed.py --queries "AI" "Python" --concurrency 20
    
    # 直接提供 URL 列表
    python browse_benchmark_detailed.py --urls-file urls.txt --goal "Find information" --concurrency 20
    
    # 使用默认测试
    python browse_benchmark_detailed.py


    python browse_benchmark_detailed.py --query-file sample_queries.txt --concurrency 500
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import statistics
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import requests

# ============ 配置 ============
DEFAULT_SEARCH_URL = "http://47.111.147.142:8010/search"
DEFAULT_BROWSE_URL = "http://localhost:8010/query"
DEFAULT_BROWSE_STATS_URL = "http://localhost:8010/stats"
DEFAULT_CONCURRENCY = 500  # 默认并发数，已更新为500
DEFAULT_NUM_REQUESTS = 2000 # 默认总请求数

# ============ 日志配置 ============
DEFAULT_SEARCH_LIMIT = 5

# ============ 日志配置 ============
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(LOG_DIR, f"benchmark_detailed_{timestamp_str}.log")
result_file = os.path.join(LOG_DIR, f"benchmark_detailed_{timestamp_str}_result.json")
csv_file = os.path.join(LOG_DIR, f"benchmark_detailed_{timestamp_str}.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PhaseTime:
    """单个阶段的时间统计"""
    phase: str
    start_time: float = 0
    end_time: float = 0
    duration: float = 0
    
    def complete(self, end_time: float = None):
        self.end_time = end_time or time.time()
        self.duration = self.end_time - self.start_time


@dataclass
class BrowseResult:
    """单个 Browse 请求的详细结果"""
    url: str
    goal: str
    request_id: int
    
    # 状态
    success: bool = False
    from_cache: bool = False
    error: str = ""
    
    # 时间（秒）
    total_time: float = 0
    
    # 响应信息
    response_length: int = 0
    evidence_length: int = 0
    summary_length: int = 0
    
    # 推断的阶段时间（基于 from_cache）
    estimated_sql_time: float = 0
    estimated_crawler_time: float = 0
    estimated_summary_time: float = 0


@dataclass
class ServiceStats:
    """服务状态快照"""
    timestamp: float
    total_running: int = 0
    total_queued: int = 0
    sql_running: int = 0
    sql_queued: int = 0
    crawler_running: int = 0
    crawler_queued: int = 0
    summary_running: int = 0
    summary_queued: int = 0


class StatsMonitor:
    """实时监控 Browse 服务状态"""
    
    def __init__(self, stats_url: str, interval: float = 0.5):
        self.stats_url = stats_url
        self.interval = interval
        self.snapshots: List[ServiceStats] = []
        self._running = False
        self._thread = None
    
    def start(self):
        """开始监控"""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"📊 状态监控已启动 | URL: {self.stats_url} | 间隔: {self.interval}s")
    
    def stop(self):
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info(f"📊 状态监控已停止 | 收集快照: {len(self.snapshots)}")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                resp = requests.get(self.stats_url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json().get("requests", {})
                    snapshot = ServiceStats(
                        timestamp=time.time(),
                        total_running=data.get("total", {}).get("running", 0),
                        total_queued=data.get("total", {}).get("queued", 0),
                        sql_running=data.get("sql", {}).get("running", 0),
                        sql_queued=data.get("sql", {}).get("queued", 0),
                        crawler_running=data.get("crawler", {}).get("running", 0),
                        crawler_queued=data.get("crawler", {}).get("queued", 0),
                        summary_running=data.get("summary", {}).get("running", 0),
                        summary_queued=data.get("summary", {}).get("queued", 0),
                    )
                    self.snapshots.append(snapshot)
            except Exception as e:
                pass  # 忽略监控错误
            
            time.sleep(self.interval)
    
    def get_peak_stats(self) -> Dict[str, int]:
        """获取峰值统计"""
        if not self.snapshots:
            return {}
        
        return {
            "peak_total_running": max(s.total_running for s in self.snapshots),
            "peak_total_queued": max(s.total_queued for s in self.snapshots),
            "peak_sql_running": max(s.sql_running for s in self.snapshots),
            "peak_crawler_running": max(s.crawler_running for s in self.snapshots),
            "peak_summary_running": max(s.summary_running for s in self.snapshots),
            "total_snapshots": len(self.snapshots),
        }


class DetailedBenchmark:
    """详细压测器"""
    
    def __init__(
        self,
        search_url: str = DEFAULT_SEARCH_URL,
        browse_url: str = DEFAULT_BROWSE_URL,
        stats_url: str = DEFAULT_BROWSE_STATS_URL,
        concurrency: int = DEFAULT_CONCURRENCY,
        search_limit: int = DEFAULT_SEARCH_LIMIT,
        enable_monitoring: bool = True,
        num_requests: int = DEFAULT_NUM_REQUESTS,
    ):
        self.search_url = search_url
        self.browse_url = browse_url
        self.stats_url = stats_url
        self.concurrency = concurrency
        self.search_limit = search_limit
        self.enable_monitoring = enable_monitoring
        self.num_requests = num_requests
        
        self.results: List[BrowseResult] = []
        self.monitor = StatsMonitor(stats_url) if enable_monitoring else None
        
        # 时间记录
        self.search_start_time: float = 0
        self.search_end_time: float = 0
        self.browse_start_time: float = 0
        self.browse_end_time: float = 0
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """调用 Search API"""
        try:
            resp = requests.post(
                self.search_url,
                json={"query": query, "search_type": "hybrid", "limit": self.search_limit},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            logger.info(f"🔍 搜索完成 | {query[:40]}... | 结果: {len(results)}")
            return results
        except Exception as e:
            logger.error(f"❌ 搜索失败 | {query[:40]}... | {e}")
            return []
    
    def search_batch(self, queries: List[str]) -> List[Dict[str, str]]:
        """批量搜索"""
        logger.info(f"📊 开始批量搜索 | 查询数: {len(queries)}")
        self.search_start_time = time.time()
        
        url_goal_pairs = []
        seen_urls = set()
        
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
                    })
        
        self.search_end_time = time.time()
        search_elapsed = self.search_end_time - self.search_start_time
        logger.info(f"✅ 搜索完成 | 耗时: {search_elapsed:.2f}s | URL: {len(url_goal_pairs)}")
        
        return url_goal_pairs
    
    async def browse_single(
        self,
        session: aiohttp.ClientSession,
        url: str,
        goal: str,
        request_id: int,
        semaphore: asyncio.Semaphore,
    ) -> BrowseResult:
        """单个 Browse 请求"""
        async with semaphore:
            start_time = time.time()
            result = BrowseResult(url=url, goal=goal, request_id=request_id)
            
            try:
                async with session.post(
                    self.browse_url,
                    json={"url": url, "goal": goal},
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as resp:
                    elapsed = time.time() - start_time
                    result.total_time = elapsed
                    
                    if resp.status == 200:
                        data = await resp.json()
                        result.success = data.get("success", False)
                        result.from_cache = data.get("from_cache", False)
                        result.error = data.get("error", "")
                        
                        response_text = data.get("result", "")
                        result.response_length = len(response_text)
                        
                        # 解析响应内容获取 evidence 和 summary 长度
                        if "Evidence in page:" in response_text:
                            parts = response_text.split("Evidence in page:")
                            if len(parts) > 1:
                                evidence_part = parts[1].split("Summary:")[0] if "Summary:" in parts[1] else parts[1]
                                result.evidence_length = len(evidence_part.strip())
                        
                        if "Summary:" in response_text:
                            parts = response_text.split("Summary:")
                            if len(parts) > 1:
                                result.summary_length = len(parts[-1].strip())
                        
                        # 估算阶段时间
                        if result.from_cache:
                            # 缓存命中：主要是 SQL + Summary
                            result.estimated_sql_time = min(0.1, elapsed * 0.05)  # SQL 很快
                            result.estimated_crawler_time = 0
                            result.estimated_summary_time = elapsed - result.estimated_sql_time
                        else:
                            # 缓存未命中：SQL + 爬虫 + Summary
                            # 根据经验估算比例
                            result.estimated_sql_time = min(0.1, elapsed * 0.02)
                            # 假设 Summary 约占 40%（LLM 调用）
                            result.estimated_summary_time = elapsed * 0.4
                            result.estimated_crawler_time = elapsed - result.estimated_sql_time - result.estimated_summary_time
                        
                        status = "💾" if result.from_cache else "🌐"
                        logger.info(f"{status} #{request_id:03d} | {elapsed:.2f}s | {url[:50]}...")
                    else:
                        result.error = f"HTTP {resp.status}"
                        logger.error(f"❌ #{request_id:03d} | HTTP {resp.status} | {url[:50]}...")
                        
            except asyncio.TimeoutError:
                result.total_time = time.time() - start_time
                result.error = "Timeout"
                logger.error(f"❌ #{request_id:03d} | 超时 | {url[:50]}...")
            except Exception as e:
                result.total_time = time.time() - start_time
                result.error = str(e)
                logger.error(f"❌ #{request_id:03d} | {e} | {url[:50]}...")
            
            return result
    
    async def browse_batch(self, url_goal_pairs: List[Dict[str, str]]) -> List[BrowseResult]:
        """批量 Browse"""
        logger.info(f"📊 开始批量 Browse | URL: {len(url_goal_pairs)} | 并发: {self.concurrency}")
        self.browse_start_time = time.time()
        
        semaphore = asyncio.Semaphore(self.concurrency)
        
        # aiohttp 默认连接数上限是 100，这里显式放大到并发上限，避免被 100 的默认限制住
        connector = aiohttp.TCPConnector(limit=self.concurrency, limit_per_host=self.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self.browse_single(session, pair["url"], pair["goal"], i, semaphore)
                for i, pair in enumerate(url_goal_pairs)
            ]
            self.results = await asyncio.gather(*tasks)
        
        self.browse_end_time = time.time()
        elapsed = self.browse_end_time - self.browse_start_time
        logger.info(f"✅ Browse 完成 | 耗时: {elapsed:.2f}s")
        
        return self.results
    
    def calculate_stats(self) -> Dict[str, Any]:
        """计算统计结果"""
        if not self.results:
            return {}
        
        def calc_percentiles(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"count": 0, "avg": 0, "min": 0, "max": 0, "median": 0, "p95": 0, "p99": 0, "std": 0}
            sorted_v = sorted(values)
            n = len(sorted_v)
            return {
                "count": n,
                "avg": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
                "p95": sorted_v[int(n * 0.95)] if n > 1 else sorted_v[0],
                "p99": sorted_v[int(n * 0.99)] if n > 1 else sorted_v[0],
                "std": statistics.stdev(values) if n > 1 else 0,
            }
        
        # 分类结果
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        cache_hits = [r for r in successful if r.from_cache]
        cache_misses = [r for r in successful if not r.from_cache]
        
        # 时间统计
        all_times = [r.total_time for r in successful]
        cache_hit_times = [r.total_time for r in cache_hits]
        cache_miss_times = [r.total_time for r in cache_misses]
        
        # 阶段时间估算
        sql_times = [r.estimated_sql_time for r in successful]
        crawler_times = [r.estimated_crawler_time for r in cache_misses]  # 只有未命中才有爬虫
        summary_times = [r.estimated_summary_time for r in successful]
        
        stats = {
            "overview": {
                "total_requests": len(self.results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": f"{len(successful) / len(self.results) * 100:.1f}%",
                "cache_hits": len(cache_hits),
                "cache_misses": len(cache_misses),
                "cache_hit_rate": f"{len(cache_hits) / len(successful) * 100:.1f}%" if successful else "0%",
            },
            "timing": {
                "search_phase_seconds": self.search_end_time - self.search_start_time,
                "browse_phase_seconds": self.browse_end_time - self.browse_start_time,
                "total_seconds": (self.search_end_time - self.search_start_time) + (self.browse_end_time - self.browse_start_time),
                "qps": len(self.results) / (self.browse_end_time - self.browse_start_time) if self.browse_end_time > self.browse_start_time else 0,
            },
            "response_time": {
                "all_requests": calc_percentiles(all_times),
                "cache_hits": calc_percentiles(cache_hit_times),
                "cache_misses": calc_percentiles(cache_miss_times),
            },
            "phase_time_estimate": {
                "sql": calc_percentiles(sql_times),
                "crawler": calc_percentiles(crawler_times),
                "summary": calc_percentiles(summary_times),
            },
        }
        
        # 添加监控数据
        if self.monitor:
            stats["service_monitoring"] = self.monitor.get_peak_stats()
        
        # 错误统计
        errors = defaultdict(int)
        for r in failed:
            errors[r.error or "Unknown"] += 1
        stats["errors"] = dict(errors)
        
        return stats
    
    def print_report(self, stats: Dict[str, Any]):
        """打印报告"""
        print("\n" + "=" * 70)
        print("📊 Browse 压测报告")
        print("=" * 70)
        
        overview = stats["overview"]
        print(f"\n🔢 请求统计")
        print(f"   总请求数: {overview['total_requests']}")
        print(f"   成功: {overview['successful']} | 失败: {overview['failed']} | 成功率: {overview['success_rate']}")
        print(f"   缓存命中: {overview['cache_hits']} | 未命中: {overview['cache_misses']} | 命中率: {overview['cache_hit_rate']}")
        
        timing = stats["timing"]
        print(f"\n⏱️  耗时统计")
        print(f"   Search 阶段: {timing['search_phase_seconds']:.2f}s")
        print(f"   Browse 阶段: {timing['browse_phase_seconds']:.2f}s")
        print(f"   总耗时: {timing['total_seconds']:.2f}s")
        print(f"   QPS: {timing['qps']:.2f}")
        
        print(f"\n📈 响应时间（秒）")
        print("-" * 50)
        
        def print_stats(name: str, data: Dict[str, float]):
            if data["count"] == 0:
                print(f"   {name}: 无数据")
                return
            print(f"   {name}:")
            print(f"      数量: {data['count']}")
            print(f"      平均: {data['avg']:.2f}s | 中位: {data['median']:.2f}s")
            print(f"      最小: {data['min']:.2f}s | 最大: {data['max']:.2f}s")
            print(f"      P95: {data['p95']:.2f}s | P99: {data['p99']:.2f}s")
        
        rt = stats["response_time"]
        print_stats("全部请求", rt["all_requests"])
        print_stats("缓存命中 (SQL + Summary)", rt["cache_hits"])
        print_stats("缓存未命中 (SQL + 爬虫 + Summary)", rt["cache_misses"])
        
        print(f"\n🔧 各阶段时间估算（秒）")
        print("-" * 50)
        phase = stats["phase_time_estimate"]
        print_stats("SQL 查询", phase["sql"])
        print_stats("爬虫抓取", phase["crawler"])
        print_stats("Summary LLM", phase["summary"])
        
        if "service_monitoring" in stats and stats["service_monitoring"]:
            print(f"\n📡 服务监控峰值")
            print("-" * 50)
            mon = stats["service_monitoring"]
            print(f"   总并发峰值: {mon.get('peak_total_running', 0)}")
            print(f"   SQL 并发峰值: {mon.get('peak_sql_running', 0)}")
            print(f"   爬虫并发峰值: {mon.get('peak_crawler_running', 0)}")
            print(f"   Summary 并发峰值: {mon.get('peak_summary_running', 0)}")
        
        if stats.get("errors"):
            print(f"\n❌ 错误统计")
            print("-" * 50)
            for error, count in stats["errors"].items():
                print(f"   {error}: {count}")
        
        print("\n" + "=" * 70)
        print(f"📁 日志文件: {log_file}")
        print(f"📁 结果文件: {result_file}")
        print(f"📁 CSV 文件: {csv_file}")
        print("=" * 70)
    
    def save_results(self, stats: Dict[str, Any]):
        """保存结果"""
        # JSON 格式
        output = {
            "config": {
                "search_url": self.search_url,
                "browse_url": self.browse_url,
                "concurrency": self.concurrency,
                "search_limit": self.search_limit,
                "timestamp": timestamp_str,
            },
            "stats": stats,
            "results": [asdict(r) for r in self.results],
        }
        
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        # CSV 格式
        with open(csv_file, "w", encoding="utf-8") as f:
            headers = ["request_id", "url", "goal", "success", "from_cache", "total_time", 
                      "response_length", "estimated_sql_time", "estimated_crawler_time", 
                      "estimated_summary_time", "error"]
            f.write(",".join(headers) + "\n")
            
            for r in self.results:
                row = [
                    str(r.request_id),
                    f'"{r.url}"',
                    f'"{r.goal[:50]}"',
                    str(r.success),
                    str(r.from_cache),
                    f"{r.total_time:.3f}",
                    str(r.response_length),
                    f"{r.estimated_sql_time:.3f}",
                    f"{r.estimated_crawler_time:.3f}",
                    f"{r.estimated_summary_time:.3f}",
                    f'"{r.error}"',
                ]
                f.write(",".join(row) + "\n")
        
        logger.info(f"💾 结果已保存")
    
    def run(self, queries: List[str] = None, url_goal_pairs: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """运行压测"""
        logger.info("=" * 60)
        logger.info("🚀 开始 Browse 详细压测")
        logger.info(f"   Search URL: {self.search_url}")
        logger.info(f"   Browse URL: {self.browse_url}")
        logger.info(f"   并发数: {self.concurrency}")
        logger.info("=" * 60)
        
        # 获取 URL 列表
        if url_goal_pairs is None:
            if queries is None:
                queries = [
                    "What is artificial intelligence?",
                    "Python programming best practices",
                    "Machine learning algorithms explained",
                ]
                logger.info(f"⚠️  使用默认测试查询 ({len(queries)} 个)")
            
            url_goal_pairs = self.search_batch(queries)
        else:
            self.search_start_time = time.time()
            self.search_end_time = time.time()
        
        # 确保 URL 数量足够
        initial_url_count = len(url_goal_pairs)
        if initial_url_count == 0:
            logger.error("❌ 没有 URL，压测终止")
            return {"error": "No URLs"}
        
        if len(url_goal_pairs) < self.num_requests:
            logger.info(f"🔄 URL 数量不足 ({initial_url_count} < {self.num_requests})，将重复使用 URL。")
            # 重复 URL 直到达到 num_requests
            repeated_url_goal_pairs = []
            for i in range(self.num_requests):
                repeated_url_goal_pairs.append(url_goal_pairs[i % initial_url_count])
            url_goal_pairs = repeated_url_goal_pairs
        elif len(url_goal_pairs) > self.num_requests:
            logger.info(f"✂️ URL 数量过多 ({initial_url_count} > {self.num_requests})，将截断 URL。")
            url_goal_pairs = url_goal_pairs[:self.num_requests]
        
        logger.info(f"最终压测 URL 数量: {len(url_goal_pairs)}")
        
        # 启动监控
        if self.monitor:
            self.monitor.start()
        
        try:
            # 运行压测
            asyncio.run(self.browse_batch(url_goal_pairs))
        finally:
            # 停止监控
            if self.monitor:
                self.monitor.stop()
        
        # 计算统计
        stats = self.calculate_stats()
        
        # 打印报告
        self.print_report(stats)
        
        # 保存结果
        self.save_results(stats)
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Browse 服务详细压测")
    
    parser.add_argument("--queries", nargs="+", help="搜索查询列表")
    parser.add_argument("--query-file", help="查询文件（每行一个）")
    parser.add_argument("--urls-file", help="URL 文件（每行一个）")
    parser.add_argument("--goal", default="Find relevant information", help="URL 模式的查询目标")
    
    parser.add_argument("--search-url", default=DEFAULT_SEARCH_URL)
    parser.add_argument("--browse-url", default=DEFAULT_BROWSE_URL)
    parser.add_argument("--stats-url", default=DEFAULT_BROWSE_STATS_URL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--search-limit", type=int, default=DEFAULT_SEARCH_LIMIT)
    parser.add_argument("--no-monitor", action="store_true", help="禁用服务监控")
    parser.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS, help="总请求数，不足时会重复使用 URL")
    
    args = parser.parse_args()
    
    # 准备输入
    queries = None
    url_goal_pairs = None
    
    if args.urls_file:
        # 从文件读取 URL
        if os.path.exists(args.urls_file):
            with open(args.urls_file, "r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
            url_goal_pairs = [{"url": url, "goal": args.goal} for url in urls]
            logger.info(f"📁 从文件加载 {len(url_goal_pairs)} 个 URL")
        else:
            logger.error(f"❌ 文件不存在: {args.urls_file}")
            sys.exit(1)
    elif args.query_file:
        # 从文件读取查询
        if os.path.exists(args.query_file):
            with open(args.query_file, "r", encoding="utf-8") as f:
                queries = [line.strip() for line in f if line.strip()]
        else:
            logger.error(f"❌ 文件不存在: {args.query_file}")
            sys.exit(1)
    elif args.queries:
        queries = args.queries
    
    # 运行压测
    benchmark = DetailedBenchmark(
        search_url=args.search_url,
        browse_url=args.browse_url,
        stats_url=args.stats_url,
        concurrency=args.concurrency,
        search_limit=args.search_limit,
        enable_monitoring=not args.no_monitor,
    )
    
    benchmark.run(queries=queries, url_goal_pairs=url_goal_pairs)


if __name__ == "__main__":
    main()

