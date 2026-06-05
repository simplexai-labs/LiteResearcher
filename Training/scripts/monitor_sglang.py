#!/usr/bin/env python3
"""
SGLang 服务器监控脚本

用于监控 verl 框架中运行的 SGLang 服务器的状态，包括：
- KV Cache 使用情况
- 请求队列长度
- 服务器吞吐量

使用方法:
    # 持续监控
    python scripts/monitor_sglang.py --host localhost --port 30000
    
    # 单次查询
    python scripts/monitor_sglang.py --host localhost --port 30000 --once
    
    # 监控多个服务器
    python scripts/monitor_sglang.py --hosts localhost:30000,localhost:30001
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("请安装 aiohttp: pip install aiohttp")
    sys.exit(1)


@dataclass
class ServerStatus:
    """服务器状态"""
    server_id: int
    address: str
    is_healthy: bool
    info: Optional[dict] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None


async def get_server_info(session: aiohttp.ClientSession, host: str, port: int) -> dict:
    """获取服务器信息"""
    url = f"http://{host}:{port}/get_server_info"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        return await resp.json()


async def get_metrics(session: aiohttp.ClientSession, host: str, port: int) -> Optional[str]:
    """获取 Prometheus metrics"""
    url = f"http://{host}:{port}/metrics"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                return await resp.text()
    except Exception:
        return None
    return None


async def check_health(session: aiohttp.ClientSession, host: str, port: int) -> bool:
    """检查服务器健康状态"""
    url = f"http://{host}:{port}/health_generate"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            return resp.status == 200
    except Exception:
        return False


def parse_metrics(text: str) -> dict:
    """解析 Prometheus metrics 格式"""
    metrics = {}
    for line in text.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                metrics[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return metrics


async def get_server_status(server_id: int, host: str, port: int) -> ServerStatus:
    """获取单个服务器的完整状态"""
    address = f"{host}:{port}"
    
    try:
        async with aiohttp.ClientSession() as session:
            # 检查健康状态
            is_healthy = await check_health(session, host, port)
            
            if not is_healthy:
                return ServerStatus(
                    server_id=server_id,
                    address=address,
                    is_healthy=False,
                    error="Server not responding"
                )
            
            # 获取服务器信息
            info = await get_server_info(session, host, port)
            
            # 尝试获取 metrics
            metrics_text = await get_metrics(session, host, port)
            metrics = parse_metrics(metrics_text) if metrics_text else None
            
            return ServerStatus(
                server_id=server_id,
                address=address,
                is_healthy=True,
                info=info,
                metrics=metrics
            )
            
    except Exception as e:
        return ServerStatus(
            server_id=server_id,
            address=address,
            is_healthy=False,
            error=str(e)
        )


def print_status(status: ServerStatus, verbose: bool = False):
    """打印服务器状态"""
    header = f"Server {status.server_id} ({status.address})"
    
    if not status.is_healthy:
        print(f"❌ {header}: {status.error}")
        return
    
    print(f"✅ {header}")
    
    if status.info:
        info = status.info
        print(f"   📦 Model: {info.get('model_path', 'N/A').split('/')[-1]}")
        print(f"   🎯 Max Running Requests: {info.get('max_running_requests', 'N/A')}")
        print(f"   📊 Max Total Tokens: {info.get('max_total_num_tokens', 'N/A')}")
        print(f"   📏 Context Length: {info.get('context_length', 'N/A')}")
        
        if verbose:
            print(f"   🔧 TP Size: {info.get('tp_size', 'N/A')}")
            print(f"   🔧 DP Size: {info.get('dp_size', 'N/A')}")
            print(f"   💾 Mem Fraction: {info.get('mem_fraction_static', 'N/A')}")
    
    if status.metrics:
        m = status.metrics
        print(f"   📈 Metrics:")
        
        queue_reqs = m.get('sglang_num_queue_reqs')
        if queue_reqs is not None:
            print(f"      Queue Requests: {int(queue_reqs)}")
        
        running_reqs = m.get('sglang_num_running_reqs')
        if running_reqs is not None:
            print(f"      Running Requests: {int(running_reqs)}")
        
        token_usage = m.get('sglang_token_usage')
        if token_usage is not None:
            # 显示颜色提示
            if token_usage > 0.9:
                indicator = "🔴"  # 高负载
            elif token_usage > 0.7:
                indicator = "🟡"  # 正常
            else:
                indicator = "🟢"  # 低负载
            print(f"      KV Cache Usage: {indicator} {token_usage:.1%}")
        
        throughput = m.get('sglang_gen_throughput')
        if throughput is not None:
            print(f"      Throughput: {throughput:.1f} tokens/s")
        
        cache_hit = m.get('sglang_cache_hit_rate')
        if cache_hit is not None:
            print(f"      Cache Hit Rate: {cache_hit:.1%}")
    else:
        print(f"   ⚠️  Metrics not available (enable with --enable-metrics)")


async def monitor_servers(servers: list[tuple[str, int]], interval: float, verbose: bool):
    """持续监控多个服务器"""
    print("🔍 SGLang Server Monitor")
    print("=" * 60)
    print(f"Monitoring {len(servers)} server(s), refresh every {interval}s")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    while True:
        print(f"\n⏰ {asyncio.get_event_loop().time():.0f}")
        print("-" * 60)
        
        tasks = [
            get_server_status(i, host, port)
            for i, (host, port) in enumerate(servers)
        ]
        statuses = await asyncio.gather(*tasks)
        
        for status in statuses:
            print_status(status, verbose)
            print()
        
        # 汇总统计
        healthy_count = sum(1 for s in statuses if s.is_healthy)
        total_queue = sum(
            s.metrics.get('sglang_num_queue_reqs', 0)
            for s in statuses
            if s.metrics
        )
        total_running = sum(
            s.metrics.get('sglang_num_running_reqs', 0)
            for s in statuses
            if s.metrics
        )
        
        print("-" * 60)
        print(f"📊 Summary: {healthy_count}/{len(servers)} servers healthy")
        if total_queue or total_running:
            print(f"   Total Queue: {int(total_queue)}, Total Running: {int(total_running)}")
        
        await asyncio.sleep(interval)


async def query_once(servers: list[tuple[str, int]], output_json: bool, verbose: bool):
    """单次查询所有服务器"""
    tasks = [
        get_server_status(i, host, port)
        for i, (host, port) in enumerate(servers)
    ]
    statuses = await asyncio.gather(*tasks)
    
    if output_json:
        result = []
        for status in statuses:
            item = {
                "server_id": status.server_id,
                "address": status.address,
                "is_healthy": status.is_healthy,
            }
            if status.info:
                item["info"] = status.info
            if status.metrics:
                item["metrics"] = status.metrics
            if status.error:
                item["error"] = status.error
            result.append(item)
        print(json.dumps(result, indent=2))
    else:
        for status in statuses:
            print_status(status, verbose)
            print()


def parse_hosts(hosts_str: str) -> list[tuple[str, int]]:
    """解析主机列表字符串"""
    servers = []
    for item in hosts_str.split(','):
        item = item.strip()
        if ':' in item:
            host, port = item.rsplit(':', 1)
            servers.append((host, int(port)))
        else:
            servers.append((item, 30000))  # 默认端口
    return servers


def main():
    parser = argparse.ArgumentParser(
        description="Monitor SGLang servers in verl framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor single server
  python monitor_sglang.py --host localhost --port 30000

  # Monitor multiple servers
  python monitor_sglang.py --hosts localhost:30000,localhost:30001

  # Query once and output JSON
  python monitor_sglang.py --hosts localhost:30000 --once --json

  # Auto-discover servers (scan ports 30000-30007)
  python monitor_sglang.py --host localhost --scan 8
        """
    )
    
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=30000, help="Server port (default: 30000)")
    parser.add_argument("--hosts", help="Comma-separated list of host:port pairs")
    parser.add_argument("--scan", type=int, help="Scan N consecutive ports starting from --port")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds (default: 2.0)")
    parser.add_argument("--once", action="store_true", help="Query once and exit")
    parser.add_argument("--json", action="store_true", help="Output as JSON (with --once)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    
    args = parser.parse_args()
    
    # 确定服务器列表
    if args.hosts:
        servers = parse_hosts(args.hosts)
    elif args.scan:
        servers = [(args.host, args.port + i) for i in range(args.scan)]
    else:
        servers = [(args.host, args.port)]
    
    try:
        if args.once:
            asyncio.run(query_once(servers, args.json, args.verbose))
        else:
            asyncio.run(monitor_servers(servers, args.interval, args.verbose))
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")


if __name__ == "__main__":
    main()
