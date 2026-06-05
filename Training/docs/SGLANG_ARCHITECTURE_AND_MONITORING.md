# SGLang 在 verl 框架中的架构与监控指南

本文档详细解释 verl 框架中 SGLang 的创建、调用流程以及如何监控 KV Cache 使用和请求队列状态。

## 目录

1. [架构概览](#架构概览)
2. [SGLang 进程创建流程](#sglang-进程创建流程)
3. [AgentLoop 调用 SGLang 流程](#agentloop-调用-sglang-流程)
4. [监控 KV Cache 和请求队列](#监控-kv-cache-和请求队列)
5. [为什么 verl 中的 SGLang 也可以监控](#为什么-verl-中的-sglang-也可以监控)
6. [实用命令和脚本](#实用命令和脚本)

---

## 架构概览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              verl 训练框架                                     │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         RayPPOTrainer (ray_trainer.py)                   │ │
│  │                                                                          │ │
│  │   fit() → generate_sequences() → actor_rollout_wg.generate_sequences() │ │
│  └────────────────────────────────────┬────────────────────────────────────┘ │
│                                        │                                      │
│                                        ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      AgentLoopManager (agent_loop.py)                    │ │
│  │                                                                          │ │
│  │  ┌────────────────────┐    ┌────────────────────────────────────────┐   │ │
│  │  │ _initialize_llm_   │    │     _init_agent_loop_workers()        │   │ │
│  │  │    servers()       │    │                                        │   │ │
│  │  │                    │    │   创建 N 个 AgentLoopWorker (Ray Actor)│   │ │
│  │  │  创建 SGLangReplica │    │   每个 Worker 负责处理一部分样本        │   │ │
│  │  │  启动 HTTP Server   │    │                                        │   │ │
│  │  └────────────────────┘    └────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      SGLang HTTP Server Layer                            │ │
│  │                                                                          │ │
│  │   ┌──────────────┐  ┌──────────────┐       ┌──────────────┐            │ │
│  │   │ SGLangHttp   │  │ SGLangHttp   │  ...  │ SGLangHttp   │            │ │
│  │   │ Server #0    │  │ Server #1    │       │ Server #N    │            │ │
│  │   │              │  │              │       │              │            │ │
│  │   │ GPU 0-3      │  │ GPU 4-7      │       │ GPU ...      │            │ │
│  │   │              │  │              │       │              │            │ │
│  │   │ Port: 30000  │  │ Port: 30001  │       │ Port: 3000N  │            │ │
│  │   └──────────────┘  └──────────────┘       └──────────────┘            │ │
│  │                                                                          │ │
│  │   每个 Server 提供:                                                       │ │
│  │   - /generate          生成接口                                          │ │
│  │   - /health            健康检查                                          │ │
│  │   - /get_server_info   服务器信息（包含 KV Cache 配置）                   │ │
│  │   - /metrics           Prometheus 指标（需 --enable-metrics）            │ │
│  │   - /flush_cache       清空缓存                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## SGLang 进程创建流程

### 1. 入口点：AgentLoopManager 初始化

```python
# verl/experimental/agent_loop/agent_loop.py

class AgentLoopManager:
    def __init__(self, config, worker_group, rm_wg):
        # ...
        self._initialize_llm_servers()    # 步骤1: 创建SGLang服务器
        self._init_agent_loop_workers()   # 步骤2: 创建Agent Loop Workers
```

### 2. SGLang 服务器初始化

```python
def _initialize_llm_servers(self):
    # 计算副本数量
    rollout_world_size = tp_size * dp_size * pp_size
    num_replicas = total_gpus // rollout_world_size
    
    # 为每个副本创建 SGLangReplica
    self.rollout_replicas = [
        SGLangReplica(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]
    
    # 启动服务器（Hybrid 或 Standalone 模式）
    if self.worker_group:
        [server.init_hybrid(self.worker_group) for server in self.rollout_replicas]
    else:
        [server.init_standalone() for server in self.rollout_replicas]
```

### 3. SGLangReplica.launch_servers() 详细流程

```python
# verl/workers/rollout/sglang_rollout/async_sglang_server.py

class SGLangReplica(RolloutReplica):
    async def launch_servers(self):
        # 1. 获取 worker 的节点信息
        worker_infos = await asyncio.gather(*[
            worker.__ray_call__.remote(
                lambda self: (ray.get_runtime_context().get_node_id(), 
                             os.environ["CUDA_VISIBLE_DEVICES"])
            )
            for worker in self.workers
        ])
        
        # 2. 在每个节点创建 SGLangHttpServer Ray Actor
        for node_rank in range(self.nnodes):
            server = SGLangHttpServer.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id),
                name=f"sglang_server_{self.replica_rank}_{node_rank}",
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)
        
        # 3. 启动 HTTP 服务器
        await asyncio.gather(*[
            server.launch_server.remote(master_address, master_port)
            for server in self.servers
        ])
```

### 4. SGLangHttpServer.launch_server() 核心代码

```python
# verl/workers/rollout/sglang_rollout/async_sglang_server.py

@ray.remote(num_cpus=1)
class SGLangHttpServer:
    async def launch_server(self, master_address=None, master_port=None):
        # 配置 SGLang 服务器参数
        args = {
            "model_path": self.model_config.local_path,
            "dtype": self.config.dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "tp_size": self.config.tensor_model_parallel_size,
            "dp_size": self.config.data_parallel_size,
            "max_running_requests": self.config.get("max_num_seqs", None),
            # ... 更多参数
        }
        
        server_args = ServerArgs(**args)
        
        # 启动 SGLang 子进程
        self.tokenizer_manager, self.template_manager, self.scheduler_info = \
            _launch_subprocesses(server_args=server_args)
        
        # 启动 HTTP 服务器 (uvicorn)
        self._server_port, self._server_task = await run_unvicorn(
            app, server_args, self._server_address
        )
```

---

## AgentLoop 调用 SGLang 流程

### 1. 请求分发

```python
# AgentLoopManager.generate_sequences()
def generate_sequences(self, prompts: DataProto) -> DataProto:
    # 将输入批次分割到多个 Worker
    chunks = prompts.chunk(len(self.agent_loop_workers))
    
    # 并行执行
    outputs = ray.get([
        worker.generate_sequences.remote(chunk)
        for worker, chunk in zip(self.agent_loop_workers, chunks)
    ])
```

### 2. Worker 执行 Agent Loop

```python
# AgentLoopWorker.generate_sequences()
async def generate_sequences(self, prompts):
    tasks = [
        self._run_agent_loop(...)  # 每个样本一个协程
        for i, sample in enumerate(prompts)
    ]
    results = await asyncio.gather(*tasks)
```

### 3. LLM 调用

```python
# AsyncLLMServerManager.generate()
async def generate(self, request_id, prompt_ids, sampling_params, image_data=None):
    # 选择服务器（负载均衡 + 粘性会话）
    server = self._choose_server(request_id)
    
    # 调用 SGLang HTTP 服务器
    output = await server.generate.remote(
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        request_id=request_id,
        image_data=image_data,
    )
    return output
```

---

## 监控 KV Cache 和请求队列

### 方法一：使用 /get_server_info 端点

```bash
# 获取服务器基本信息（包括 KV Cache 配置）
curl http://<server_ip>:30000/get_server_info | python -m json.tool
```

返回示例：
```json
{
    "model_path": "/path/to/model",
    "max_total_num_tokens": 131072,
    "max_prefill_tokens": 16384,
    "mem_fraction_static": 0.85,
    "max_running_requests": 256,
    "context_length": 32768,
    "kv_cache_dtype": "fp16",
    ...
}
```

### 方法二：使用 /metrics 端点（Prometheus 格式）

**注意**：需要在启动时添加 `--enable-metrics` 参数。

在 verl 框架中，可以通过修改配置来启用：

```yaml
# config 中添加
actor_rollout_ref:
  rollout:
    engine_kwargs:
      sglang:
        enable_metrics: true
```

或者直接修改 `async_sglang_server.py`。

启用后：
```bash
curl http://<server_ip>:30000/metrics
```

返回示例：
```prometheus
# HELP sglang_num_queue_reqs Number of requests in queue
# TYPE sglang_num_queue_reqs gauge
sglang_num_queue_reqs 12

# HELP sglang_token_usage KV cache token usage ratio
# TYPE sglang_token_usage gauge
sglang_token_usage 0.75

# HELP sglang_num_running_reqs Number of currently running requests
# TYPE sglang_num_running_reqs gauge
sglang_num_running_reqs 8

# HELP sglang_gen_throughput Generation throughput (tokens/sec)
# TYPE sglang_gen_throughput gauge
sglang_gen_throughput 1234.5
```

### 方法三：在 verl 框架中添加监控代码

我们可以在 `AgentLoopManager` 中添加一个监控方法：

```python
# 在 agent_loop.py 中添加
class AgentLoopManager:
    async def get_server_stats(self) -> list[dict]:
        """获取所有 SGLang 服务器的状态信息"""
        import aiohttp
        
        stats = []
        for i, addr in enumerate(self.server_addresses):
            try:
                async with aiohttp.ClientSession() as session:
                    # 获取服务器信息
                    async with session.get(f"http://{addr}/get_server_info") as resp:
                        info = await resp.json()
                    
                    # 尝试获取 metrics（如果启用）
                    try:
                        async with session.get(f"http://{addr}/metrics") as resp:
                            metrics_text = await resp.text()
                            # 解析 Prometheus 格式的 metrics
                            metrics = self._parse_prometheus_metrics(metrics_text)
                    except:
                        metrics = {}
                    
                    stats.append({
                        "server_id": i,
                        "address": addr,
                        "info": info,
                        "metrics": metrics,
                    })
            except Exception as e:
                stats.append({
                    "server_id": i,
                    "address": addr,
                    "error": str(e),
                })
        
        return stats
    
    def _parse_prometheus_metrics(self, text: str) -> dict:
        """解析 Prometheus 格式的 metrics"""
        metrics = {}
        for line in text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                metrics[parts[0]] = float(parts[1])
        return metrics
```

---

## 为什么 verl 中的 SGLang 也可以监控

### 原因分析

1. **verl 使用标准的 SGLang HTTP Server**：
   - verl 框架并没有重新实现 SGLang，而是直接调用 SGLang 的标准 HTTP 服务器
   - 使用 `sglang.srt.entrypoints.http_server` 中的 `launch_server` 和相关组件

2. **HTTP 端点完全保留**：
   ```python
   # async_sglang_server.py 中
   from sglang.srt.entrypoints.http_server import (
       ServerArgs,
       _GlobalState,
       _launch_subprocesses,
       app,  # ← 这是 FastAPI app，包含所有标准端点
       set_global_state,
   )
   ```

3. **服务器地址可访问**：
   - 每个 SGLang 服务器都绑定到一个端口（如 30000, 30001...）
   - 可以通过 `self.server_addresses` 获取所有服务器地址

### 架构对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                        原始 SGLang 部署                              │
│                                                                      │
│  python -m sglang.launch_server --model-path xxx --port 30000       │
│                            ↓                                         │
│  ┌────────────────────────────────────────┐                         │
│  │  SGLang HTTP Server (FastAPI)          │                         │
│  │                                         │                         │
│  │  /generate      ✓                       │                         │
│  │  /get_server_info  ✓                    │                         │
│  │  /metrics       ✓ (需 --enable-metrics) │                         │
│  │  /health        ✓                       │                         │
│  └────────────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        verl 框架中的 SGLang                          │
│                                                                      │
│  AgentLoopManager._initialize_llm_servers()                         │
│                            ↓                                         │
│  SGLangReplica.launch_servers()                                     │
│                            ↓                                         │
│  SGLangHttpServer.launch_server()                                   │
│                            ↓                                         │
│  _launch_subprocesses(server_args)  ← 调用标准 SGLang 启动函数       │
│  run_unvicorn(app, ...)             ← 使用标准 FastAPI app           │
│                            ↓                                         │
│  ┌────────────────────────────────────────┐                         │
│  │  SGLang HTTP Server (FastAPI)          │  ← 完全相同！            │
│  │                                         │                         │
│  │  /generate      ✓                       │                         │
│  │  /get_server_info  ✓                    │                         │
│  │  /metrics       ✓ (需配置启用)          │                         │
│  │  /health        ✓                       │                         │
│  └────────────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 实用命令和脚本

### 1. 查看服务器地址

```python
# 在训练脚本中添加
from verl.experimental.agent_loop.agent_loop import AgentLoopManager

# 假设已有 manager 实例
print("SGLang Server Addresses:")
for i, addr in enumerate(manager.server_addresses):
    print(f"  Server {i}: http://{addr}")
```

### 2. 监控脚本

创建文件 `scripts/monitor_sglang.py`：

```python
#!/usr/bin/env python3
"""SGLang 服务器监控脚本"""

import argparse
import asyncio
import json
import aiohttp
from typing import Optional


async def get_server_info(host: str, port: int) -> dict:
    """获取服务器信息"""
    url = f"http://{host}:{port}/get_server_info"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            return await resp.json()


async def get_metrics(host: str, port: int) -> Optional[str]:
    """获取 Prometheus metrics"""
    url = f"http://{host}:{port}/metrics"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.text()
    except:
        return None


def parse_metrics(text: str) -> dict:
    """解析 Prometheus metrics"""
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


async def monitor(host: str, port: int, interval: float = 2.0):
    """持续监控服务器状态"""
    print(f"Monitoring SGLang server at {host}:{port}")
    print("=" * 60)
    
    while True:
        try:
            # 获取服务器信息
            info = await get_server_info(host, port)
            
            print(f"\n📊 Server Info:")
            print(f"  Model: {info.get('model_path', 'N/A')}")
            print(f"  Max Tokens: {info.get('max_total_num_tokens', 'N/A')}")
            print(f"  Max Running Requests: {info.get('max_running_requests', 'N/A')}")
            print(f"  Context Length: {info.get('context_length', 'N/A')}")
            
            # 尝试获取 metrics
            metrics_text = await get_metrics(host, port)
            if metrics_text:
                metrics = parse_metrics(metrics_text)
                print(f"\n📈 Metrics:")
                print(f"  Queue Requests: {metrics.get('sglang_num_queue_reqs', 'N/A')}")
                print(f"  Running Requests: {metrics.get('sglang_num_running_reqs', 'N/A')}")
                print(f"  KV Cache Usage: {metrics.get('sglang_token_usage', 'N/A'):.2%}" 
                      if 'sglang_token_usage' in metrics else "  KV Cache Usage: N/A")
                print(f"  Throughput: {metrics.get('sglang_gen_throughput', 'N/A'):.1f} tokens/s"
                      if 'sglang_gen_throughput' in metrics else "  Throughput: N/A")
            else:
                print("\n⚠️  Metrics endpoint not available (enable with --enable-metrics)")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor SGLang server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    
    args = parser.parse_args()
    
    if args.once:
        info = asyncio.run(get_server_info(args.host, args.port))
        print(json.dumps(info, indent=2))
    else:
        asyncio.run(monitor(args.host, args.port, args.interval))


if __name__ == "__main__":
    main()
```

### 3. 快速检查脚本

```bash
#!/bin/bash
# scripts/check_sglang_servers.sh

# 默认端口范围
START_PORT=${1:-30000}
NUM_SERVERS=${2:-8}
HOST=${3:-localhost}

echo "Checking SGLang servers..."
echo "=========================="

for i in $(seq 0 $((NUM_SERVERS - 1))); do
    PORT=$((START_PORT + i))
    echo -n "Server $i (port $PORT): "
    
    # 检查健康状态
    if curl -s --connect-timeout 2 "http://${HOST}:${PORT}/health" > /dev/null 2>&1; then
        echo "✅ OK"
        
        # 获取服务器信息
        INFO=$(curl -s "http://${HOST}:${PORT}/get_server_info" 2>/dev/null)
        if [ -n "$INFO" ]; then
            MAX_REQS=$(echo "$INFO" | python3 -c "import sys,json; print(json.load(sys.stdin).get('max_running_requests', 'N/A'))" 2>/dev/null)
            echo "   Max Running Requests: $MAX_REQS"
        fi
    else
        echo "❌ Not responding"
    fi
done
```

### 4. 启用 Metrics 的配置方法

修改 `async_sglang_server.py` 中的 `launch_server` 方法：

```python
# 在 args 字典中添加
args = {
    # ... 现有参数 ...
    "enable_metrics": True,  # 启用 Prometheus metrics
}
```

或者通过配置文件：

```yaml
# 在 config.yaml 中
actor_rollout_ref:
  rollout:
    engine_kwargs:
      sglang:
        enable_metrics: true
```

---

## 关键指标说明

| 指标名 | 说明 | 建议值 |
|-------|------|-------|
| `sglang_num_queue_reqs` | 等待队列中的请求数 | < max_running_requests * 2 |
| `sglang_num_running_reqs` | 正在执行的请求数 | ≈ max_running_requests |
| `sglang_token_usage` | KV Cache 使用率 | 0.7 - 0.9 最佳 |
| `sglang_gen_throughput` | 生成吞吐量 (tokens/s) | 越高越好 |
| `sglang_cache_hit_rate` | 前缀缓存命中率 | 越高越好（multi-turn场景） |

---

## 总结

1. **verl 框架完全复用 SGLang 的标准 HTTP 服务器**，所有监控端点都可用
2. **服务器地址**可通过 `AgentLoopManager.server_addresses` 获取
3. **基础监控**使用 `/get_server_info` 端点（始终可用）
4. **详细监控**使用 `/metrics` 端点（需启用 `enable_metrics`）
5. 推荐使用提供的监控脚本进行实时观察

---

*文档更新日期: 2026-01-13*
