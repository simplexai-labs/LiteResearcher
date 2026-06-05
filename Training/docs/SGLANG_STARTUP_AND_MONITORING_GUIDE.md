# VERL AgentLoop 中 SGLang 启动机制与监控深度指南

> **作者**: Claude Code
> **日期**: 2025-01-14
> **SGLang 版本**: 0.5.2 (verl-v060 环境)
> **适用场景**: VERL AgentLoop 多轮对话训练

---

## 目录

1. [SGLang 启动机制详解](#1-sglang-启动机制详解)
2. [与命令行启动的区别](#2-与命令行启动的区别)
3. [监控端点深度剖析](#3-监控端点深度剖析)
4. [为何现有监控可能失效](#4-为何现有监控可能失效)
5. [正确的监控方案](#5-正确的监控方案)
6. [底层源码分析](#6-底层源码分析)
7. [常见问题排查](#7-常见问题排查)

---

## 1. SGLang 启动机制详解

### 1.1 VERL 中的启动流程

VERL 的 AgentLoop 系统使用 **Ray 分布式框架** + **SGLang HTTP Server** 的混合架构：

```
AgentLoopManager 初始化
    ↓
计算 replica 数量 (总GPU数 / TP/DP/PP并行度)
    ↓
创建 SGLangReplica 实例
    ↓
调用 init_hybrid() 或 init_standalone()
    ↓
在每个节点创建 SGLangHttpServer Ray Actor
    ↓
启动 SGLang HTTP 服务器 (端口: 30000 + replica_rank)
    ↓
创建 AgentLoopWorker 连接到 SGLang 服务器
    ↓
开始多轮对话推理
```

**关键源码位置**：
- `verl/experimental/agent_loop/agent_loop.py:854-885` (`_initialize_llm_servers`)
- `verl/workers/rollout/sglang_rollout/async_sglang_server.py:251-308` (`launch_servers`)

### 1.2 两种启动模式

#### HYBRID 模式 (推荐)

**特点**: 重用 Actor 训练的 GPU 进程，在同一 GPU 上共享显存

```python
# verl/workers/rollout/sglang_rollout/async_sglang_server.py:251-308
async def launch_servers(self):
    # 1. 获取所有 worker 的 node_id 和 CUDA_VISIBLE_DEVICES
    worker_infos = await asyncio.gather(
        *[
            worker.__ray_call__.remote(
                lambda self: (ray.get_runtime_context().get_node_id(),
                             os.environ["CUDA_VISIBLE_DEVICES"])
            )
            for worker in self.workers
        ]
    )

    # 2. 在每个节点创建 SGLangHttpServer Ray Actor
    for node_rank in range(self.nnodes):
        node_id = worker_node_ids[node_rank * self.gpus_per_node]
        server = SGLangHttpServer.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
            name=f"sglang_server_{self.replica_rank}_{node_rank}",
        ).remote(...)

    # 3. 启动 HTTP 服务器
    await asyncio.gather(
        *[server.launch_server.remote(master_address=master_address,
                                     master_port=master_port)
          for server in self.servers]
    )
```

**显存共享机制**：
- Actor 进程和 SGLang 进程共享同一 GPU
- 通过 `wake_up()` / `sleep()` 方法控制显存占用
- 训练时：SGLang 释放 KV Cache 和权重显存 (`sleep`)
- 推理时：SGLang 重新占用显存，Actor 权重同步 (`wake_up`)

#### STANDALONE 模式

**特点**: 独立的 SGLang 服务器进程，不与训练共享 GPU

```python
# verl/workers/rollout/sglang_rollout/async_sglang_server.py
async def launch_servers(self):
    # 创建独立的 Ray Actor，不与 worker_group 绑定
    server = SGLangHttpServer.remote(...)
```

### 1.3 SGLang HTTP 服务器启动细节

**源码位置**: `verl/workers/rollout/sglang_rollout/async_sglang_server.py:119-181`

```python
async def launch_server(self, master_address: str = None, master_port: int = None):
    # 关键启动参数
    args = {
        "model_path": self.model_config.local_path,
        "dtype": self.config.dtype,
        "mem_fraction_static": self.config.gpu_memory_utilization,  # GPU 显存占用比例
        "disable_cuda_graph": self.config.enforce_eager,
        "enable_memory_saver": True,  # 启用内存节省模式
        "tp_size": self.config.tensor_model_parallel_size,
        "dp_size": self.config.data_parallel_size,
        "max_running_requests": self.config.get("max_num_seqs", None),
        "enable_metrics": engine_kwargs.pop("enable_metrics", True),  # ⚠️ 默认启用 Prometheus
        "dist_init_addr": dist_init_addr,  # NCCL 初始化地址
        "nnodes": self.nnodes,  # 多节点支持
    }

    # 调用 SGLang 内部启动方法
    sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config
    os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
    server_args = ServerArgs(**args)

    # 启动子进程 (tokenizer processor + inference engine)
    self.tokenizer_manager, self.template_manager, self.scheduler_info = _launch_subprocesses(
        server_args=server_args
    )

    # 启动 HTTP API 服务器
    fixed_http_port = self._base_http_port + self.replica_rank  # 30000 + replica_rank
    self._server_port, self._server_task = await run_unvicorn(app, server_args,
                                                               self._server_address,
                                                               fixed_port=fixed_http_port)
```

**端口分配规则**：
- `replica_rank=0` → `30000`
- `replica_rank=1` → `30001`
- `replica_rank=2` → `30002`
- ...

### 1.4 NCCL 多节点初始化

**源码位置**: `verl/workers/rollout/sglang_rollout/async_sglang_server.py:99-112`

```python
# node_rank=0 的节点创建 master address
if self.node_rank == 0:
    self._master_address = self._server_address
    self._master_port, self._master_sock = get_free_port(self._server_address)

# node_rank>0 的节点连接到 master
else:
    await server.launch_server.remote(master_address=master_address,
                                     master_port=master_port)
```

**NCCL 初始化地址格式**：
```python
dist_init_addr = f"[{self._master_address}]:{self._master_port}"  # IPv6
# 或
dist_init_addr = f"{self._master_address}:{self._master_port}"    # IPv4
```

---

## 2. 与命令行启动的区别

### 2.1 命令行启动方式 (传统方式)

```bash
# 单节点启动
python -m sglang.launch_server \
    --node-rank 0 \
    --nnode 1 \
    --model-path /path/to/model \
    --tp 1 \
    --dp 1 \
    --port 30000
```

**特点**：
- ✅ 简单直接，适合推理服务
- ❌ 无法与训练框架共享 GPU
- ❌ 需要手动管理多节点分布式
- ❌ 无法动态调整显存占用

### 2.2 VERL AgentLoop 启动方式

**核心区别**：

| 维度 | 命令行启动 | VERL AgentLoop |
|------|-----------|----------------|
| **进程管理** | 独立进程 | Ray Actor (托管) |
| **GPU 共享** | 独占 GPU | 与训练共享 (HYBRID) |
| **端口分配** | 手动指定 | 自动分配 (30000+rank) |
| **多节点** | 需要手动配置 | Ray 自动调度 |
| **显存管理** | 固定占用 | 动态 sleep/wake |
| **负载均衡** | 无 | AsyncLLMServerManager (least-requests + sticky session) |
| **容错机制** | 无 | Ray 自动重启 |
| **监控集成** | 手动配置 | 自动启用 Prometheus |

### 2.3 关键技术差异

#### A. Sticky Session (前缀缓存)

**命令行启动**：每个请求随机分配到不同的服务器

**VERL AgentLoop**：通过 `AsyncLLMServerManager` 实现 sticky session

```python
# verl/experimental/agent_loop/agent_loop.py:48-112
class AsyncLLMServerManager:
    def __init__(self, config, server_handles, max_cache_size=10000):
        # LRU 缓存：request_id → server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

        # 最小堆：(请求数, (hash(server), server))
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

    def _choose_server(self, request_id):
        # 如果已有 session，返回同一 server (启用 prefix caching)
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        # 选择当前最少请求的 server
        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1  # 增加请求计数
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server
```

**效果**：同一对话的多轮请求发送到同一服务器，充分利用 KV Cache。

#### B. KV Cache 动态管理

**命令行启动**：KV Cache 始终占用显存

**VERL AgentLoop**：通过 `sleep()` / `wake_up()` 动态释放/占用显存

```python
# verl/workers/rollout/sglang_rollout/async_sglang_server.py:196-203
async def sleep(self):
    if self.rollout_mode == RolloutMode.HYBRID:
        await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
    elif self.rollout_mode == RolloutMode.COLOCATED:
        # 释放 KV Cache 和权重显存
        obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
        await self.tokenizer_manager.release_memory_occupation(obj, None)

async def wake_up(self):
    if self.rollout_mode == RolloutMode.HYBRID:
        await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
    elif self.rollout_mode == RolloutMode.COLOCATED:
        # 重新占用 KV Cache 和权重显存
        obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])
        await self.tokenizer_manager.resume_memory_occupation(obj, None)
        await self.tokenizer_manager.flush_cache()
```

**效果**：训练时释放显存给训练使用，推理时重新占用。

#### C. Ray Actor 并发调度

**命令行启动**：单进程处理请求

**VERL AgentLoop**：多个 `AgentLoopWorker` Ray Actor 并发处理

```python
# verl/experimental/agent_loop/agent_loop.py:886-901
def _init_agent_loop_workers(self):
    self.agent_loop_workers = []
    num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers  # 例如 16

    node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"]]
    for i in range(num_workers):
        # Round-robin 调度到不同节点
        node_id = node_ids[i % len(node_ids)]
        self.agent_loop_workers.append(
            AgentLoopWorker.options(
                name=f"agent_loop_worker_{i}",
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=True
                ),
            ).remote(self.config, self.server_handles, self.rm_executor)
        )
```

**效果**：16 个 Worker 并发处理 768 个样本 (每个 Worker 48 个样本)。

---

## 3. 监控端点深度剖析

### 3.1 SGLang 原生监控端点

SGLang 提供以下 HTTP API 用于监控：

| 端点 | 方法 | 描述 | 返回内容 |
|------|------|------|----------|
| `/health` | GET | 健康检查 | `{"status": "ok"}` |
| `/health_generate` | GET | 生成服务健康检查 | `{"status": "ok"}` |
| `/get_server_info` | GET | 服务器配置信息 | 模型路径、显存配置、并行度等 |
| `/get_load` | GET | 负载信息 | 请求数量、KV Cache 占用等 |
| `/metrics` | GET | Prometheus 指标 | 所有性能指标 (Prometheus 格式) |
| `/flush_cache` | POST | 清空 Radix Cache | 缓存清理结果 |

### 3.2 `/get_server_info` 端点详解

**源码位置**: `sglang/srt/entrypoints/http_server.py:437-448`

```python
@app.get("/get_server_info")
async def get_server_info():
    # 获取每个 DP rank 的内部状态
    internal_states: List[Dict[Any, Any]] = (
        await _global_state.tokenizer_manager.get_internal_state()
    )
    return {
        **dataclasses.asdict(_global_state.tokenizer_manager.server_args),
        **_global_state.scheduler_info,
        "internal_states": internal_states,  # 每个 DP rank 的状态
        "version": __version__,
    }
```

**返回字段示例**：
```json
{
  "model_path": "/path/to/qwen3-4b",
  "max_total_num_tokens": 131072,
  "max_running_requests": 256,
  "tp_size": 1,
  "dp_size": 1,
  "context_length": 32768,
  "mem_fraction_static": 0.85,
  "internal_states": [
    {
      "dp_rank": 0,
      "num_running_reqs": 48,
      "num_used_tokens": 62453,
      "token_usage": 0.48,
      "cache_hit_rate": 0.72
    }
  ]
}
```

**关键监控字段**：
- `num_running_reqs`: 正在运行的请求数
- `num_used_tokens`: 已使用的 token 数量
- `token_usage`: KV Cache 占用率 (0-1)
- `cache_hit_rate`: Prefix Cache 命中率

### 3.3 `/metrics` 端点详解

**源码位置**: `sglang/srt/utils.py:1265-1276`

```python
def add_prometheus_middleware(app):
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
```

**关键 Metrics 列表**：

| Metric 名称 | 类型 | 描述 | 单位 |
|------------|------|------|------|
| `sglang:num_running_reqs` | Gauge | 正在运行的请求数 | 个 |
| `sglang:num_used_tokens` | Gauge | 已使用的 token 数 | 个 |
| `sglang:token_usage` | Gauge | KV Cache 占用率 | 比例 (0-1) |
| `sglang:swa_token_usage` | Gauge | SWA (sliding window attention) token 占用率 | 比例 (0-1) |
| `sglang:cache_hit_rate` | Gauge | Prefix Cache 命中率 | 比例 (0-1) |
| `sglang:gen_throughput` | Gauge | 生成吞吐量 | tokens/s |
| `sglang:num_queue_reqs` | Gauge | 队列中的请求数 | 个 |
| `sglang:avg_request_queue_latency` | Histogram | 请求队列延迟 | 秒 |

**源码位置**: `sglang/srt/metrics/collector.py:177-241`

```python
class SchedulerMetricsCollector:
    def __init__(self, labels: Dict[str, str]) -> None:
        from prometheus_client import Counter, Gauge, Histogram

        self.num_running_reqs = Gauge(
            name="sglang:num_running_reqs",
            documentation="The number of running requests.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.token_usage = Gauge(
            name="sglang:token_usage",
            documentation="The token usage.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.cache_hit_rate = Gauge(
            name="sglang:cache_hit_rate",
            documentation="The cache hit rate.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
```

### 3.4 KV Cache 计算逻辑

**源码位置**: `sglang/srt/managers/scheduler.py:1507-1508`

```python
def _get_token_info(self):
    num_used = self.total_num_available_tokens - self.current_available_tokens
    token_usage = num_used / self.max_total_num_tokens  # 占用率
    return num_used, token_usage, available_size, evictable_size
```

**关键变量**：
- `max_total_num_tokens`: 最大可用 token 数 (GPU 显存 / 每个token的显存占用)
- `total_num_available_tokens`: 总可用 token 数
- `current_available_tokens`: 当前剩余可用 token 数

**Prefix Cache 命中率计算**：

源码位置: `sglang/srt/managers/scheduler_metrics_mixin.py:135-142`

```python
cache_hit_rate = (
    self.num_cache_hits / self.num_cache_total
    if self.num_cache_total > 0
    else 0.0
)
self.stats.cache_hit_rate = cache_hit_rate
```

---

## 4. 为何现有监控可能失效

### 4.1 常见问题

#### 问题 1: `/metrics` 端点返回 404

**原因**：
- `enable_metrics=False` (被配置覆盖)
- 多进程模式下 Prometheus 环境变量未设置

**解决方案**：
```python
# verl/workers/rollout/sglang_rollout/async_sglang_server.py:154
"enable_metrics": engine_kwargs.pop("enable_metrics", True),  # 确保为 True
```

**验证**：
```bash
curl http://localhost:30000/metrics
# 应该返回 Prometheus 格式的指标
```

#### 问题 2: `token_usage` 始终为 0

**原因**：
- Scheduler 的 `update_stats` 未被调用
- 多进程模式下指标未正确聚合

**底层原因**：
SGLang 使用 `prometheus_client` 的 `multiprocess_mode="mostrecent"`，需要设置 `PROMETHEUS_MULTIPROC_DIR` 环境变量。

**源码位置**: `sglang/srt/utils.py:1254-1263`

```python
def set_prometheus_multiproc_dir():
    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        import tempfile
        prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
```

**解决方案**：
在 VERL 中，已自动调用 `set_prometheus_multiproc_dir()` (源码: `verl/workers/rollout/sglang_rollout/sglang_rollout.py:100-102`)

```python
if server_args.enable_metrics:
    set_prometheus_multiproc_dir()  # ⚠️ VERL 已处理
```

#### 问题 3: 监控脚本无法连接到服务器

**原因**：
- 端口分配错误 (非 30000)
- 防火墙阻止
- 服务器绑定在 `127.0.0.1` 而非 `0.0.0.0`

**解决方案**：

**1. 确认正确的端口**：
```bash
# VERL 中端口 = 30000 + replica_rank
# 检查实际运行的端口
bash scripts/find_sglang_servers.sh
```

**2. 检查服务器地址**：
```python
# verl/workers/rollout/sglang_rollout/async_sglang_server.py:94
self._server_address = ray.util.get_node_ip_address().strip("[]")
```

注意：IPv6 地址会被 `[]` 包裹，需要去除。

**3. 测试连接**：
```bash
# 测试健康检查
curl http://<server_ip>:30000/health_generate

# 测试 metrics
curl http://<server_ip>:30000/metrics
```

#### 问题 4: 多节点监控不完整

**原因**：
- 只监控了 node_rank=0 的服务器
- 其他节点的服务器未被发现

**解决方案**：
```bash
# 扫描所有可能的端口
python scripts/monitor_sglang.py --host localhost --scan 8
# 扫描端口 30000-30007
```

或手动指定所有服务器地址：
```bash
python scripts/monitor_sglang.py \
    --hosts node1:30000,node1:30001,node2:30000,node2:30001
```

---

## 5. 正确的监控方案

### 5.1 方法 1: 使用 VERL 内置监控脚本

**快速检查**：
```bash
# 检查所有服务器状态
bash scripts/check_sglang_servers.sh

# 查找运行的服务器
bash scripts/find_sglang_servers.sh
```

**持续监控**：
```bash
# 监控单个服务器
python scripts/monitor_sglang.py --host localhost --port 30000

# 监控多个服务器 (自动扫描)
python scripts/monitor_sglang.py --host localhost --scan 8

# JSON 输出 (便于脚本解析)
python scripts/monitor_sglang.py --host localhost --scan 8 --once --json
```

**输出示例**：
```
✅ Server 0 (localhost:30000)
   📦 Model: qwen3-4b
   🎯 Max Running Requests: 256
   📊 Max Total Tokens: 131072
   📏 Context Length: 32768
   📈 Metrics:
      Queue Requests: 12
      Running Requests: 48
      KV Cache Usage: 🟢 48.2%
      Throughput: 1245.3 tokens/s
      Cache Hit Rate: 72.1%
```

### 5.2 方法 2: 直接使用 SGLang API

**通过 `/get_server_info` 获取详细信息**：
```bash
curl -s http://localhost:30000/get_server_info | jq '.internal_states'
```

**通过 `/get_load` 获取实时负载**：
```bash
curl -s http://localhost:30000/get_load | jq .
```

**通过 `/metrics` 获取 Prometheus 指标**：
```bash
# 获取 KV Cache 占用率
curl -s http://localhost:30000/metrics | grep "sglang:token_usage"

# 获取请求数量
curl -s http://localhost:30000/metrics | grep "sglang:num_running_reqs"

# 获取吞吐量
curl -s http://localhost:30000/metrics | grep "sglang:gen_throughput"
```

### 5.3 方法 3: 使用 Ray Dashboard

VERL 基于 Ray，可以使用 Ray Dashboard 监控：

**启动 Dashboard**：
```bash
ray start --head --port=8265
```

**访问**：
```
http://localhost:8265
```

**可监控内容**：
- Ray Actor 状态 (SGLangHttpServer, AgentLoopWorker)
- GPU 使用率
- 内存占用
- 任务日志

### 5.4 方法 4: 集成 Prometheus + Grafana

**启用 SGLang Prometheus 端点**：
```yaml
# config/rollout/sglang_rollout.yaml
actor_rollout_ref:
  rollout:
    engine_kwargs:
      sglang:
        enable_metrics: true  # 确保启用
```

**配置 Prometheus**：
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'sglang'
    static_configs:
      - targets: ['localhost:30000', 'localhost:30001', 'localhost:30002']
    scrape_interval: 5s
```

**启动 Prometheus**：
```bash
prometheus --config.file=prometheus.yml
```

**访问 Prometheus UI**：
```
http://localhost:9090
```

**查询示例**：
```promql
# KV Cache 占用率
sglang:token_usage

# 平均吞吐量
rate(sglang:gen_throughput[1m])

# Cache 命中率
sglang:cache_hit_rate
```

### 5.5 方法 5: 自定义监控脚本

**Python 示例**：
```python
import aiohttp
import asyncio

async def monitor_sglang_server(host: str, port: int):
    """监控单个 SGLang 服务器"""
    url = f"http://{host}:{port}/get_server_info"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()

            for state in data.get("internal_states", []):
                print(f"DP Rank {state['dp_rank']}:")
                print(f"  Running Requests: {state['num_running_reqs']}")
                print(f"  Token Usage: {state['token_usage']:.1%}")
                print(f"  Cache Hit Rate: {state['cache_hit_rate']:.1%}")

# 监控多个服务器
async def monitor_all_servers():
    servers = ["localhost:30000", "localhost:30001", "localhost:30002"]
    tasks = [monitor_sglang_server(*s.split(":")) for s in servers]
    await asyncio.gather(*tasks)

asyncio.run(monitor_all_servers())
```

---

## 6. 底层源码分析

### 6.1 SGLang HTTP 服务器启动流程

**源码路径**: `sglang/srt/entrypoints/http_server.py`

```
1. 创建 FastAPI app (line 213)
   ↓
2. 设置 middleware (CORS, API Key, Prometheus) (line 1193-1195)
   ↓
3. 注册 API 端点 (line 344-1192)
   ├─ /health, /health_generate
   ├─ /get_server_info
   ├─ /get_load
   ├─ /generate
   ├─ /flush_cache
   └─ /metrics (通过 add_prometheus_middleware)
   ↓
4. 启动 Uvicorn server (line 1211-1229)
   ↓
5. 启动 tokenizer manager 和 scheduler (line 162-164)
   ↓
6. 等待请求
```

### 6.2 Prometheus Metrics 暴露机制

**源码位置**: `sglang/srt/utils.py:1265-1276`

```python
def add_prometheus_middleware(app):
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

    # 创建独立的 registry (避免与其他服务冲突)
    registry = CollectorRegistry()

    # 多进程模式：收集所有子进程的指标
    multiprocess.MultiProcessCollector(registry)

    # 创建 ASGI 应用
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # 添加到 FastAPI 路由
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
```

**关键点**：
- 使用 `multiprocess.MultiProcessCollector` 收集所有子进程指标
- `multiprocess_mode="mostrecent"` 确保只保留最新值
- 需要 `PROMETHEUS_MULTIPROC_DIR` 环境变量

### 6.3 KV Cache 统计实现

**源码位置**: `sglang/srt/managers/scheduler_metrics_mixin.py:95-142`

```python
def _update_stats_metrics(self):
    # 获取 token 使用信息
    num_used, token_usage, _, _ = self._get_token_info()

    # 计算 Cache 命中率
    cache_hit_rate = (
        self.num_cache_hits / self.num_cache_total
        if self.num_cache_total > 0
        else 0.0
    )

    # 更新统计
    self.stats.token_usage = round(token_usage, 2)
    self.stats.cache_hit_rate = cache_hit_rate

    # 上报到 Prometheus
    self.metrics_collector.log(
        iteration=self.iteration,
        stats=self.stats,
    )
```

**Metrics 上报**：

源码位置: `sglang/srt/metrics/collector.py:533-541`

```python
def log(self, iteration: int, stats: SchedulerStats):
    self._log_gauge(self.num_running_reqs, stats.num_running_reqs)
    self._log_gauge(self.num_used_tokens, stats.num_used_tokens)
    self._log_gauge(self.token_usage, stats.token_usage)
    self._log_gauge(self.swa_token_usage, stats.swa_token_usage)
    self._log_gauge(self.cache_hit_rate, stats.cache_hit_rate)
```

### 6.4 VERL Agent Loop 调用链

```
RayPPOTrainer.fit()
  ↓
AgentLoopManager.generate_sequences()
  ↓
wake_up() (激活 KV Cache)
  ↓
AgentLoopWorker.generate_sequences() [并行]
  ↓
AsyncLLMServerManager._choose_server() (负载均衡)
  ↓
ToolAgentLoop.run() (多轮对话状态机)
  ↓
SGLangHttpServer.generate() [HTTP 调用]
  ↓
TokenizerManager.generate_request()
  ↓
Scheduler.schedule() (调度请求)
  ↓
InferenceEngine.forward() (模型推理)
  ↓
返回结果
  ↓
sleep() (释放 KV Cache)
```

---

## 7. 常见问题排查

### 7.1 SGLang 服务器启动失败

**症状**：
```
RuntimeError: SGLang http server should run on GPU node
```

**原因**：Ray Actor 未调度到 GPU 节点

**解决方案**：
```python
# 确保使用 NodeAffinitySchedulingStrategy
server = SGLangHttpServer.options(
    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id,
        soft=False,  # ⚠️ 必须为 False
    ),
).remote(...)
```

### 7.2 端口冲突

**症状**：
```
OSError: [Errno 48] Address already in use
```

**原因**：端口已被占用

**解决方案**：
```bash
# 查找占用端口的进程
lsof -i :30000

# 或使用 VERL 提供的脚本
bash scripts/find_sglang_servers.sh
```

**修改端口**：
```python
# verl/workers/rollout/sglang_rollout/async_sglang_server.py:96
self._base_http_port = 30000  # 修改为其他值
```

### 7.3 NCCL 初始化超时

**症状**：
```
RuntimeError: NCCL timeout in node_rank=1
```

**原因**：
- 防火墙阻止 NCCL 通信
- master 地址配置错误

**解决方案**：
```bash
# 测试 NCCL 连通性
# 在所有节点执行
nc -lzv <master_ip> <master_port>
```

### 7.4 监控数据不更新

**症状**：`token_usage` 始终为 0

**原因**：
- Scheduler 的 `update_stats` 未被调用
- Prometheus multiprocess 模式配置错误

**解决方案**：
```python
# 检查 enable_metrics
print(f"enable_metrics = {server_args.enable_metrics}")

# 检查 PROMETHEUS_MULTIPROC_DIR
import os
print(f"PROMETHEUS_MULTIPROC_DIR = {os.environ.get('PROMETHEUS_MULTIPROC_DIR')}")
```

### 7.5 显存泄漏

**症状**：显存占用持续增长

**原因**：
- KV Cache 未正确释放
- 请求未正常结束

**排查步骤**：
```bash
# 监控显存使用
watch -n 1 nvidia-smi

# 检查 KV Cache 占用
curl -s http://localhost:30000/get_server_info | jq '.internal_states[].token_usage'

# 手动清空 Cache
curl -X POST http://localhost:30000/flush_cache
```

---

## 附录

### A. 相关文件路径

| 文件 | 路径 | 说明 |
|------|------|------|
| AgentLoopManager | `verl/experimental/agent_loop/agent_loop.py:814-950` | Agent Loop 主管理器 |
| SGLangHttpServer | `verl/workers/rollout/sglang_rollout/async_sglang_server.py:48-308` | SGLang HTTP 服务器 |
| AsyncLLMServerManager | `verl/experimental/agent_loop/agent_loop.py:48-112` | 负载均衡管理 |
| 监控脚本 | `scripts/monitor_sglang.py` | 监控脚本 |
| SGLang HTTP Server | `sglang/srt/entrypoints/http_server.py` | SGLang HTTP API |
| Metrics Collector | `sglang/srt/metrics/collector.py` | Prometheus 指标收集 |
| Scheduler | `sglang/srt/managers/scheduler.py` | 调度器 |

### B. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PROMETHEUS_MULTIPROC_DIR` | 自动创建 | Prometheus 多进程目录 |
| `SGLANG_BLOCK_NONZERO_RANK_CHILDREN` | `0` | NCCL 子进程阻塞 |
| `CUDA_VISIBLE_DEVICES` | 自动设置 | 可见 GPU |
| `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` | `1` | Ray 不修改 CUDA 设备 |

### C. 参考链接

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang Metrics Integration - BytePlus](https://docs.byteplus.com/zh-CN/docs/vmp/SgLang_Integration)
- [VERL GitHub](https://github.com/volcengine/verl)
- [Prometheus Python Client](https://github.com/prometheus/client_python)

### D. 贡献者

如有问题或建议，请提交 Issue 或 PR 到 [VERL GitHub](https://github.com/volcengine/verl)。

---

**最后更新**: 2025-01-14
**文档版本**: v1.0.0
