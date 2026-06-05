# 🔥 DISKANN RAG 服务压测工具

这个目录包含三个压测脚本，用于模拟 AgentLoop 中的工具调用模式来压测 RAG 服务的各个组件。

## 📂 文件说明

```
benchmark/
├── benchmark_embedding.py    # Embedding 服务压测
├── benchmark_milvus.py       # Milvus 数据库压测
├── benchmark_query.py        # 完整查询流程压测 (Embedding + Milvus)
├── run_benchmark.sh          # 快速运行脚本
└── README.md                 # 本文档
```

## 🎯 设计理念

这些压测脚本完全模拟了 `tool_agent_loop.py` 中的调用模式：

1. **使用 `asyncio.gather` 并发执行** - 与 AgentLoop 中工具调用的方式一致
2. **支持分层并发** - 模拟 Workers → Samples → Tool Calls 的并发结构
3. **统计详细指标** - QPS、延迟分布(P50/P95/P99)、成功率等

### 对应代码

```python
# tool_agent_loop.py 中的并发调用模式
tasks = []
for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
    tasks.append(self._call_tool(...))
responses = await asyncio.gather(*tasks)
```

## 🚀 快速开始

### 1. Embedding 服务压测

```bash
# 基础压测 (100 并发，1000 请求)
python benchmark_embedding.py --concurrency 100 --total 1000

# 高并发压测 (500 并发，10000 请求)
python benchmark_embedding.py -c 500 -n 10000

# 批量文本压测 (每请求包含 8 个文本)
python benchmark_embedding.py -c 200 -n 5000 --batch-size 8
```

### 2. Milvus 数据库压测

```bash
# 基础压测 (100 并发，1000 请求，hybrid 搜索)
python benchmark_milvus.py --concurrency 100 --total 1000

# Dense 搜索压测
python benchmark_milvus.py -c 200 -n 5000 --search-type dense

# 高并发 Hybrid 搜索
python benchmark_milvus.py -c 300 -n 10000 --search-type hybrid
```

### 3. 完整查询流程压测

```bash
# 基础压测 (100 并发，1000 请求)
python benchmark_query.py --concurrency 100 --total 1000

# 高并发压测
python benchmark_query.py -c 300 -n 10000 --search-type hybrid

# 🔥 Rollout 模式压测 (模拟真实训练场景)
python benchmark_query.py --rollout-mode --workers 8 --samples-per-worker 16 --tools-per-sample 2 --turns 5
```

## 📊 Rollout 模式详解

Rollout 模式完全模拟了训练时的并发结构：

```
Rollout 并发层级:
├── Layer 1: Workers 并发 (num_workers=8)
│   ├── Layer 2: Samples 并发 (samples_per_worker=16)
│   │   └── Layer 3: Tool Calls 并发 (tools_per_sample=2)
│   │       └── 每个 Sample 有多个 Turns (turns=5)
```

### 示例配置

| 配置 | 说明 | 理论最大并发 |
|-----|------|------------|
| `--workers 8 --samples-per-worker 16 --tools-per-sample 2` | 标准配置 | 8×16×2 = 256 |
| `--workers 8 --samples-per-worker 32 --tools-per-sample 3` | 高并发配置 | 8×32×3 = 768 |
| `--workers 4 --samples-per-worker 8 --tools-per-sample 2` | 轻量配置 | 4×8×2 = 64 |

## 📈 输出示例

```
================================================================================
📊 完整查询流程压测结果
================================================================================

📈 总体统计:
   总请求数: 1000
   成功请求: 998
   失败请求: 2
   成功率: 99.80%
   总耗时: 45.23s
   QPS: 22.07 req/s

⏱️  端到端延迟 (ms):
   平均: 45.32
   中位数: 42.15
   P50: 42.15
   P95: 78.43
   P99: 125.67
   最小: 18.23
   最大: 245.89

🤖 Embedding 阶段 (ms):
   平均: 25.67
   P95: 45.32
   P99: 78.21

🔍 Milvus 搜索阶段 (ms):
   平均: 15.43
   P95: 28.76
   P99: 42.15

📊 时间占比分析:
   Embedding: 56.6%
   Milvus: 34.0%
   其他(网络等): 9.4%
================================================================================
```

## ⚙️ 参数说明

### 通用参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--url` | 服务地址 | http://localhost:8018/search |
| `--concurrency, -c` | 并发数 | 100 |
| `--total, -n` | 总请求数 | 1000 |
| `--search-type, -s` | 搜索类型 (hybrid/dense/sparse) | hybrid |
| `--limit, -l` | 每次返回结果数 | 10 |
| `--timeout, -t` | 请求超时(秒) | 60 |

### Rollout 模式参数

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--rollout-mode, -r` | 启用 Rollout 模式 | False |
| `--workers, -w` | Worker 数量 | 8 |
| `--samples-per-worker` | 每 Worker 的 Sample 数 | 16 |
| `--tools-per-sample` | 每 Sample 每 Turn 的工具调用数 | 2 |
| `--turns` | 每 Sample 的 Turn 数 | 5 |

## 🔧 服务地址配置

默认配置：
- Embedding 服务: `http://10.160.199.231:8028/embed`
- RAG 服务: `http://localhost:8018/search`

修改方式：
```bash
# 通过命令行参数
python benchmark_query.py --url http://your-server:8018/search

# 或修改脚本中的默认值
RAG_SERVICE_URL = "http://your-server:8018/search"
```

## 📝 注意事项

1. **确保服务已启动** - 运行压测前请确保 Embedding 服务和 RAG 服务都已启动
2. **逐步增加并发** - 建议从低并发开始，逐步增加以观察系统表现
3. **监控系统资源** - 压测时建议同时监控 CPU、GPU、内存使用情况
4. **网络稳定性** - 确保压测机器与服务之间网络稳定

## 🎯 典型压测场景

### 场景 1: 性能基准测试
```bash
# 测试服务的基础性能
python benchmark_query.py -c 50 -n 500
python benchmark_query.py -c 100 -n 1000
python benchmark_query.py -c 200 -n 2000
```

### 场景 2: 极限压力测试
```bash
# 找出服务的性能上限
python benchmark_query.py -c 500 -n 10000
python benchmark_query.py -c 1000 -n 20000
```

### 场景 3: 模拟真实训练负载
```bash
# 模拟 batch_size=128, 8 workers 的训练场景
python benchmark_query.py --rollout-mode \
    --workers 8 \
    --samples-per-worker 16 \
    --tools-per-sample 2 \
    --turns 10
```

### 场景 4: 单独测试各组件
```bash
# 先测试 Embedding
python benchmark_embedding.py -c 200 -n 5000

# 再测试 Milvus
python benchmark_milvus.py -c 200 -n 5000

# 最后测试完整流程
python benchmark_query.py -c 200 -n 5000
```

