# Worker数量对Rollout速度的影响分析

## 🎯 核心问题

**Q: Worker开得越多，rollout速度会越快吗？**

**A: 不一定！** 有最优点，超过最优点会变慢。

## 📊 理论分析

### 理想情况（线性加速）

```
1个worker:  2048 samples ÷ 1  = 2048 samples/worker → 需要 305s
2个worker:  2048 samples ÷ 2  = 1024 samples/worker → 需要 153s  (快2倍)
4个worker:  2048 samples ÷ 4  = 512 samples/worker  → 需要 76s   (快4倍)
8个worker:  2048 samples ÷ 8  = 256 samples/worker  → 需要 38s   (快8倍)
16个worker: 2048 samples ÷ 16 = 128 samples/worker  → 需要 19s   (快16倍)
```

**理想加速比 = worker数量**

### 实际情况（受多种因素限制）

## 🔍 代码层面分析

### 1. Worker初始化（`agent_loop.py:885-900`）

```python
def _init_agent_loop_workers(self):
    self.agent_loop_workers = []
    num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
    
    node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
    for i in range(num_workers):
        # Round-robin scheduling over the all nodes
        node_id = node_ids[i % len(node_ids)]
        self.agent_loop_workers.append(
            AgentLoopWorker.options(
                name=f"agent_loop_worker_{i}",
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=True  # ← soft=True：允许调度到其他节点
                ),
            ).remote(self.config, self.server_handles, self.rm_executor)
        )
```

**关键点**：
- ✅ Round-robin调度：均匀分布到所有节点
- ⚠️  `soft=True`：如果目标节点资源不足，会调度到其他节点
- ⚠️  每个worker是独立的Ray Actor（有内存开销）

### 2. Batch分发（`agent_loop.py:902-937`）

```python
def generate_sequences(self, prompts: DataProto) -> DataProto:
    """Split input batch and dispatch to agent loop workers."""
    
    # 1. 数据切分
    chunkes = prompts.chunk(len(self.agent_loop_workers))
    
    # 2. 并行执行（关键！）
    outputs = ray.get(
        [
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
        ]
    )
    
    # 3. 结果合并
    output = DataProto.concat(outputs)
```

**性能特点**：
- ✅ `ray.get()`会等待**所有**worker完成
- ⚠️  总时间 = **最慢worker的时间**（木桶效应）
- ⚠️  数据切分和合并有开销（但很小）

### 3. 瓶颈识别（`agent_loop.py:939-959`）

```python
def _performance_metrics(self, metrics, output):
    # 最慢的样本决定总时间
    slowest = np.argmax(t_generate_sequences + t_tool_calls)
    timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
```

**关键洞察**：
- 框架已经内置了性能分析
- 会记录最慢样本的指标
- 可以用来识别瓶颈

## 🎯 性能瓶颈分析

### 瓶颈1: LLM Server容量 ⭐⭐⭐⭐⭐

**最重要的瓶颈！**

从代码看：
```python
# agent_loop.py:853-864
rollout_world_size = (
    tensor_model_parallel_size
    * data_parallel_size
    * pipeline_model_parallel_size
)
num_replicas = world_size // rollout_world_size
```

**实际情况**（以你的配置为例）：
```
GPU总数: 8 (n_gpus_per_node=8, nnodes=1)
Tensor Parallel: 1
Data Parallel: 1
Pipeline Parallel: 1
→ rollout_world_size = 1
→ num_replicas = 8 / 1 = 8

实际LLM Server数量: 8个
```

**关键结论**：
```
如果 num_workers ≤ num_replicas (8个):
  ✅ 每个worker有独立的LLM server
  ✅ 无竞争，近乎线性加速
  
如果 num_workers > num_replicas (8个):
  ⚠️  多个worker共享LLM server
  ⚠️  产生排队等待
  ❌ 速度反而下降！
```

**示例**：
```
8个worker + 8个LLM servers:
  Worker 0 → Server 0 (无等待)
  Worker 1 → Server 1 (无等待)
  ...
  Worker 7 → Server 7 (无等待)
  → 总时间: 38s

16个worker + 8个LLM servers:
  Worker 0 → Server 0 (无等待)
  Worker 1 → Server 1 (无等待)
  ...
  Worker 7 → Server 7 (无等待)
  Worker 8 → Server 0 (等待Worker 0完成)  ← 排队！
  Worker 9 → Server 1 (等待Worker 1完成)
  ...
  → 总时间: ~38s (没有提速，甚至因调度开销变慢)
```

### 瓶颈2: CPU资源

**每个worker需要CPU**：
- 解析工具调用
- 数据预处理
- 异步任务调度

**示例**：
```
8核CPU + 8个worker: ✅ 每个worker 1核，流畅
8核CPU + 16个worker: ⚠️  CPU竞争，上下文切换开销
8核CPU + 32个worker: ❌ 严重竞争，大量等待
```

### 瓶颈3: 工具服务（Search/Browse）

**共享资源**：
- Google Search API（有QPS限制）
- Browse服务（并发请求限制）

**示例**：
```
Search API QPS: 100/s

8个worker × 256 samples/worker × 平均5次search/sample = 10,240次请求
如果同时发起: 需要 10,240 / 100 = 102秒（受API限制）

16个worker: 仍然 ~102秒（瓶颈在API，不在worker）
```

### 瓶颈4: 内存

**每个worker的内存开销**：
```python
# 每个worker都会加载：
- tokenizer
- processor  
- reward_manager
- 样本数据缓存
```

**估算**：
```
每个worker: ~500MB-1GB
8个worker: 4-8GB
16个worker: 8-16GB
32个worker: 16-32GB  ← 可能OOM
```

### 瓶颈5: Ray调度开销

**Worker越多，开销越大**：
- Actor创建时间
- 消息传递延迟
- GC压力

## 📈 实际性能曲线

### 理论 vs 实际

```
加速比
  ^
8 |       理想线性加速
  |      ／
7 |     ／
  |    ／
6 |   ／
  |  ／        实际加速
5 | ／       ／￣￣￣＼
  |／      ／          ＼
4|      ／              ＼
  |    ／                 ＼
3|   ／                    ＼
  |  ／                      ＼
2| ／                         ＼
  |／                           ＼___
1|________________________________
  0  2  4  6  8 10 12 14 16 18 20 → Worker数量
         ↑
      最优点（≈LLM Server数量）
```

### 不同配置下的最优Worker数

| GPU数 | LLM Servers | 推荐Worker数 | 说明 |
|-------|-------------|--------------|------|
| 1 | 1 | 1-2 | CPU轻度并行 |
| 2 | 2 | 2-4 | 每GPU 1-2 worker |
| 4 | 4 | 4-8 | 每GPU 1-2 worker |
| 8 | 8 | **8-16** | **你的配置** ⭐ |
| 16 | 16 | 16-32 | 大规模训练 |

## 🧪 实验建议

### 测试不同Worker配置

```yaml
# 实验1: 基准（当前配置）
actor_rollout_ref.rollout.agent.num_workers=8

# 实验2: 减半
actor_rollout_ref.rollout.agent.num_workers=4

# 实验3: 加倍
actor_rollout_ref.rollout.agent.num_workers=16

# 实验4: 激进
actor_rollout_ref.rollout.agent.num_workers=32
```

### 监控指标

在WandB中对比：
```python
# 1. 总时间
rollout/total_duration

# 2. 吞吐量
rollout/throughput

# 3. 负载均衡
agent_loop/generate_sequences/min  # 最快worker
agent_loop/generate_sequences/max  # 最慢worker
差值越小 = 负载越均衡

# 4. 资源利用
ray_dashboard -> CPU utilization
ray_dashboard -> Memory usage
```

## 📋 推荐配置

### 你的环境（8×H20）

**当前配置分析**：
```
GPU: 8个
LLM Servers: 8个
Worker: 默认8个（推测）
```

**推荐配置**：

#### 选项1: 保守（稳定优先）
```yaml
actor_rollout_ref:
  rollout:
    agent:
      num_workers: 8  # = LLM Servers
```
**优点**：
- 1:1映射，无竞争
- 负载均衡好
- 稳定可靠

**缺点**：
- CPU利用率可能不足（如果CPU > 8核）

#### 选项2: 激进（性能优先）⭐ 推荐
```yaml
actor_rollout_ref:
  rollout:
    agent:
      num_workers: 16  # = 2 × LLM Servers
```
**优点**：
- 利用CPU多线程处理工具调用
- 工具服务等待时，LLM Server不闲置
- 可能提速20-30%

**缺点**：
- 轻微的LLM Server竞争
- 内存增加 ~4GB

#### 选项3: 极限测试
```yaml
actor_rollout_ref:
  rollout:
    agent:
      num_workers: 32
```
**目的**：测试极限，找到拐点

**预期**：可能反而变慢（排队严重）

### 判断标准

运行后查看日志：
```bash
# 好的情况（负载均衡）
Worker 0: 256/256 [00:38]  6.7 samples/s
Worker 1: 256/256 [00:38]  6.7 samples/s
Worker 2: 256/256 [00:39]  6.6 samples/s
...
最慢 - 最快 < 5s  ✅

# 坏的情况（负载不均）
Worker 0: 256/256 [00:38]  6.7 samples/s
Worker 1: 256/256 [00:38]  6.7 samples/s
Worker 2: 256/256 [00:58]  4.4 samples/s  ← 慢了50%！
...
最慢 - 最快 > 15s  ❌
```

## 🎯 最终建议

### 快速答案

**对于你的配置（8 GPU）**：

```
✅ 最佳: 8-16 workers
⚠️  可尝试: 4 workers（减少内存）
❌ 不推荐: > 16 workers（会变慢）
```

### 优化步骤

1. **先测试当前配置**
   ```bash
   num_workers=8  # 运行一次，记录时间
   ```

2. **测试加倍配置**
   ```bash
   num_workers=16  # 如果快了20%+，采用
   ```

3. **如果变慢了**
   ```bash
   num_workers=12  # 二分查找最优点
   ```

4. **检查瓶颈**
   - 如果CPU < 50%：瓶颈在LLM Server → 保持8 workers
   - 如果CPU 90%+：瓶颈在CPU → 减少到4-6 workers
   - 如果GPU利用率低：瓶颈在工具服务 → worker数量影响小

### 长期优化

如果确实需要更快：

1. **增加LLM Servers**（最有效）
   ```yaml
   # 使用更多GPU或降低tensor_parallel_size
   tensor_model_parallel_size: 1 → 1 (保持)
   # 使用多节点
   nnodes: 1 → 2  (16个LLM servers)
   num_workers: 16-32
   ```

2. **优化工具服务**
   - 部署多个Browse服务实例
   - 使用Search API的企业版（更高QPS）

3. **数据预处理**
   - 提前tokenize
   - 缓存常见工具响应

## 📚 总结

### 核心结论

1. **最优worker数 ≈ LLM Server数**
2. **可以略大（1.5-2倍），利用CPU并行处理工具**
3. **超过2倍，通常会变慢**
4. **要实际测试，理论只能参考**

### 记住这个公式

```
最优Worker数 = min(
    LLM Server数 × 1.5-2.0,  # 利用CPU并行
    CPU核心数,                # 避免CPU竞争
    可用内存 / 单Worker内存,   # 避免OOM
    工具服务并发限制           # 避免API限制
)
```

### 开始实验！

```bash
# 测试脚本
for workers in 4 8 12 16 20; do
    echo "Testing with $workers workers..."
    python3 -m verl.trainer.main_ppo \
        ... \
        actor_rollout_ref.rollout.agent.num_workers=$workers \
        2>&1 | tee log_workers_${workers}.log
    
    # 提取时间
    grep "Rollout Complete Summary" -A 10 log_workers_${workers}.log
done
```

找到最快的配置，采用它！🚀
