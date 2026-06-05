# 分布式Rollout进度监控说明

## 问题：为什么显示256而不是2048？

### 场景分析

你的配置：
- **原始batch**: 256个prompt
- **GRPO n**: 8（每个prompt生成8个response）
- **实际样本**: 256 × 8 = 2048个样本
- **Agent Loop Workers**: 通常设置为8个（配置中的`num_workers`）

### 实际执行流程

```
                    原始Batch (256 prompts)
                            ↓
                    × rollout.n (8)
                            ↓
                    总样本数 (2048)
                            ↓
              ┌─────────────┴─────────────┐
              │  AgentLoopManager         │
              │  .generate_sequences()    │
              └─────────────┬─────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │ 分发到8个Worker（chunkes方法）      │
          └─────────────────┬─────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   Worker 0            Worker 1   ...      Worker 7
  (256 samples)      (256 samples)       (256 samples)
        │                   │                   │
        ▼                   ▼                   ▼
    Monitor 0           Monitor 1   ...     Monitor 7
    显示: 2/256        显示: 2/256         显示: 2/256
```

**关键**：每个Worker独立运行，只看到自己负责的256个样本！

## 解决方案：显示全局进度

### 更新后的输出格式

#### 进度条（每个Worker）
```
[Step 1] Worker 0/8 Rollout (Global: 2048 samples): 45%|████| 115/256 [02:15<02:47, 6.73sample/s]
[Step 1] Worker 1/8 Rollout (Global: 2048 samples): 42%|████| 108/256 [02:15<02:50, 6.45sample/s]
...
[Step 1] Worker 7/8 Rollout (Global: 2048 samples): 47%|████| 120/256 [02:15<02:44, 6.89sample/s]
```

**解读**：
- `Worker 0/8`: 当前是第0个worker，总共8个worker
- `(Global: 2048 samples)`: 全局总样本数
- `115/256`: 这个worker完成了115/256个样本
- **全局进度**: 大约 `(115+108+...+120) / 2048 ≈ 45%`

#### 定期统计
```
[Step 1] Worker 0/8 Rollout Progress: 128/256 completed
  🌐 Global Progress: ~1024/2048 (estimated)
  ⏱️  Duration: avg=8.3s, p50=7.9s, p95=15.2s
  ✅ Success: 128, ❌ Failed: 0
```

**新增**：
- `🌐 Global Progress`: 估计的全局进度（当前worker完成数 × 总worker数）

#### 最终汇总
```
================================================================================
[Step 1] Worker 0/8 - Rollout Complete Summary
================================================================================
🌐 Global Total:      2048 samples across 8 workers
📦 This Worker:       256 samples
✅ Completed:         256 (100.0%)
⏱️  Total Time:        38.2s (0.6min)
⚡ Throughput:        6.70 samples/s
...
================================================================================
```

**新增**：
- `🌐 Global Total`: 全局总样本数和worker数
- `📦 This Worker`: 当前worker的样本数

## 理解进度计算

### 单Worker视角
```python
current_worker_progress = completed / 256  # 例如: 115/256 = 45%
```

### 全局视角（估算）
```python
# 假设所有worker进度相似
global_progress = (completed * num_workers) / global_total
                = (115 * 8) / 2048
                = 920 / 2048
                = 45%
```

### 为什么是"估算"？

因为：
1. 不同worker处理速度不同（负载不均）
2. 某些worker可能快完成，某些才开始
3. 真实全局进度需要跨worker通信（开销大）

**解决方案**：
- 每个worker显示自己的准确进度
- 全局进度作为参考（标记为"estimated"）
- 最终所有worker完成后，总时间最准确

## 配置说明

### 查看Worker数量

在你的配置中查找：
```yaml
actor_rollout_ref:
  rollout:
    agent:
      num_workers: 8  # 这里定义了worker数量
```

或在日志中查找：
```
[INFO] Initializing 8 agent loop workers...
```

### 调整Worker数量

```yaml
# 方式1: 配置文件
actor_rollout_ref:
  rollout:
    agent:
      num_workers: 16  # 增加到16个worker

# 方式2: 命令行
actor_rollout_ref.rollout.agent.num_workers=16
```

**注意**：
- worker数量受CPU核心数限制
- 更多worker = 更高并发，但CPU开销更大
- 建议: worker数 = CPU核心数 或 2 × CPU核心数

## 性能分析

### 吞吐量计算

#### 单Worker吞吐量
```
Worker throughput = 256 samples / 38.2s = 6.7 samples/s
```

#### 全局吞吐量（理想）
```
Global throughput = 6.7 samples/s × 8 workers = 53.6 samples/s
Total time = 2048 samples / 53.6 samples/s = 38.2s
```

**注意**：实际可能略慢，因为：
1. Worker间负载不均
2. 调度开销
3. 最慢的worker决定总时间

### 优化建议

#### 如果某些Worker特别慢
```
Worker 0: 256/256 [00:38]  ✅ 6.7 samples/s
Worker 1: 256/256 [00:39]  ✅ 6.6 samples/s
Worker 2: 256/256 [00:52]  ⚠️  4.9 samples/s  ← 瓶颈！
...
```

**排查**：
1. 检查Worker 2的硬件资源
2. 查看是否分配到了更复杂的样本
3. 考虑重新分配负载

#### 启用负载均衡

在配置中（如果支持）：
```yaml
actor_rollout_ref:
  rollout:
    agent:
      enable_load_balancing: true  # 动态分配样本
```

## 监控最佳实践

### 1. 开发阶段
- 保持默认配置
- 观察所有worker的输出
- 找到最慢的worker

### 2. 生产部署
- 禁用进度条（减少输出）
  ```yaml
  actor_rollout_ref:
    rollout:
      enable_progress_monitor: true
      # 但在代码中设置 enable_progress_bar=False
  ```
- 只记录指标到WandB
- 定期检查worker性能均衡性

### 3. 调优阶段
- 对比不同worker数的性能
- 绘制worker吞吐量分布
- 找到最优配置

## 总结

### 核心理解
1. **256是单个worker的样本数**，不是错误
2. **2048是全局样本数**，分布在8个worker上
3. **每个worker独立监控**，并发执行
4. **全局进度是估算的**，但单worker进度是准确的

### 现在你会看到
```
[Step 1] Worker 0/8 Rollout (Global: 2048 samples): 45%|████| 115/256
[Step 1] Worker 1/8 Rollout (Global: 2048 samples): 42%|████| 108/256
...
```

**这是正确的！** 每个worker处理256个样本，8个worker总共处理2048个样本。

### 验证全局进度

在WandB中查看：
```
rollout/total_samples = 2048  ← 这是全局总数
rollout/completed = 2048      ← 所有worker完成总数
```

这些指标是聚合后的，显示真实的全局进度。
