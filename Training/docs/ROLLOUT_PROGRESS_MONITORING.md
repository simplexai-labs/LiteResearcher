# Rollout Progress Monitoring 使用指南

## 功能介绍

这个监控系统专门用于追踪单个training step内的agent rollout进度，特别适合大batch（如256×8=2048样本）的场景。

### 核心功能

1. **实时进度条**：显示batch内样本完成进度
2. **性能统计**：追踪每个样本的完成时间、对话轮数、工具调用次数
3. **定期报告**：每完成一定比例（如5%）打印统计信息
4. **最终汇总**：rollout完成后展示详细统计
5. **指标记录**：自动将统计信息发送到WandB/SwanLab

## 快速开始

### 1. 默认启用（推荐）

监控功能已经**默认开启**，无需任何配置。运行训练时会自动显示进度：

```bash
./qwen3_agentloop.sh
```

你会看到类似这样的输出：

```
[Step 5] Agent Rollout: 45%|████████████▌              | 921/2048 [02:15<02:47,  6.73sample/s, completed=921, failed=0, avg_time=8.2s]

[Step 5] Rollout Progress: 1024/2048 completed
  ⏱️  Duration: avg=8.3s, p50=7.9s, p95=15.2s
  ✅ Success: 1024, ❌ Failed: 0
```

### 2. 自定义配置

如果需要调整监控参数，可以在配置文件中设置：

```yaml
actor_rollout_ref:
  rollout:
    enable_progress_monitor: true  # 是否启用监控（默认true）
```

### 3. 禁用监控

如果你想完全关闭监控（不推荐，因为性能开销很小）：

```yaml
actor_rollout_ref:
  rollout:
    enable_progress_monitor: false
```

或者在命令行中：

```bash
python3 -m verl.trainer.main_ppo \
    ... \
    actor_rollout_ref.rollout.enable_progress_monitor=false
```

## 输出说明

### 进度条解读

```
[Step 5] Agent Rollout: 45%|████████████▌              | 921/2048 [02:15<02:47,  6.73sample/s, completed=921, failed=0, avg_time=8.2s]
```

- `45%`: 完成百分比
- `921/2048`: 已完成/总样本数
- `[02:15<02:47]`: 已用时间<预计剩余时间
- `6.73sample/s`: 处理速度
- `completed=921`: 成功完成的样本数
- `failed=0`: 失败的样本数
- `avg_time=8.2s`: 单个样本平均用时

### 定期报告（每5%）

```
[Step 5] Rollout Progress: 1024/2048 completed
  ⏱️  Duration: avg=8.3s, p50=7.9s, p95=15.2s
  ✅ Success: 1024, ❌ Failed: 0
```

- `avg`: 平均完成时间
- `p50`: 中位数（50%样本的完成时间）
- `p95`: 95分位数（95%样本的完成时间）

### 最终汇总

在rollout完成后会打印详细统计：

```
================================================================================
[Step 5] Rollout Complete Summary
================================================================================
📊 Total Samples:     2048
✅ Completed:         2048 (100.0%)
❌ Failed:            0 (0.0%)
⏱️  Total Time:        305.2s (5.1min)
⚡ Throughput:        6.71 samples/s

🔄 Agent Statistics:
   Avg Turns/Sample:  3.2
   Avg Tools/Sample:  4.8
   Total Turns:       6554
   Total Tool Calls:  9830

⏱️  Sample Duration:
   Min:  3.2s
   P50:  7.9s
   P95:  15.2s
   Max:  23.4s
   Avg:  8.3s

🐌 Slowest 5 Samples:
   1. Sample 1523: 23.4s (7 turns, 12 tools)
   2. Sample 892: 21.8s (6 turns, 10 tools)
   3. Sample 456: 19.3s (5 turns, 8 tools)
   4. Sample 1834: 18.7s (6 turns, 9 tools)
   5. Sample 234: 17.9s (5 turns, 7 tools)
================================================================================
```

这个汇总包含：
- **吞吐量统计**：总时间、平均速度
- **Agent统计**：每个样本的平均对话轮数、工具调用次数
- **时间分布**：最小、中位数、95分位、最大、平均时间
- **慢样本分析**：最慢的5个样本，帮助定位性能瓶颈

## WandB/SwanLab指标

以下指标会自动记录到日志系统：

| 指标名称 | 说明 |
|---------|------|
| `rollout/total_samples` | 总样本数 |
| `rollout/completed` | 完成数 |
| `rollout/failed` | 失败数 |
| `rollout/success_rate` | 成功率 |
| `rollout/duration_min` | 最短完成时间 |
| `rollout/duration_max` | 最长完成时间 |
| `rollout/duration_avg` | 平均完成时间 |
| `rollout/duration_p50` | 中位数完成时间 |
| `rollout/duration_p95` | 95分位完成时间 |
| `rollout/total_duration` | rollout总时间 |
| `rollout/throughput` | 吞吐量（样本/秒） |
| `rollout/avg_turns_per_sample` | 每样本平均对话轮数 |
| `rollout/avg_tools_per_sample` | 每样本平均工具调用数 |
| `rollout/total_turns` | 总对话轮数 |
| `rollout/total_tool_calls` | 总工具调用数 |

这些指标可以在WandB/SwanLab的仪表板中绘制趋势图，帮助你：
- 监控训练过程中rollout性能变化
- 发现性能瓶颈
- 对比不同配置的效果

## 性能开销

监控系统设计为轻量级，对训练性能影响极小：

- **CPU开销**：< 0.5%（主要是进度条更新）
- **内存开销**：< 10MB（存储2048个样本的状态）
- **IO开销**：几乎为0（仅在定期打印时输出）

在实测中，2048样本的rollout时间差异在1秒以内（总时间约5分钟）。

## 高级用法

### 调整打印频率

如果你觉得输出太频繁或太少，可以在代码中调整：

```python
# 在 agent_loop.py 中
monitor = RolloutProgressMonitor(
    total_samples=batch_size,
    step=global_step,
    enable_progress_bar=True,
    enable_logging=True,
    log_interval=50,  # 每50个样本打印一次（默认是batch_size//20）
)
```

### 在自定义Agent Loop中使用

如果你实现了自定义的Agent Loop，可以这样集成：

```python
from verl.utils.rollout_progress import RolloutProgressMonitor

async def my_generate_sequences(self, batch):
    monitor = RolloutProgressMonitor(
        total_samples=len(batch),
        step=self.global_step,
    )
    
    async with monitor:
        tasks = []
        for i in range(len(batch)):
            # 使用monitor.track_sample包装你的协程
            coro = self._my_agent_loop(batch[i])
            task = monitor.track_sample(i, coro)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # 获取统计信息
        stats = monitor.get_stats()
        # 可以将stats记录到日志系统
    
    return results
```

## 故障排除

### 进度条不显示

**原因**：某些终端不支持tqdm进度条

**解决方案**：
1. 检查是否在支持ANSI的终端中运行
2. 禁用进度条但保留日志：修改代码中的`enable_progress_bar=False`

### 统计信息不准确

**原因**：Agent Loop返回的结果缺少`num_turns`或`metrics`

**解决方案**：确保你的Agent Loop实现正确返回`AgentLoopOutput`，包含：
- `num_turns`: 对话轮数
- `metrics`: AgentLoopMetrics对象（包含tool_calls等）

### 性能下降明显

**原因**：不太可能，但如果确实遇到

**解决方案**：
1. 禁用进度条：`enable_progress_bar=False`
2. 减少日志频率：增大`log_interval`
3. 完全禁用：`enable_progress_monitor=false`

## 与现有监控的关系

这个监控系统**补充**了现有的监控体系：

| 监控层次 | 系统 | 监控内容 |
|---------|------|---------|
| Step级别 | tqdm (ray_trainer.py) | 训练总体进度 |
| **Batch级别** | **RolloutProgressMonitor** | **单个step内样本进度** ⭐ |
| Sample级别 | RolloutTrace (weave/mlflow) | 单个样本的详细trace |
| Tool级别 | Agent Metrics | 工具调用统计 |
| Server级别 | Prometheus | SGLang服务器指标 |

新增的监控填补了"Batch级别"的空白，让你能实时看到2048个样本的处理进度。

## 示例场景

### 场景1：发现性能瓶颈

你发现某个step的rollout特别慢，查看汇总：

```
🐌 Slowest 5 Samples:
   1. Sample 1523: 23.4s (7 turns, 12 tools)
   2. Sample 892: 21.8s (6 turns, 10 tools)
```

分析：
- 这些样本都有较多的对话轮数和工具调用
- 可能是prompt质量问题，导致agent需要多次尝试
- 可以检查这些样本的内容，优化prompt或数据

### 场景2：监控训练稳定性

在WandB中绘制`rollout/success_rate`的趋势图：
- 如果成功率逐渐下降，可能训练不稳定
- 如果`rollout/avg_turns_per_sample`突然增加，可能模型开始过度思考

### 场景3：对比配置效果

测试不同temperature对rollout的影响：
- 记录不同temperature下的`rollout/duration_avg`
- 记录`rollout/avg_tools_per_sample`
- 找到最优的配置

## 总结

这个监控系统为agent场景的大batch rollout提供了**实时、细粒度的进度追踪**，帮助你：

✅ 实时了解rollout进度，不再"盲等"  
✅ 快速发现性能瓶颈和异常样本  
✅ 记录详细统计数据，辅助调优  
✅ 零配置启用，性能开销极小  

现在就试试吧！
