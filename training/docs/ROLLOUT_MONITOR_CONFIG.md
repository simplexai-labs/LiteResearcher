# Rollout Progress Monitor Configuration Examples

## 配置示例1：默认配置（推荐）

```yaml
actor_rollout_ref:
  rollout:
    # 监控功能默认开启，无需配置
    # enable_progress_monitor: true  # 默认为true
```

## 配置示例2：完全禁用监控

如果你想关闭监控（不推荐）：

```yaml
actor_rollout_ref:
  rollout:
    enable_progress_monitor: false
```

## 配置示例3：在命令行中控制

```bash
# 启用监控（默认）
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='your_config' \
    actor_rollout_ref.rollout.enable_progress_monitor=true

# 禁用监控
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='your_config' \
    actor_rollout_ref.rollout.enable_progress_monitor=false
```

## 高级配置：在代码中调整

如果需要更细粒度的控制，可以直接修改 `agent_loop.py`:

```python
# 在 verl/experimental/agent_loop/agent_loop.py 中

monitor = RolloutProgressMonitor(
    total_samples=batch_size,
    step=global_step,
    enable_progress_bar=True,      # 是否显示进度条
    enable_logging=True,            # 是否打印日志
    log_interval=max(10, batch_size // 20),  # 打印间隔（每N个样本）
)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `total_samples` | int | 必填 | batch中的样本总数 |
| `step` | int | 0 | 当前训练步数 |
| `enable_progress_bar` | bool | True | 是否显示tqdm进度条 |
| `enable_logging` | bool | True | 是否打印详细日志 |
| `log_interval` | int | 10 | 每N个样本完成后打印统计 |

### 调整建议

#### 大Batch场景（> 1000样本）

```python
monitor = RolloutProgressMonitor(
    total_samples=batch_size,
    step=global_step,
    enable_progress_bar=True,
    enable_logging=True,
    log_interval=batch_size // 20,  # 每5%打印一次
)
```

#### 小Batch场景（< 100样本）

```python
monitor = RolloutProgressMonitor(
    total_samples=batch_size,
    step=global_step,
    enable_progress_bar=True,
    enable_logging=True,
    log_interval=10,  # 每10个样本打印一次
)
```

#### 快速测试（不需要详细输出）

```python
monitor = RolloutProgressMonitor(
    total_samples=batch_size,
    step=global_step,
    enable_progress_bar=True,
    enable_logging=False,  # 只显示进度条，不打印统计
)
```

#### 生产环境（最小输出）

```python
monitor = RolloutProgressMonitor(
    total_samples=batch_size,
    step=global_step,
    enable_progress_bar=False,  # 不显示进度条
    enable_logging=False,       # 不打印日志
)
# 注意：即使关闭所有输出，统计信息仍会记录到WandB/SwanLab
```

## 环境变量控制

你也可以通过环境变量控制监控行为：

```bash
# 方式1：在shell脚本中设置
export VERL_ROLLOUT_MONITOR_ENABLED=false
./qwen3_agentloop.sh

# 方式2：直接在命令前设置
VERL_ROLLOUT_MONITOR_ENABLED=false ./qwen3_agentloop.sh
```

然后在代码中读取：

```python
import os

enable_monitor = os.getenv("VERL_ROLLOUT_MONITOR_ENABLED", "true").lower() == "true"
```

## 与其他监控系统集成

### 集成WandB

统计信息会自动记录到WandB，不需要额外配置。你可以在WandB Dashboard中看到：

- `rollout/duration_avg`: 平均完成时间
- `rollout/throughput`: 吞吐量
- `rollout/avg_turns_per_sample`: 每样本平均轮数
- 等等...

### 集成SwanLab

同样自动集成，无需配置。

### 集成Prometheus

如果你使用Prometheus，可以添加自定义exporter：

```python
# 在 monitor 中添加
from prometheus_client import Gauge

rollout_duration = Gauge('rollout_duration_seconds', 'Rollout duration')
rollout_throughput = Gauge('rollout_throughput', 'Rollout throughput')

# 在 __aexit__ 中更新
rollout_duration.set(self.end_time - self.start_time)
rollout_throughput.set(self.total_samples / (self.end_time - self.start_time))
```

## 故障排查

### 问题1：进度条刷新太频繁

**解决方案**：增加 `mininterval` 参数

```python
self.pbar = tqdm(
    total=self.total_samples,
    desc=f"[Step {self.step}] Agent Rollout",
    unit="sample",
    mininterval=1.0,  # 最少1秒刷新一次
)
```

### 问题2：日志输出太多

**解决方案**：增大 `log_interval`

```python
monitor = RolloutProgressMonitor(
    total_samples=batch_size,
    step=global_step,
    log_interval=200,  # 每200个样本打印一次
)
```

### 问题3：在非交互环境中运行

如果在Jupyter Notebook或某些CI环境中，tqdm可能显示异常。

**解决方案**：使用简单模式

```python
monitor = RolloutProgressMonitor(
    total_samples=batch_size,
    step=global_step,
    enable_progress_bar=False,  # 禁用进度条
    enable_logging=True,        # 使用文本日志
)
```

## 最佳实践

### 1. 开发阶段

```python
# 完整的输出，帮助调试
monitor = RolloutProgressMonitor(
    total_samples=batch_size,
    step=global_step,
    enable_progress_bar=True,
    enable_logging=True,
    log_interval=50,
)
```

### 2. 训练阶段

```python
# 适中的输出，不干扰训练
monitor = RolloutProgressMonitor(
    total_samples=batch_size,
    step=global_step,
    enable_progress_bar=True,
    enable_logging=True,
    log_interval=batch_size // 10,  # 每10%打印一次
)
```

### 3. 生产部署

```python
# 最小输出，但保留指标记录
monitor = RolloutProgressMonitor(
    total_samples=batch_size,
    step=global_step,
    enable_progress_bar=False,
    enable_logging=False,
)
# 统计信息仍会记录到WandB/SwanLab
```

## 总结

- ✅ **默认配置已经很好**，大多数情况下无需调整
- ✅ **性能开销极小**（< 1%），可以放心使用
- ✅ **灵活配置**，从详细到静默都可以调整
- ✅ **自动集成**，统计信息自动记录到日志系统
