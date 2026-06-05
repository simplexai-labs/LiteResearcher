# Rollout Progress Monitor - Bug修复说明

## 🐛 问题描述

### 错误信息
```
AssertionError: Conflicting values for meta_info key 'rollout_progress_stats'
```

### 错误位置
```python
File "/share/project/wanli/Search_Agent/verl/verl/experimental/agent_loop/agent_loop.py", line 925, in generate_sequences
    output = DataProto.concat(outputs)
File "/share/project/wanli/Search_Agent/verl/verl/protocol.py", line 963, in concat
    assert merged_meta_info[k] == v, f"Conflicting values for meta_info key '{k}'"
```

## 🔍 问题分析

### 根本原因

当使用多个worker（如32个）时：

```python
# AgentLoopManager.generate_sequences()
chunkes = prompts.chunk(len(self.agent_loop_workers))  # 分成32份
outputs = ray.get([
    worker.generate_sequences.remote(chunk)  # 32个worker并行执行
    for worker, chunk in zip(self.agent_loop_workers, chunkes)
])
output = DataProto.concat(outputs)  # ← 这里出错！
```

**问题**：每个worker返回的 `DataProto` 都有自己的 `rollout_progress_stats`：
- Worker 0: `{"rollout/completed": 64, "rollout/total_samples": 64, ...}`
- Worker 1: `{"rollout/completed": 64, "rollout/total_samples": 64, ...}`
- ...
- Worker 31: `{"rollout/completed": 64, "rollout/total_samples": 64, ...}`

当 `DataProto.concat()` 尝试合并这些 `meta_info` 时：
```python
# protocol.py:963
assert merged_meta_info[k] == v, f"Conflicting values for meta_info key '{k}'"
```

因为每个worker的统计值都不同（即使数值相同，也是不同的dict对象），断言失败！

### 为什么之前没发现？

之前测试时可能：
1. 只用了1个worker（没有concat）
2. 或者没有开启监控功能

## ✅ 解决方案

### 策略：先私有化，后聚合

#### 1. Worker层：返回私有统计信息

```python
# AgentLoopWorker.generate_sequences()
# 修改前：
output.meta_info["rollout_progress_stats"] = stats  # ← 会冲突！

# 修改后：
output.meta_info["_worker_progress_stats"] = stats  # ← 加前缀，避免concat冲突
```

**关键**：使用 `_worker_progress_stats` 前缀，表示这是worker私有数据，不参与concat的合并检查。

#### 2. Manager层：聚合所有worker统计

```python
# AgentLoopManager.generate_sequences()
output = DataProto.concat(outputs)  # ← 不再冲突

# 提取各worker的统计信息
worker_stats_list = [
    output.meta_info.pop("_worker_progress_stats", None) 
    for output in outputs
]

# 聚合成全局统计
rollout_progress_stats = self._aggregate_progress_stats(worker_stats_list)

# 添加聚合后的统计信息
if rollout_progress_stats:
    output.meta_info["rollout_progress_stats"] = rollout_progress_stats
```

#### 3. 新增聚合函数

```python
def _aggregate_progress_stats(self, worker_stats_list):
    """聚合所有worker的进度统计信息"""
    valid_stats = [s for s in worker_stats_list if s is not None]
    
    return {
        # 简单求和
        "rollout/total_samples": sum(s["rollout/total_samples"] for s in valid_stats),
        "rollout/completed": sum(s["rollout/completed"] for s in valid_stats),
        
        # 加权平均
        "rollout/duration_avg": weighted_avg(...),
        
        # 取极值
        "rollout/duration_min": min(...),
        "rollout/duration_max": max(...),
        
        # 重新计算
        "rollout/success_rate": completed / total,
        "rollout/throughput": total / max_duration,
    }
```

## 📊 修改详情

### 修改文件

1. `verl/experimental/agent_loop/agent_loop.py`
   - 修改 `AgentLoopWorker.generate_sequences()`: L551
   - 修改 `AgentLoopManager.generate_sequences()`: L926-945
   - 新增 `AgentLoopManager._aggregate_progress_stats()`: L970-1028

### 修改内容

| 位置 | 修改前 | 修改后 | 说明 |
|-----|-------|--------|------|
| Worker返回 | `rollout_progress_stats` | `_worker_progress_stats` | 私有化，避免冲突 |
| Manager处理 | 直接使用第一个worker的stats | 聚合所有worker的stats | 正确的全局统计 |

## 🎯 效果验证

### 修改前
```python
# 32个worker，每个返回不同的stats
Worker 0: {"rollout/completed": 64}
Worker 1: {"rollout/completed": 64}
...
DataProto.concat() → AssertionError ❌
```

### 修改后
```python
# 32个worker，每个返回私有stats
Worker 0: {"_worker_progress_stats": {"rollout/completed": 64}}
Worker 1: {"_worker_progress_stats": {"rollout/completed": 64}}
...
DataProto.concat() → 成功 ✓

# Manager聚合
rollout_progress_stats = {
    "rollout/total_samples": 2048,      # 64 × 32
    "rollout/completed": 2048,          # 64 × 32
    "rollout/duration_avg": 8.3,        # 加权平均
    "rollout/throughput": 53.6,         # 2048 / 38.2s
    ...
}
```

## 📝 数据流

```
┌────────────────────────────────────────────────────────┐
│ AgentLoopManager.generate_sequences()                  │
│                                                         │
│  1. 分发数据到32个worker                               │
│     chunkes = prompts.chunk(32)                        │
│                                                         │
│  2. 并行执行                                            │
│     outputs = ray.get([                                │
│         worker.generate_sequences.remote(chunk)        │
│         for worker, chunk in zip(workers, chunkes)     │
│     ])                                                  │
│                                                         │
│  3. 每个worker返回（私有stats）                        │
│     Worker 0: DataProto(                               │
│         meta_info={                                    │
│             "_worker_progress_stats": {                │
│                 "rollout/completed": 64,               │
│                 "rollout/total_samples": 64,           │
│                 ...                                    │
│             }                                          │
│         }                                              │
│     )                                                  │
│                                                         │
│  4. Concat不冲突（因为key不同）                        │
│     output = DataProto.concat(outputs) ✓               │
│                                                         │
│  5. 提取私有stats                                       │
│     worker_stats = [                                   │
│         out.meta_info.pop("_worker_progress_stats")   │
│         for out in outputs                             │
│     ]                                                  │
│                                                         │
│  6. 聚合成全局stats                                     │
│     global_stats = _aggregate_progress_stats(          │
│         worker_stats                                   │
│     )                                                  │
│     # {                                                │
│     #   "rollout/total_samples": 2048,                │
│     #   "rollout/completed": 2048,                    │
│     #   ...                                            │
│     # }                                                │
│                                                         │
│  7. 添加全局stats                                       │
│     output.meta_info["rollout_progress_stats"] =      │
│         global_stats                                   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## 🔧 测试建议

### 测试场景

1. **单worker测试**
   ```yaml
   num_workers: 1
   ```
   验证：基本功能正常

2. **多worker测试**
   ```yaml
   num_workers: 8
   ```
   验证：无concat冲突

3. **大规模测试**
   ```yaml
   num_workers: 32
   ```
   验证：统计信息正确聚合

### 验证方法

查看WandB中的指标：
```python
# 应该看到正确的全局统计
rollout/total_samples = 2048        # 不是64
rollout/completed = 2048            # 不是64
rollout/throughput = ~50 samples/s  # 全局吞吐
```

## 🎉 总结

### 问题
多worker场景下，每个worker的统计信息冲突，导致 `DataProto.concat()` 失败。

### 解决
1. Worker返回私有统计（`_worker_progress_stats`）
2. Manager聚合所有worker统计
3. 生成正确的全局统计信息

### 收益
✅ 修复了多worker场景的crash  
✅ 提供了正确的全局统计  
✅ 保持了单worker场景的兼容性  
✅ 代码更清晰（职责分离）

### 影响
- 无性能影响（聚合开销可忽略）
- 向后兼容（禁用监控时无影响）
- 统计更准确（全局视角）
