# FSDP1 vs FSDP2 深度对比指南

## 📋 概述

PyTorch 提供了两种 FSDP (Fully Sharded Data Parallel) 实现：
- **FSDP1**：基于 `torch.distributed.fsdp.FSDP` 类（PyTorch 1.11+）
- **FSDP2**：基于 `torch.distributed._composable.fsdp.fully_shard` API（PyTorch 2.4+）

FSDP2 是 PyTorch 团队对 FSDP 的重新设计，采用更现代化的可组合（composable）架构。

---

## 🔄 核心架构差异

### FSDP1 架构

```
┌─────────────────────────────────────┐
│         FSDP Wrapper Module         │
│  ┌─────────────────────────────┐    │
│  │    Original Model Module    │    │
│  │  ┌───────┐  ┌───────┐      │    │
│  │  │Layer 1│  │Layer 2│ ...  │    │
│  │  └───────┘  └───────┘      │    │
│  └─────────────────────────────┘    │
│         (Flat Parameters)           │
└─────────────────────────────────────┘
```

**特点**：
- 包装式设计，FSDP 作为外层容器
- 参数被展平为一维张量（FlatParameter）
- 模块结构被修改

### FSDP2 架构

```
┌─────────────────────────────────────┐
│       Original Model Module         │
│  ┌───────────┐  ┌───────────┐      │
│  │  Layer 1  │  │  Layer 2  │ ...  │
│  │ (sharded) │  │ (sharded) │      │
│  └───────────┘  └───────────┘      │
│     (DTensor - Distributed Tensor)  │
└─────────────────────────────────────┘
```

**特点**：
- 非侵入式设计，不改变模块结构
- 使用 DTensor（分布式张量）表示分片参数
- 保持原始参数形状和名称

---

## 📊 功能对比表

| 特性 | FSDP1 | FSDP2 |
|------|-------|-------|
| **PyTorch 版本要求** | 1.11+ | **2.4+** |
| **API 风格** | 包装类 `FSDP(module)` | 函数式 `fully_shard(module)` |
| **参数表示** | FlatParameter（展平） | **DTensor（保持形状）** |
| **模块结构** | 被修改 | **保持不变** |
| **CPU Offload** | `CPUOffload` + 分离的 param/optimizer offload | **统一的 `CPUOffloadPolicy`** |
| **Forward 后重分片** | 不支持 | ✅ `reshard_after_forward` |
| **混合精度** | `MixedPrecision` | **`MixedPrecisionPolicy`** |
| **Checkpoint 兼容性** | 需要特殊处理 | **原生支持** |
| **与其他技术组合** | 困难 | **容易组合** |
| **调试友好性** | 较差 | **较好** |

---

## ⚙️ 配置参数对比

### FSDP1 配置

```python
from torch.distributed.fsdp import FSDP, CPUOffload, MixedPrecision, ShardingStrategy

model = FSDP(
    module,
    cpu_offload=CPUOffload(offload_params=True),  # 参数 offload
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32
    ),
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    auto_wrap_policy=auto_wrap_policy,
    device_id=device_id,
    sync_module_states=True,
    use_orig_params=False,
    forward_prefetch=False,  # FSDP1 专有
)
```

### FSDP2 配置

```python
from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    cast_forward_inputs=True
)

fsdp_kwargs = {
    "mesh": device_mesh,
    "mp_policy": mp_policy,
    "offload_policy": CPUOffloadPolicy(pin_memory=True),  # 统一 offload
    "reshard_after_forward": True,  # FSDP2 专有：前向后立即重分片
}

# 逐层应用 FSDP2
for layer in model.layers:
    fully_shard(layer, **fsdp_kwargs)
fully_shard(model, **fsdp_kwargs)
```

---

## 🔧 verl 中的配置差异

### FSDP1 启动脚本配置

```bash
# 策略选择
actor_rollout_ref.actor.strategy=fsdp \
actor_rollout_ref.actor.fsdp_config.strategy=fsdp \

# CPU Offload（分离控制）
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \

# FSDP1 专有参数
actor_rollout_ref.actor.fsdp_config.forward_prefetch=False \
actor_rollout_ref.actor.fsdp_config.use_orig_params=False \

# ZeRO 级别
actor_rollout_ref.actor.fsdp_config.use_zero2=True \
```

### FSDP2 启动脚本配置

```bash
# 策略选择
actor_rollout_ref.actor.strategy=fsdp2 \
actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \

# CPU Offload（统一控制）
actor_rollout_ref.actor.fsdp_config.offload_policy=False \

# FSDP2 专有参数
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \

# ZeRO 级别（FSDP2 同样支持）
actor_rollout_ref.actor.fsdp_config.use_zero2=True \
```

---

## 🚀 性能对比

### 训练速度

| 场景 | FSDP1 | FSDP2 | 说明 |
|------|-------|-------|------|
| **小模型 (<7B)** | 基准 | +5-10% | FSDP2 overhead 更低 |
| **大模型 (7B-70B)** | 基准 | +10-15% | DTensor 通信更高效 |
| **超长序列** | 基准 | +15-20% | `reshard_after_forward` 优化 |

### 显存效率

| 配置 | FSDP1 | FSDP2 |
|------|-------|-------|
| **基础显存** | 基准 | -5-10%（DTensor 更高效） |
| **reshard_after_forward** | ❌ 不支持 | ✅ 可节省 10-20% |
| **混合精度** | 基准 | 略优（cast_forward_inputs） |

---

## 📝 代码实现对比（verl 源码）

### FSDP1 实现 (`fsdp_workers.py`)

```python
if fsdp_strategy == "fsdp":
    actor_module_fsdp = FSDP(
        actor_module,
        cpu_offload=cpu_offload,
        param_init_fn=init_fn,
        auto_wrap_policy=auto_wrap_policy,
        device_id=get_device_id(),
        sharding_strategy=sharding_strategy,  # FULL_SHARD/SHARD_GRAD_OP
        mixed_precision=mixed_precision,
        sync_module_states=True,
        device_mesh=self.device_mesh,
        use_orig_params=self.use_orig_params,
        forward_prefetch=fsdp_config.get("forward_prefetch", False),
    )
```

### FSDP2 实现 (`fsdp_workers.py`)

```python
elif fsdp_strategy == "fsdp2":
    assert CPUOffloadPolicy is not None, "PyTorch >= 2.4 required"
    
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, 
        reduce_dtype=reduce_dtype, 
        cast_forward_inputs=True
    )
    
    fsdp_kwargs = {
        "mesh": fsdp_mesh,
        "mp_policy": mp_policy,
        "offload_policy": cpu_offload,
        "reshard_after_forward": fsdp_config.reshard_after_forward,
        "shard_placement_fn": get_shard_placement_fn(fsdp_size),
    }
    
    full_state = actor_module.state_dict()
    apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
    fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
    actor_module_fsdp = actor_module
```

---

## 🎯 选择建议

### 选择 FSDP1 的场景

1. **PyTorch 版本 < 2.4**
2. **需要与旧代码兼容**
3. **使用特定的 `auto_wrap_policy`**
4. **需要 `forward_prefetch` 优化**

### 选择 FSDP2 的场景

1. **PyTorch >= 2.4**（推荐 2.5+）
2. **需要最佳性能和显存效率**
3. **长序列训练**（利用 `reshard_after_forward`）
4. **需要与其他 PyTorch 技术组合**（如 torch.compile）
5. **调试需求高**（保持原始模块结构）
6. **Checkpoint 兼容性要求高**

---

## ⚠️ 迁移注意事项

### 从 FSDP1 迁移到 FSDP2

1. **检查 PyTorch 版本**：确保 >= 2.4
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

2. **修改配置参数**：
   ```bash
   # 旧配置 (FSDP1)
   actor_rollout_ref.actor.fsdp_config.param_offload=True
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
   
   # 新配置 (FSDP2)
   actor_rollout_ref.actor.fsdp_config.offload_policy=True
   ```

3. **Checkpoint 处理**：
   - FSDP1 checkpoint 可能需要转换才能被 FSDP2 加载
   - 建议从头开始训练或使用 HuggingFace 格式 checkpoint

4. **移除 FSDP1 专有参数**：
   - `forward_prefetch`（FSDP2 不需要）
   - `use_orig_params`（FSDP2 默认保持原参数）

---

## 📚 参考资料

- [PyTorch FSDP2 官方文档](https://pytorch.org/docs/stable/fsdp.html)
- [FSDP2 Design RFC](https://github.com/pytorch/pytorch/issues/114299)
- [verl FSDP 实现](https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py)

---

## 🔬 总结

| 维度 | 推荐 |
|------|------|
| **新项目** | FSDP2 |
| **PyTorch < 2.4** | FSDP1 |
| **最佳性能** | FSDP2 |
| **最佳兼容性** | FSDP1 |
| **长序列训练** | FSDP2 + `reshard_after_forward=True` |
| **显存受限** | FSDP2 + `offload_policy=True` |
| **速度优先** | FSDP2 + `use_zero2=True` + `offload_policy=False` |

**总体建议**：如果你的 PyTorch 版本 >= 2.4，优先选择 FSDP2，它代表了 PyTorch 分布式训练的未来方向。

