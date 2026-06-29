# FSDP2 的 ZERO-2 初始化详解

## 🔍 关键发现：FSDP2 没有显式的 ShardingStrategy！

### FSDP1 vs FSDP2 的根本区别

#### FSDP1 (显式 Sharding Strategy)
```python
# verl/workers/fsdp_workers.py:468
use_zero2 = fsdp_config.get("use_zero2", False)
sharding_strategy = get_sharding_strategy(fsdp_mesh, use_zero2=use_zero2)

# FSDP1 使用显式的 ShardingStrategy 枚举
if fsdp_strategy == "fsdp":
    actor_module_fsdp = FSDP(
        actor_module,
        sharding_strategy=sharding_strategy,  # ← 显式指定 ZERO-2 或 ZERO-3
        device_mesh=self.device_mesh,
        # ... 其他参数
    )
```

#### FSDP2 (通过 mesh 维度隐式控制)
```python
# verl/workers/fsdp_workers.py:489-511
elif fsdp_strategy == "fsdp2":
    # FSDP2 不使用 sharding_strategy 参数！
    # 而是通过 mesh 的维度和 shard_placement_fn 来控制分片行为
    
    fsdp_kwargs = {
        "mesh": fsdp_mesh,                          # ← 关键：mesh 的维度决定分片策略
        "mp_policy": mp_policy,
        "offload_policy": cpu_offload,
        "reshard_after_forward": fsdp_config.reshard_after_forward,
        "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
    }
    
    # 应用 FSDP2
    apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
```

---

## 🔥 FSDP2 的 ZERO-2 实现原理

### 核心机制：Device Mesh 维度

**源码位置**: `verl/workers/fsdp_workers.py:98-105`

```python
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        # 1D mesh: 纯 FSDP
        device_mesh = init_device_mesh(
            device_name, 
            mesh_shape=(world_size,),      # 例如: (8,) - 8个GPU
            mesh_dim_names=["fsdp"]
        )
    else:
        # 2D mesh: HSDP (Hybrid Sharding Data Parallel)
        device_mesh = init_device_mesh(
            device_name, 
            mesh_shape=(world_size // fsdp_size, fsdp_size),  # 例如: (4, 2)
            mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh
```

### FSDP2 如何实现 ZERO-2？

**答案：通过 `mesh` 的配置 + PyTorch 内部逻辑**

在 FSDP2 中，`use_zero2` 参数**不直接影响** `fully_shard` API，而是：

1. **FSDP1 路径** (如果 `strategy=fsdp`):
   ```python
   use_zero2 = fsdp_config.get("use_zero2", False)
   sharding_strategy = get_sharding_strategy(mesh, use_zero2=use_zero2)
   # → ShardingStrategy.SHARD_GRAD_OP (ZERO-2)
   # → ShardingStrategy.FULL_SHARD (ZERO-3)
   
   FSDP(module, sharding_strategy=sharding_strategy, ...)
   ```

2. **FSDP2 路径** (如果 `strategy=fsdp2`):
   ```python
   # ⚠️ 注意：fsdp_kwargs 中没有 sharding_strategy 参数！
   fsdp_kwargs = {
       "mesh": fsdp_mesh,
       "mp_policy": mp_policy,
       "offload_policy": cpu_offload,
       "reshard_after_forward": fsdp_config.reshard_after_forward,
       "shard_placement_fn": get_shard_placement_fn(fsdp_size),
   }
   
   # fully_shard 根据 mesh 和其他参数自动决定分片策略
   fully_shard(module, **fsdp_kwargs)
   ```

---

## 🎯 FSDP2 的 ZERO-2 由什么控制？

### 答案：主要由 `reshard_after_forward` 参数控制

**源码**: PyTorch `fully_shard` API (torch >= 2.4)

```python
def fully_shard(
    module: nn.Module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Union[bool, int] = True,  # ← 关键参数
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    offload_policy: Optional[CPUOffloadPolicy] = None,
) -> nn.Module:
    """
    reshard_after_forward 控制分片行为:
    - True:  ZERO-3 行为 (前向后立即重分片参数，节省显存)
    - False: ZERO-2 行为 (保持参数 all-gathered，提高速度)
    - int:   部分重分片 (高级用法)
    """
```

### 你的配置中的 ZERO-2

```bash
# qwen3_agentloop_packing_resume_fsdp2.sh
actor_rollout_ref.actor.fsdp_config.use_zero2=True             # ← FSDP1 的配置
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True # ← FSDP2 的关键配置
```

**实际效果**:

| 配置 | FSDP1 行为 | FSDP2 行为 |
|------|-----------|-----------|
| `use_zero2=True` | ShardingStrategy.SHARD_GRAD_OP | **不影响 FSDP2** |
| `reshard_after_forward=True` | 不适用 | **前向后重分片** (更像 ZERO-3) |
| `reshard_after_forward=False` | 不适用 | **保持参数** (更像 ZERO-2) |

---

## 🔬 深入理解：FSDP2 的分片策略

### FSDP2 的参数分片

FSDP2 使用 **DTensor (Distributed Tensor)** 表示分片参数：

```python
# verl/utils/fsdp_utils.py:540-550
def get_shard_placement_fn(fsdp_size):
    """Choose the dimension that can divide fsdp_size to avoid padding"""
    
    def shard_placement_fn(param):
        shape = list(param.shape)
        # 选择可以被 fsdp_size 整除的维度进行分片
        for i in range(len(shape)):
            if shape[i] % fsdp_size == 0:
                return Shard(i)  # ← 返回 DTensor 的 Shard 维度
        return Shard(0)
    
    return shard_placement_fn
```

**示例**:

```python
# 假设参数形状: (4096, 4096)
# fsdp_size = 8

# FSDP2 会将参数分片为 DTensor:
# 每个 GPU 持有: (4096, 512) - 沿第 1 维分片
# 而不是 FSDP1 的 FlatParameter 表示
```

---

## 📊 FSDP2 ZERO-2 vs ZERO-3 对比

### ZERO-2 (reshard_after_forward=False)

```
前向传播:
├─ all-gather 参数 (通信)
├─ 执行前向
└─ 保持参数 (不重分片) ✅

反向传播:
├─ 使用已有的完整参数 ✅ (无需通信)
├─ 执行反向
├─ reduce-scatter 梯度 (通信)
└─ 更新优化器状态

优点: 速度快 (减少通信)
缺点: 显存占用高 (保持完整参数)
```

### ZERO-3 (reshard_after_forward=True)

```
前向传播:
├─ all-gather 参数 (通信)
├─ 执行前向
└─ 立即重分片参数 ✅ (节省显存)

反向传播:
├─ all-gather 参数 (通信) ❌
├─ 执行反向
├─ 重分片参数
├─ reduce-scatter 梯度 (通信)
└─ 更新优化器状态

优点: 显存占用低
缺点: 速度慢 (更多通信)
```

---

## 🎯 实际建议：FSDP2 如何配置 ZERO-2？

### 方案 1：真正的 ZERO-2 (更快)

```bash
actor_rollout_ref.actor.strategy=fsdp2 \
actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \  # ← 关键：保持参数
actor_rollout_ref.actor.fsdp_config.use_zero2=True \              # ← 仅影响 FSDP1
```

**效果**:
- ✅ 参数保持 all-gathered，无重分片
- ✅ 反向传播无需重新 all-gather
- ✅ 速度最快
- ❌ 显存占用高

### 方案 2：混合策略 (你当前的配置)

```bash
actor_rollout_ref.actor.strategy=fsdp2 \
actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \   # ← 前向后重分片
actor_rollout_ref.actor.fsdp_config.use_zero2=True \
```

**效果**:
- ✅ 显存占用低 (前向后重分片)
- ⚠️ 速度中等 (反向需要重新 all-gather)
- 更接近 ZERO-3 的行为

---

## 🔄 修改建议：真正的 FSDP2 ZERO-2

如果你想要**最大化速度**，建议修改为：

```bash
# qwen3_agentloop_packing_resume_fsdp2.sh

# Actor - 真正的 ZERO-2 (速度优先)
actor_rollout_ref.actor.strategy=fsdp2 \
actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \  # ← 改为 False
actor_rollout_ref.actor.fsdp_config.use_zero2=True \
actor_rollout_ref.actor.fsdp_config.offload_policy=False \

# Ref - 显存优先 (可以用 ZERO-3)
actor_rollout_ref.ref.strategy=fsdp2 \
actor_rollout_ref.ref.fsdp_config.strategy=fsdp2 \
actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \     # ← Ref 可以保持 True
actor_rollout_ref.ref.fsdp_config.use_zero2=True \
```

---

## 📈 性能对比（FSDP2）

| 配置 | 通信次数/step | 显存占用 | 训练速度 | 适用场景 |
|------|--------------|---------|---------|---------|
| **reshard_after_forward=False** | 1次 all-gather | 高 | **最快** ⭐⭐⭐⭐⭐ | 显存充足 |
| **reshard_after_forward=True** | 2次 all-gather | 低 | 中等 ⭐⭐⭐ | 显存紧张 |
| **reshard_after_forward=int** | 部分 all-gather | 中等 | 较快 ⭐⭐⭐⭐ | 高级用法 |

---

## 💡 总结

### FSDP1 的 ZERO-2
```python
use_zero2 = True → ShardingStrategy.SHARD_GRAD_OP
```

### FSDP2 的 ZERO-2
```python
# 方式1: reshard_after_forward=False (推荐，真正的 ZERO-2)
reshard_after_forward = False → 保持参数，减少通信

# 方式2: 通过 mesh 配置 (PyTorch 内部逻辑)
mesh + shard_placement_fn → DTensor 分片策略
```

### 你的配置

**当前**: `reshard_after_forward=True` → 更接近 ZERO-3 行为

**建议**: 如果显存充足，改为 `reshard_after_forward=False` → 真正的 ZERO-2，更快

### 关键区别

| 特性 | FSDP1 | FSDP2 |
|------|-------|-------|
| **ZERO-2 控制** | `use_zero2=True` | `reshard_after_forward=False` |
| **参数表示** | FlatParameter | DTensor |
| **API** | `FSDP()` 类 | `fully_shard()` 函数 |
| **显式策略** | ShardingStrategy 枚举 | mesh + 行为参数 |

**FSDP2 更灵活，但需要理解 `reshard_after_forward` 的含义！**
