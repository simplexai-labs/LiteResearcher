# 超长序列训练优化指南

针对你的场景：**Qwen3-4B + 45K tokens 序列 + 8×H20 (96GB)**

---

## 📊 你当前的配置分析

```bash
# 当前配置
max_response_length=45056        # ~45K tokens
max_model_len=49152              # ~49K tokens
ppo_micro_batch_size_per_gpu=1   # 每 GPU 1 个样本
n=8                              # 每个 prompt 生成 8 个响应
```

**显存消耗估算**（45K tokens, Qwen3-4B）：
| 组件 | 显存占用 |
|------|----------|
| 模型参数 (bf16) | ~8 GB |
| 梯度 | ~8 GB |
| 优化器状态 (ZeRO-2) | ~4 GB |
| KV Cache (45K) | ~18 GB |
| 激活值 (gradient checkpointing) | ~15 GB |
| **总计** | **~53 GB** |

你的 H20 有 96GB，还有余量可以优化！

---

## 🚀 优化方案汇总

| 优化项 | 当前状态 | 预期收益 | 难度 |
|--------|----------|----------|------|
| **Gradient Checkpointing** | ✅ 已开启 | 节省 60% 激活显存 | - |
| **use_remove_padding** | ✅ 已开启 | 节省 10-30% 计算 | - |
| **use_liger** | ✅ 已开启 | 节省 20-30% 显存 | - |
| **ZeRO-2** | ✅ 已开启 | 提速 30-50% | - |
| **禁用 CPU Offload** | ✅ 已开启 | 提速 20-40% | - |
| **Ulysses 序列并行** | ❌ 未开启 | 支持更长序列 | ⭐⭐ |
| **减少 n 值** | n=8 | 提速 2-4x | ⭐ |
| **Flash Attention** | 默认开启 | - | - |
| **entropy_checkpointing** | ❌ 未开启 | 节省 5-10% | ⭐ |
| **torch.compile** | 未开启 | 提速 10-20% | ⭐⭐ |

---

## 🔧 具体优化配置

### 1️⃣ 开启 Ulysses 序列并行（推荐）

**原理**：将长序列切分到多个 GPU 上并行处理

```bash
# 序列并行 size = 2 (2 个 GPU 处理一个序列)
actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
```

**效果**：
- 每个 GPU 只需处理 45K/2 = 22.5K tokens
- 显存减少 ~40%
- 可支持更长序列

**注意**：需要 `n_gpus_per_node` 能被 `ulysses_sequence_parallel_size` 整除

### 2️⃣ 开启 entropy_checkpointing

**原理**：重计算 entropy 而不是存储

```bash
actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
```

**效果**：节省 5-10% 激活显存

### 3️⃣ 减少 n 值（每样本生成数量）

**当前**：每个 prompt 生成 8 个响应
**建议**：减少到 4 或 2

```bash
# 从 n=8 减少到 n=4
actor_rollout_ref.rollout.n=4 \
```

**效果**：
- Rollout 时间减少 50%
- 显存峰值降低

### 4️⃣ 开启 torch.compile（实验性）

```bash
actor_rollout_ref.actor.fsdp_config.use_torch_compile=True \
```

**效果**：提速 10-20%（首次编译会较慢）

### 5️⃣ 增加 micro_batch_size（如果显存充足）

```bash
# 如果显存允许，尝试增加到 2
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
```

**效果**：提高 GPU 利用率

---

## 📝 推荐的优化脚本

```bash
#!/bin/bash
# qwen3_agentloop_optimized.sh - 超长序列优化版

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='google_search_browse_multiturn_grpo' \
    # ... 基础配置 ...
    
    # ============ 已有优化（保持） ============
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.use_zero2=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    
    # ============ 新增优化 ============
    # 1. Ulysses 序列并行（SP=2，每个序列由 2 个 GPU 处理）
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    
    # 2. Entropy 重计算（节省显存）
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    
    # 3. 减少每样本生成数量（可选，提速）
    # actor_rollout_ref.rollout.n=4 \
    
    # 4. torch.compile（可选，提速）
    # actor_rollout_ref.actor.fsdp_config.use_torch_compile=True \
    
    # ... 其他配置 ...
```

---

## ⚠️ Ulysses 序列并行注意事项

### 有效 GPU 数计算

```
有效 DP size = n_gpus / ulysses_sp_size

例如：
- 8 GPU + SP=2 → 有效 DP=4
- 8 GPU + SP=4 → 有效 DP=2
```

### 配置约束

```bash
# train_batch_size 需要能被有效 DP size 整除
data.train_batch_size=128  # 128 / 4 = 32 ✓

# ppo_mini_batch_size 也需要调整
actor_rollout_ref.actor.ppo_mini_batch_size=64  # 64 / 4 = 16 ✓
```

---

## 📊 优化效果预估

| 配置 | 显存占用 | 训练速度 | 说明 |
|------|----------|----------|------|
| **当前配置** | ~53 GB | 基准 | ZeRO-2 + Liger |
| **+ SP=2** | ~35 GB | +20% | 序列并行 |
| **+ entropy_ckpt** | ~32 GB | 持平 | 显存优化 |
| **+ n=4** | ~32 GB | +50% | 减少生成量 |
| **综合优化** | ~32 GB | **+70%** | 全部优化 |

---

## 🎯 针对你场景的最终推荐

### 方案 A：保守优化（低风险）

```bash
# 只添加 entropy_checkpointing
actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
```

### 方案 B：中度优化（推荐）

```bash
# entropy_checkpointing + 减少 n
actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
actor_rollout_ref.rollout.n=4 \
```

### 方案 C：激进优化（最大提速）

```bash
# Ulysses SP + entropy_checkpointing + 减少 n
actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
actor_rollout_ref.rollout.n=4 \
```

---

## 🔬 进阶优化（如果还需要更长序列）

### 支持 100K+ tokens 的方案

1. **增大 Ulysses SP**：SP=4 或 SP=8
2. **使用 FSDP2**：更好的显存管理
3. **Tensor Parallelism**：`tensor_model_parallel_size=2`
4. **Ring Attention**：需要自定义实现

### 显存极限优化

```bash
# 最小显存配置
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
```

---

## 📚 参考

- [Ulysses: Long Sequence Transformer](https://arxiv.org/abs/2309.14509)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)

