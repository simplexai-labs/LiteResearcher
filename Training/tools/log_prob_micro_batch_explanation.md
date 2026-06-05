# log_prob_micro_batch_size_per_gpu 参数详解

## 问题：这两个参数分别在算什么？

```bash
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
```

**简短回答**：这两个参数都是控制**micro batch大小**，但用于**不同的计算阶段**：

1. **`rollout.log_prob_micro_batch_size_per_gpu=4`**  
   → 用于 **Actor模型** 计算 **Old Log Probs**（PPO的重要采样）

2. **`ref.log_prob_micro_batch_size_per_gpu=4`**  
   → 用于 **Ref模型** 计算 **Reference Log Probs**（KL散度计算）

---

## 详细说明

### 1. PPO训练中的Log Prob计算阶段

在PPO训练的每个iteration中，需要计算多次log probability：

```
训练流程（每个iteration）：
┌─────────────────────────────────────────────────────────────┐
│ 1. Rollout 阶段                                              │
│    ↓ 生成 trajectories (使用 SGLang/vLLM)                   │
├─────────────────────────────────────────────────────────────┤
│ 2. Reward 计算                                               │
│    ↓ 计算 token-level scores                                │
├─────────────────────────────────────────────────────────────┤
│ 3. 💙 Old Log Prob 计算  ← rollout.log_prob_micro_batch_size │
│    ↓ 使用 Actor FSDP 重新计算 log_prob                      │
│    ↓ 目的：PPO importance sampling                          │
├─────────────────────────────────────────────────────────────┤
│ 4. 🧡 Ref Log Prob 计算  ← ref.log_prob_micro_batch_size    │
│    ↓ 使用 Ref FSDP 计算 reference log_prob                  │
│    ↓ 目的：KL散度约束 (KL(π||π_ref))                        │
├─────────────────────────────────────────────────────────────┤
│ 5. Advantage 计算                                            │
│    ↓ 使用 old_log_prob, ref_log_prob, rewards              │
├─────────────────────────────────────────────────────────────┤
│ 6. Actor 更新                                                │
│    ↓ PPO gradient update                                    │
└─────────────────────────────────────────────────────────────┘
```

---

### 2. 参数 1：`rollout.log_prob_micro_batch_size_per_gpu`

#### 2.1 用途：计算Old Log Probs

**调用链**：
```
Trainer (ray_trainer.py:1143)
    ↓
actor_rollout_wg.compute_log_prob(batch)
    ↓
fsdp_workers.py:971-1008 (compute_log_prob)
    ↓ 使用 Actor FSDP 模型
    ↓ micro_batch_size = config.rollout.log_prob_micro_batch_size_per_gpu
    ↓
dp_actor.py:297-389 (compute_log_prob)
    ↓ 将batch按micro_batch_size分割
    ↓ 逐个micro batch forward计算log_prob
```

#### 2.2 代码位置

**文件**：`verl/workers/fsdp_workers.py:984`

```python
def compute_log_prob(self, data: DataProto):
    """计算 Old Log Probs（PPO重要采样用）"""
    assert self._is_actor
    
    # 👇 从rollout配置读取micro batch size
    data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
    data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
    data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
    data.meta_info["temperature"] = self.config.rollout.temperature
    
    # 使用 Actor FSDP 模型计算
    with self.ulysses_sharding_manager:
        with adapter_ctx:
            output, entropys = self.actor.compute_log_prob(
                data=data, 
                calculate_entropy=True  # ✅ 计算entropy用于loss
            )
    
    return output  # 包含 old_log_probs 和 entropys
```

**文件**：`verl/workers/actor/dp_actor.py:297-335`

```python
def compute_log_prob(self, data: DataProto, calculate_entropy=False):
    """实际执行log prob计算的函数"""
    self.actor_module.eval()
    
    # 👇 读取micro batch size
    micro_batch_size = data.meta_info["micro_batch_size"]  # 从上面传入的
    use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
    
    if use_dynamic_bsz:
        # 动态批处理：根据token数量分割
        max_token_len = data.meta_info["max_token_len"]
        micro_batches, batch_idx_list = prepare_dynamic_batch(
            data, max_token_len=max_token_len
        )
    else:
        # 固定批处理：按样本数分割
        micro_batches = data.split(micro_batch_size)  # 👈 这里分割！
    
    log_probs_lst = []
    entropy_lst = []
    
    # 👇 逐个micro batch计算
    for micro_batch in micro_batches:
        # micro_batch shape: [micro_batch_size, seq_len]
        output = self._forward_micro_batch(
            micro_batch, 
            calculate_entropy=calculate_entropy,
            temperature=temperature
        )
        log_probs_lst.append(output.log_probs)
        if calculate_entropy:
            entropy_lst.append(output.entropy)
    
    # 合并所有micro batch结果
    log_probs = torch.cat(log_probs_lst, dim=0)  # [total_batch_size, seq_len]
    entropys = torch.cat(entropy_lst, dim=0) if calculate_entropy else None
    
    return log_probs, entropys
```

#### 2.3 为什么需要分micro batch？

**显存限制**：
- 假设batch有64个样本，每个样本8k tokens
- 如果一次forward全部64个样本：64 × 8k = 512k tokens
- 显存可能不够（激活值、KV cache等）

**解决方案**：
- 分成 64÷4 = 16个micro batch
- 每个micro batch: 4 × 8k = 32k tokens ✅ 显存够用
- 串行计算16次，结果拼接

---

### 3. 参数 2：`ref.log_prob_micro_batch_size_per_gpu`

#### 3.1 用途：计算Reference Log Probs

**调用链**：
```
Trainer (ray_trainer.py:1163)
    ↓
ref_policy_wg.compute_ref_log_prob(batch)  或
actor_rollout_wg.compute_ref_log_prob(batch) (如果ref_in_actor)
    ↓
fsdp_workers.py:1012-1044 (compute_ref_log_prob)
    ↓ 使用 Ref FSDP 模型
    ↓ micro_batch_size = config.ref.log_prob_micro_batch_size_per_gpu
    ↓
dp_actor.py:297-389 (compute_log_prob)
    ↓ 将batch按micro_batch_size分割
    ↓ 逐个micro batch forward计算log_prob
```

#### 3.2 代码位置

**文件**：`verl/workers/fsdp_workers.py:1024`

```python
def compute_ref_log_prob(self, data: DataProto):
    """计算 Reference Log Probs（KL散度约束用）"""
    assert self._is_ref
    
    # 👇 从ref配置读取micro batch size
    micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
    data.meta_info["micro_batch_size"] = micro_batch_size
    data.meta_info["temperature"] = self.config.rollout.temperature
    data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
    data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
    
    # 使用 Ref FSDP 模型计算
    with self.ulysses_sharding_manager:
        data = data.to("cpu")  # 数据先到CPU
        output, _ = self.ref_policy.compute_log_prob(
            data=data, 
            calculate_entropy=False  # ❌ Ref不需要entropy
        )
    
    return output  # 包含 ref_log_prob
```

#### 3.3 与Old Log Prob的区别

| 项目 | Old Log Prob | Ref Log Prob |
|------|--------------|--------------|
| **使用模型** | Actor FSDP（正在训练的模型） | Ref FSDP（固定的参考模型） |
| **micro_batch_size** | `rollout.log_prob_micro_batch_size_per_gpu` | `ref.log_prob_micro_batch_size_per_gpu` |
| **计算entropy** | ✅ Yes（用于actor loss） | ❌ No（只需log_prob） |
| **用途** | PPO importance sampling | KL散度约束 |
| **调用位置** | `trainer:1143` | `trainer:1163` |
| **输入数据位置** | GPU | CPU→GPU（按需加载） |

---

### 4. 实际配置示例

#### 4.1 你的当前配置（从日志）

```yaml
actor_rollout_ref:
  rollout:
    log_prob_micro_batch_size_per_gpu: 4  # Actor计算old_log_prob时用
    log_prob_max_token_len_per_gpu: 16384
    log_prob_use_dynamic_bsz: False
  
  ref:
    log_prob_micro_batch_size_per_gpu: 4  # Ref计算ref_log_prob时用
    log_prob_max_token_len_per_gpu: 16384
    log_prob_use_dynamic_bsz: False
```

#### 4.2 配置建议

**根据显存和性能调整**：

| Batch Size | 平均Seq Len | 建议micro_batch_size | 理由 |
|-----------|-------------|---------------------|------|
| 8 | 2k tokens | 4 | 平衡性能和显存 |
| 8 | 4k tokens | 2 | 减少显存压力 |
| 8 | 8k tokens | 1 | 长序列需要更小batch |
| 16 | 2k tokens | 8 | 可以用更大micro batch |

**原则**：
- `micro_batch_size × avg_seq_len ≈ 8k-16k tokens` 是合理范围
- 太小（=1）：计算效率低，串行次数多
- 太大（=batch_size）：显存可能OOM

---

### 5. 为什么两个参数可以不同？

虽然你的配置中两者都是4，但实际上**可以设置不同值**：

#### 5.1 Actor可能需要更大micro batch

```yaml
actor_rollout_ref:
  rollout:
    log_prob_micro_batch_size_per_gpu: 8  # ✅ Actor有LoRA或更小
  ref:
    log_prob_micro_batch_size_per_gpu: 4  # Ref是完整模型，需要更多显存
```

**原因**：
- Actor使用LoRA：参数少，激活值小 → 可以用更大micro batch
- Ref是完整模型：参数多，激活值大 → 需要更小micro batch

#### 5.2 计算目的不同

```yaml
actor_rollout_ref:
  rollout:
    log_prob_micro_batch_size_per_gpu: 4  # 需要计算entropy，显存更多
  ref:
    log_prob_micro_batch_size_per_gpu: 8  # 只计算log_prob，显存更少
```

**原因**：
- Old Log Prob需要计算entropy → 额外显存开销
- Ref Log Prob只需log_prob → 可以用更大micro batch

---

### 6. 计算流程详细示例

假设：
- `train_batch_size = 8`（全局batch，8个样本）
- `rollout.log_prob_micro_batch_size_per_gpu = 4`
- `ref.log_prob_micro_batch_size_per_gpu = 4`
- 每个样本平均2048 tokens

#### 6.1 Old Log Prob计算流程

```python
# 输入：8个样本的batch
batch = {
    'input_ids': torch.Size([8, 2048]),      # [batch_size, seq_len]
    'attention_mask': torch.Size([8, 2048]),
    'position_ids': torch.Size([8, 2048]),
    'responses': torch.Size([8, 1024]),      # response部分
}

# 步骤1：分割成micro batches
micro_batch_size = 4  # 从 rollout.log_prob_micro_batch_size_per_gpu
micro_batches = batch.split(micro_batch_size)
# → [micro_batch_0, micro_batch_1]
# → 每个micro_batch: [4, 2048]

# 步骤2：串行计算每个micro batch
log_probs_lst = []
entropy_lst = []

for i, micro_batch in enumerate(micro_batches):
    print(f"Processing micro_batch {i}: shape {micro_batch['input_ids'].shape}")
    # micro_batch_0: [4, 2048]  → 4 samples × 2048 tokens = 8192 tokens
    # micro_batch_1: [4, 2048]  → 4 samples × 2048 tokens = 8192 tokens
    
    # Actor FSDP forward pass
    output = actor_module(
        input_ids=micro_batch['input_ids'],
        attention_mask=micro_batch['attention_mask'],
        position_ids=micro_batch['position_ids'],
    )
    # output.logits: [4, 2048, vocab_size]
    
    # 计算log_prob和entropy
    log_probs = compute_log_probs(output.logits, micro_batch['responses'])
    entropy = compute_entropy(output.logits)
    
    log_probs_lst.append(log_probs)  # [4, 1024]
    entropy_lst.append(entropy)      # [4, 1024]

# 步骤3：合并结果
old_log_probs = torch.cat(log_probs_lst, dim=0)  # [8, 1024]
entropys = torch.cat(entropy_lst, dim=0)          # [8, 1024]

print(f"Final old_log_probs shape: {old_log_probs.shape}")
# Output: Final old_log_probs shape: torch.Size([8, 1024])
```

**显存使用分析**：
```
单个micro batch显存：
  - 输入 (input_ids): 4 × 2048 × 2 bytes = 16 KB
  - 激活值 (中间层): 4 × 2048 × 2560 × 2 bytes ≈ 40 MB
  - 输出logits: 4 × 2048 × 151936 × 2 bytes ≈ 2.4 GB
  
如果不分micro batch（全部8个样本）：
  - 输出logits: 8 × 2048 × 151936 × 2 bytes ≈ 4.8 GB
  → 可能OOM！
```

#### 6.2 Ref Log Prob计算流程

```python
# 输入：相同的8个样本的batch
# 但数据先移到CPU
batch = batch.to("cpu")

# 步骤1：分割成micro batches
micro_batch_size = 4  # 从 ref.log_prob_micro_batch_size_per_gpu
micro_batches = batch.split(micro_batch_size)

# 步骤2：串行计算每个micro batch
ref_log_probs_lst = []

for i, micro_batch in enumerate(micro_batches):
    # 👇 每个micro batch单独移到GPU
    micro_batch = micro_batch.to("cuda")
    
    print(f"Processing micro_batch {i}: shape {micro_batch['input_ids'].shape}")
    
    # Ref FSDP forward pass
    output = ref_module(
        input_ids=micro_batch['input_ids'],
        attention_mask=micro_batch['attention_mask'],
        position_ids=micro_batch['position_ids'],
    )
    
    # 计算log_prob（不需要entropy）
    ref_log_probs = compute_log_probs(output.logits, micro_batch['responses'])
    
    ref_log_probs_lst.append(ref_log_probs.cpu())  # 移回CPU节省显存
    
    # 👇 micro batch处理完，释放显存
    del micro_batch, output
    torch.cuda.empty_cache()

# 步骤3：合并结果
ref_log_probs = torch.cat(ref_log_probs_lst, dim=0)  # [8, 1024]
```

**关键区别**：
- Old Log Prob：整个batch在GPU，分micro batch只是为了forward
- Ref Log Prob：batch在CPU，按micro batch逐个加载到GPU，更节省显存

---

### 7. 如何选择合适的micro_batch_size？

#### 7.1 决策因素

1. **显存容量**（最重要）
   ```python
   所需显存 ≈ micro_batch_size × avg_seq_len × hidden_dim × 2 bytes
              + micro_batch_size × avg_seq_len × vocab_size × 2 bytes  # logits
   ```

2. **计算效率**
   - 太小：串行次数多，overhead大
   - 太大：可能OOM或降低吞吐

3. **序列长度**
   - 短序列（<2k）：可以用较大micro_batch_size
   - 长序列（>8k）：必须用较小micro_batch_size

#### 7.2 推荐配置（Qwen3-4B，80GB A100）

| 平均Seq Len | rollout.micro_batch_size | ref.micro_batch_size | 理由 |
|-------------|-------------------------|----------------------|------|
| 1k-2k | 8 | 8 | 显存充足 |
| 2k-4k | 4 | 4 | **你的配置（推荐）** |
| 4k-8k | 2 | 2 | 中等长度 |
| 8k-16k | 1 | 1 | 长序列 |
| 16k-32k | 1 (dynamic_bsz=True) | 1 (dynamic_bsz=True) | 超长序列，启用动态batch |

#### 7.3 如何验证配置是否合适？

**方法1：监控显存使用**
```bash
# 训练时查看显存
watch -n 1 nvidia-smi

# 期望：
# - 显存使用率：70-90%（太低浪费，太高容易OOM）
# - 每张卡显存：~60-70GB / 80GB
```

**方法2：查看训练日志**
```bash
# 如果出现以下错误，说明micro_batch_size太大：
# CUDA out of memory. Tried to allocate XXX GB

# 解决：减小micro_batch_size
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2  # 从4减到2
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
```

**方法3：性能测试**
```bash
# 测试不同micro_batch_size的速度
# 记录 "old_log_prob" 和 "ref" 阶段的时间

# 示例（假设）：
# micro_batch_size=2:  old_log_prob=5.2s, ref=4.8s
# micro_batch_size=4:  old_log_prob=3.1s, ref=2.9s  ← 最优
# micro_batch_size=8:  old_log_prob=OOM
```

---

### 8. 常见问题

#### Q1: 为什么不直接用ppo_micro_batch_size_per_gpu？

**A**: PPO有三个不同的micro_batch_size：

```yaml
actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: 2    # 用于 PPO gradient update
  
  rollout:
    log_prob_micro_batch_size_per_gpu: 4  # 用于 compute old_log_prob
  
  ref:
    log_prob_micro_batch_size_per_gpu: 4  # 用于 compute ref_log_prob
```

**原因**：
- **PPO update**：需要backward，显存开销大 → 用更小的2
- **Log prob计算**：只需forward，显存开销小 → 可以用更大的4

#### Q2: 两个log_prob_micro_batch_size必须相同吗？

**A**: 不必相同，但通常设置相同值方便管理。

**可以不同的场景**：
```yaml
# 场景1：Actor用LoRA
actor_rollout_ref:
  model:
    lora_rank: 64  # Actor有LoRA
  rollout:
    log_prob_micro_batch_size_per_gpu: 8  # Actor小，可以用大batch
  ref:
    log_prob_micro_batch_size_per_gpu: 4  # Ref完整模型，用小batch
```

```yaml
# 场景2：Ref offload到CPU
actor_rollout_ref:
  ref:
    fsdp_config:
      param_offload: True  # Ref在CPU
    log_prob_micro_batch_size_per_gpu: 2  # offload慢，用更小batch减少等待
  rollout:
    log_prob_micro_batch_size_per_gpu: 4  # Actor在GPU，可以用大batch
```

#### Q3: 如果batch_size < micro_batch_size怎么办？

**A**: 不会分割，直接使用整个batch。

```python
# 例如：
train_batch_size = 2
micro_batch_size = 4

# 实际：
micro_batches = batch.split(4)
# → 只有1个micro_batch，大小为2（不会padding到4）
```

#### Q4: dynamic_bsz是什么？

**A**: 动态批处理，按**token数量**而非**样本数量**分割batch。

```yaml
actor_rollout_ref:
  rollout:
    log_prob_use_dynamic_bsz: True
    log_prob_max_token_len_per_gpu: 16384  # 每个micro batch最多16k tokens
```

**优点**：
- 适应变长序列（有的样本1k tokens，有的20k tokens）
- 每个micro batch token数接近，显存使用更均衡

**缺点**：
- 实现复杂，可能有bug
- 如果序列长度相近，不如固定batch简单

---

### 9. 总结

#### 核心要点

1. **两个参数的用途**：
   ```
   rollout.log_prob_micro_batch_size_per_gpu  → Actor计算old_log_prob
   ref.log_prob_micro_batch_size_per_gpu      → Ref计算ref_log_prob
   ```

2. **为什么需要micro batch**：
   - 显存限制：无法一次forward整个batch
   - 解决方案：分割成多个小batch，串行计算，结果拼接

3. **配置建议**（Qwen3-4B, 80GB GPU）：
   ```yaml
   # 你的当前配置（推荐保持）
   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu: 4
   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu: 4
   ```

4. **调优原则**：
   - 显存使用率70-90%：配置合理
   - OOM错误：减小micro_batch_size
   - 显存使用<50%：可以增大micro_batch_size提升速度

#### 代码位置速查

| 功能 | 文件 | 行号 | 关键变量 |
|------|------|------|---------|
| Actor compute_log_prob入口 | `fsdp_workers.py` | 971-1008 | `rollout.log_prob_micro_batch_size_per_gpu` |
| Ref compute_ref_log_prob入口 | `fsdp_workers.py` | 1012-1044 | `ref.log_prob_micro_batch_size_per_gpu` |
| 实际计算log_prob | `dp_actor.py` | 297-389 | `micro_batch_size` |
| Trainer调用old_log_prob | `ray_trainer.py` | 1143 | - |
| Trainer调用ref_log_prob | `ray_trainer.py` | 1163 | - |

---

## 附录：完整配置示例

```yaml
actor_rollout_ref:
  model:
    path: /path/to/model
    use_liger: True
    use_fused_kernels: True
    use_remove_padding: True
  
  actor:
    # PPO update时的micro batch size（需要backward）
    ppo_micro_batch_size_per_gpu: 2
    ppo_mini_batch_size: 8
  
  rollout:
    # Actor计算old_log_prob时的micro batch size（只需forward）
    log_prob_micro_batch_size_per_gpu: 4
    log_prob_max_token_len_per_gpu: 16384
    log_prob_use_dynamic_bsz: False
    
    # Rollout生成时的配置
    n: 8  # 每个prompt生成8个样本
    max_model_len: 49152
  
  ref:
    # Ref计算ref_log_prob时的micro batch size（只需forward）
    log_prob_micro_batch_size_per_gpu: 4
    log_prob_max_token_len_per_gpu: 16384
    log_prob_use_dynamic_bsz: False
    
    # Ref model的FSDP配置
    fsdp_config:
      param_offload: False  # 不offload到CPU
      use_zero2: True       # 使用ZeRO-2分片
```

