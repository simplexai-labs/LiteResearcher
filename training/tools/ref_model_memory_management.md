# Ref Model 内存管理详解

## 问题：Ref Model 在计算后是否放在 CPU 中？

**简短回答**：**取决于配置**。在你当前的配置中（从日志分析），ref model **没有**offload到CPU，它一直保持在GPU上（使用FSDP ZeRO-2分片）。

---

## 详细分析

### 1. 当前配置状态（从日志提取）

从 `logs/qwen3_deepresearch_tis_test1-debug-fused-liger_20260104_134657.log` 第131行可以看到：

```yaml
'ref': {
    'fsdp_config': {
        'param_offload': False,        # ❌ 未开启参数offload
        'optimizer_offload': False,     # ❌ 未开启优化器offload  
        'offload_policy': False,        # ❌ 未开启offload策略
        'use_zero2': True,              # ✅ 使用ZeRO-2 (SHARD_GRAD_OP)
        'strategy': 'fsdp',
        ...
    }
}
```

**结论**：你的ref model使用FSDP ZeRO-2策略，参数分片在8张GPU上，**不会offload到CPU**。

---

### 2. 训练各阶段的内存管理

PPO训练包含多个阶段，每个阶段使用不同的模型：

```
┌──────────────┬──────────────────┬─────────────────────────────────┐
│   阶段       │   使用的模型     │   Ref Model 位置                 │
├──────────────┼──────────────────┼─────────────────────────────────┤
│ 1. Rollout   │ Rollout Engine   │ GPU (FSDP分片，待命状态)        │
│              │  (SGLang/vLLM)   │                                 │
├──────────────┼──────────────────┼─────────────────────────────────┤
│ 2. Reward    │ Reward Model/Fn  │ GPU (FSDP分片，待命状态)        │
├──────────────┼──────────────────┼─────────────────────────────────┤
│ 3. Old       │ Actor FSDP       │ GPU (FSDP分片，待命状态)        │
│ Log Prob     │                  │                                 │
├──────────────┼──────────────────┼─────────────────────────────────┤
│ 4. Ref       │ Ref FSDP         │ 👉 GPU (FSDP分片，开始计算)     │
│ Log Prob     │                  │    使用ZeRO-2，参数未offload    │
├──────────────┼──────────────────┼─────────────────────────────────┤
│ 5. Advantage │ CPU计算          │ GPU (FSDP分片，等待下一batch)   │
│ Calculation  │                  │                                 │
├──────────────┼──────────────────┼─────────────────────────────────┤
│ 6. Actor     │ Actor FSDP       │ GPU (FSDP分片，待命状态)        │
│ Update       │                  │                                 │
└──────────────┴──────────────────┴─────────────────────────────────┘
```

---

### 3. Ref Model 的内存管理机制

#### 3.1 核心代码流程

**文件位置**：`verl/workers/fsdp_workers.py:1012-1044`

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
@DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
def compute_ref_log_prob(self, data: DataProto):
    # 步骤1: 如果开启了param_offload，从CPU加载参数到GPU
    # （你的配置中没有开启，所以这段不执行）
    if self._is_offload_param:  # False in your config
        load_fsdp_model_to_gpu(self.ref_module_fsdp)  # 不执行
    
    # 步骤2: 使用Ref FSDP计算log_prob
    with self.ulysses_sharding_manager:
        data = data.to("cpu")  # 输入数据先移到CPU
        output, _ = self.ref_policy.compute_log_prob(
            data=data, 
            calculate_entropy=False
        )  # Ref模型forward计算
        output = DataProto.from_dict(tensors={"ref_log_prob": output})
    
    # 步骤3: 输出结果移到CPU
    output = output.to("cpu")
    
    # 步骤4: FSDP reshard（释放unsharded参数）
    if self.world_size > 1:
        if fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)
    
    # 步骤5: 如果开启了param_offload，offload参数到CPU
    # （你的配置中没有开启，所以这段不执行）
    if self._is_offload_param:  # False in your config
        offload_fsdp_model_to_cpu(self.ref_module_fsdp)  # 不执行
    
    return output
```

#### 3.2 关键点说明

1. **输入数据流**：
   - 输入`data`被移到CPU：`data.to("cpu")`
   - 在`compute_log_prob`内部，每个micro batch会再移回GPU进行计算

2. **Ref Model参数位置**：
   - 由于`param_offload=False`，参数始终保持在GPU（FSDP分片状态）
   - 计算时FSDP会自动allgather参数分片→计算→reshard
   - Reshard操作释放allgather的完整参数，但保留本地分片

3. **输出数据流**：
   - 计算结果`output`移到CPU：`output.to("cpu")`
   - 这是为了不占用GPU显存，让后续阶段使用

---

### 4. 内存占用分析

假设Qwen3-4B模型（4.02B参数，每参数2字节BF16）：

#### 4.1 当前配置（param_offload=False, use_zero2=True）

```
┌─────────────────────┬──────────────┬─────────────────────────┐
│      阶段           │   每GPU显存  │   说明                   │
├─────────────────────┼──────────────┼─────────────────────────┤
│ Ref计算前           │   ~1GB       │ ZeRO-2分片 (4GB÷8卡)    │
│                     │              │ 梯度不分片但ref不训练    │
├─────────────────────┼──────────────┼─────────────────────────┤
│ Ref计算中           │   ~1GB       │ FSDP自动allgather        │
│ (forward pass)      │ + 临时激活值  │ 按需加载完整参数层       │
├─────────────────────┼──────────────┼─────────────────────────┤
│ Ref计算后           │   ~1GB       │ Reshard后恢复分片状态    │
│ (reshard完成)       │              │ 参数仍在GPU              │
└─────────────────────┴──────────────┴─────────────────────────┘
```

#### 4.2 如果开启param_offload=True

```
┌─────────────────────┬──────────────┬─────────────────────────┐
│      阶段           │   每GPU显存  │   说明                   │
├─────────────────────┼──────────────┼─────────────────────────┤
│ Ref计算前           │   ~0GB       │ 参数在CPU（pinned memory)│
├─────────────────────┼──────────────┼─────────────────────────┤
│ load_to_gpu         │   ~1GB       │ 从CPU加载分片到GPU       │
├─────────────────────┼──────────────┼─────────────────────────┤
│ Ref计算中           │   ~1GB       │ FSDP计算                 │
│                     │ + 临时激活值  │                          │
├─────────────────────┼──────────────┼─────────────────────────┤
│ offload_to_cpu      │   ~0GB       │ 参数回到CPU              │
└─────────────────────┴──────────────┴─────────────────────────┘
```

---

### 5. 为什么数据要移到CPU？

你可能注意到代码中有：
```python
data = data.to("cpu")  # 为什么输入移到CPU？
```

**原因**：

1. **内存管理策略**：输入batch数据在CPU，避免长时间占用GPU显存
2. **按需加载**：在`compute_log_prob`内部，按micro batch加载数据到GPU
3. **流水线效率**：CPU可以准备下一个batch，而GPU计算当前batch

**实际流程**：
```python
# verl/workers/actor/dp_actor.py:compute_log_prob
def compute_log_prob(self, data):
    # 数据在CPU
    for micro_batch in split_micro_batches(data):
        # 每个micro batch移到GPU
        micro_batch = micro_batch.to(device)  
        # GPU计算
        output = model(micro_batch)
        # 输出累积在CPU
```

---

### 6. 如何开启CPU Offload？

如果你想节省GPU显存，可以开启ref model的CPU offload：

#### 6.1 修改配置文件

在你的训练脚本配置中（`qwen3_agentloop.sh`或对应的yaml）添加：

```bash
actor_rollout_ref.ref.fsdp_config.param_offload=True
```

#### 6.2 预期效果

**优点**：
- ✅ 节省GPU显存：ref计算时释放~1GB显存（8卡配置）
- ✅ 其他阶段（rollout/actor update）有更多显存可用

**缺点**：
- ❌ 增加计算时间：
  - CPU→GPU拷贝：~100-200ms（1GB数据，PCIe 3.0）
  - GPU→CPU拷贝：~100-200ms
  - 每个batch增加约200-400ms延迟

#### 6.3 适用场景

| 场景 | 是否推荐offload | 原因 |
|------|----------------|------|
| 小模型(<7B) | ❌ 不推荐 | 显存充足，offload纯属浪费时间 |
| 大模型(7B-13B) | ⚠️ 可选 | 如果OOM可以尝试 |
| 超大模型(>13B) | ✅ 推荐 | 显存紧张，必须offload |
| 你的场景(4B) | ❌ 不推荐 | 显存充足，不需要offload |

---

### 7. 其他内存优化选项

除了`param_offload`，还有其他方式管理ref model内存：

#### 7.1 使用LoRA（推荐）

```bash
actor_rollout_ref.model.lora_rank=64  # 使用LoRA，actor base作为ref
```

**优点**：
- ✅ 完全消除ref model显存占用
- ✅ actor的base parameters作为ref（免费）
- ✅ 性能损失可忽略

#### 7.2 FSDP Offload Policy（仅FSDP2）

```bash
actor_rollout_ref.ref.fsdp_config.offload_policy=True  # 仅FSDP2支持
```

**优点**：
- ✅ 更细粒度的offload控制
- ✅ 可以offload部分参数/梯度/优化器状态

---

### 8. 总结

#### 当前状态：

```
┌──────────────────────────────────────────────────────┐
│  你的配置：param_offload=False, use_zero2=True      │
│                                                      │
│  Ref Model 位置：                                    │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  │  一直在 GPU 上（FSDP ZeRO-2 分片）              │  │
│  │  • 计算前：GPU（分片，~1GB/卡）                 │  │
│  │  • 计算中：GPU（allgather, forward, reshard）  │  │
│  │  • 计算后：GPU（分片，~1GB/卡）                 │  │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
└──────────────────────────────────────────────────────┘
```

#### 关键代码位置：

| 功能 | 文件位置 | 行号 |
|------|---------|------|
| Ref计算入口 | `verl/workers/fsdp_workers.py` | 1012-1044 |
| CPU offload函数 | `verl/utils/fsdp_utils.py` | 144-167 |
| CPU load函数 | `verl/utils/fsdp_utils.py` | 未显示（在后续行） |
| Offload配置读取 | `verl/workers/fsdp_workers.py` | 248-250 |
| 训练主循环 | `verl/trainer/ppo/ray_trainer.py` | 1159-1166 |

#### 建议：

对于你的4B模型：
1. **保持当前配置**（param_offload=False）- 性能最优
2. 如果遇到OOM，优先尝试：
   - 减少`ppo_micro_batch_size_per_gpu`
   - 减少`max_model_len`
   - 减少`n`（rollout采样数）
3. 最后才考虑开启`param_offload=True`

---

## 参考资料

- FSDP官方文档: https://pytorch.org/docs/stable/fsdp.html
- Verl架构文档: `docs/verl_rollout_training_architecture.md`
- 配置示例: `tests/trainer/config/legacy_ppo_trainer.yaml:341-345`

