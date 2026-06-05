# 跨 GPU 数量 (world_size) Resume 指南

本文档介绍如何将 VERL FSDP checkpoint 从一个 GPU 数量（如 8 卡）转换到另一个 GPU 数量（如 16 卡），并恢复训练（包括模型权重、优化器状态、LR scheduler）。

## 背景

VERL 使用 FSDP (Fully Sharded Data Parallel) 保存的 checkpoint 是按 `world_size` 分片的：

```
actor/
├── model_world_size_8_rank_0.pt    # 模型权重分片（DTensor 格式）
├── model_world_size_8_rank_1.pt
├── ...
├── optim_world_size_8_rank_0.pt    # 优化器状态分片（flat tensor）
├── optim_world_size_8_rank_1.pt
├── ...
├── extra_state_world_size_8_rank_0.pt  # LR scheduler + RNG 状态
├── ...
└── fsdp_config.json
```

**直接在不同 `world_size` 上加载会报错**，因为文件名中硬编码了 `world_size`。

## 解决方案

### 步骤 1：转换 Checkpoint

使用 `scripts/convert_checkpoint_worldsize.py` 脚本进行转换：

```bash
cd /share/project/wanli/Search_Agent/verl
conda activate verl-v060

# 转换 actor checkpoint（从 8 卡 → 16 卡）
python scripts/convert_checkpoint_worldsize.py \
    --ckpt_dir /path/to/global_step_25/actor \
    --target_world_size 16
```

**参数说明：**
- `--ckpt_dir`：checkpoint 的 actor 目录路径，或 `global_step_XX` 目录
- `--target_world_size`：目标 GPU 数量
- `--components`：要转换的组件（默认 `actor`，可选 `actor critic`）

**转换脚本做了什么：**

| 组件 | 转换方式 | 精度 |
|------|---------|------|
| Model | 合并 N 个 DTensor 分片 → `model_full.pt`（完整参数） | bit-exact |
| Optimizer (exp_avg, exp_avg_sq) | 合并 N 个 flat shard → 按 `torch.chunk(target_ws)` 重切 M 份 | bit-exact |
| Optimizer (step) | 标量，直接复制 | exact |
| LR Scheduler | 直接从 rank 0 复制（各 rank 完全相同） | exact |
| RNG | 从 rank 0 复制（各 rank 独立，无法拆分合并） | rank 0 状态 |

> **注意：** 转换过程完全是 `torch.cat` + 切片操作，不涉及任何浮点运算，**零精度损失**。

**转换后生成的文件：**

```
actor/
├── model_full.pt                          # 新增：合并后的完整模型
├── optim_world_size_16_rank_0.pt          # 新增：resharded 优化器分片
├── optim_world_size_16_rank_1.pt
├── ...
├── extra_state_world_size_16_rank_0.pt    # 新增：复制的 extra_state
├── ...
├── extra_state_full.pt                    # 新增：extra_state 备份
├── fsdp_config.json                       # 更新：checkpoint_format="full"
├── model_world_size_8_rank_0.pt           # 保留：原始分片（不动）
├── ...
└── optim_world_size_8_rank_0.pt           # 保留：原始分片（不动）
```

### 步骤 2：整理目录（推荐）

建议将转换后的 16 卡文件移到独立目录，保持和原始 checkpoint 格式一致：

```bash
CKPT_BASE="/path/to/checkpoints"
SRC="$CKPT_BASE/experiment_name"
DST="$CKPT_BASE/experiment_name_16gpu"

# 创建目录
mkdir -p "$DST/global_step_25/actor/huggingface"

# 移动 16 卡文件
mv "$SRC/global_step_25/actor/model_full.pt" "$DST/global_step_25/actor/"
mv "$SRC/global_step_25/actor"/optim_world_size_16_rank_*.pt "$DST/global_step_25/actor/"
mv "$SRC/global_step_25/actor"/extra_state_world_size_16_rank_*.pt "$DST/global_step_25/actor/"
mv "$SRC/global_step_25/actor/extra_state_full.pt" "$DST/global_step_25/actor/"

# 复制配置和 tokenizer
cp -r "$SRC/global_step_25/actor/huggingface/"* "$DST/global_step_25/actor/huggingface/"
cp "$SRC/global_step_25/data.pt" "$DST/global_step_25/data.pt"

# 创建 16 卡 fsdp_config.json
cat > "$DST/global_step_25/actor/fsdp_config.json" << EOF
{
    "FSDP_version": 1,
    "world_size": 16,
    "checkpoint_format": "full",
    "original_world_size": 8,
    "target_world_size": 16
}
EOF

# 恢复原始 8 卡目录的 fsdp_config
cat > "$SRC/global_step_25/actor/fsdp_config.json" << EOF
{
    "FSDP_version": 1,
    "world_size": 8,
    "checkpoint_format": "sharded"
}
EOF
```

### 步骤 3：修改训练脚本并启动

在训练脚本中设置以下参数：

```bash
# 1. 修改节点数
trainer.nnodes=2 \          # 8卡→16卡: 1→2

# 2. 添加 resume 配置
trainer.resume_mode=resume_path \
trainer.resume_from_path="/path/to/experiment_name_16gpu/global_step_25" \
```

启动训练：

```bash
ray job submit \
    --address=http://127.0.0.1:8265 \
    --runtime-env=/path/to/verl/verl/trainer/runtime_env.yaml \
    -- bash /path/to/training_script.sh
```

## 加载流程原理

加载时 `FSDPCheckpointManager` 自动检测格式：

```
_detect_checkpoint_format()
  ├── 检查 model_world_size_16_rank_0.pt → 不存在
  ├── 检查 model_full.pt → 存在 ✅
  └── 返回 "full" → 进入 _load_full_checkpoint()
```

`_load_full_checkpoint()` 加载过程：

1. **Model**: 每个 rank 读 `model_full.pt`，在 `FULL_STATE_DICT` context 下 FSDP 自动 reshard
2. **Optimizer**: 每个 rank 读自己的 `optim_world_size_16_rank_{rank}.pt`，在 `SHARDED_STATE_DICT` context 下直接加载
3. **Extra State**: 每个 rank 读自己的 `extra_state_world_size_16_rank_{rank}.pt`，恢复 LR scheduler 和 RNG

## 恢复的内容

| 组件 | 状态 | 说明 |
|------|------|------|
| 模型权重 | ✅ 完整恢复 | FSDP 自动 reshard |
| Optimizer (exp_avg/exp_avg_sq) | ✅ 完整恢复 | Adam 一阶/二阶矩估计 |
| Optimizer (step) | ✅ 恢复 | 如 step=25 |
| LR Scheduler | ✅ 恢复 | last_epoch、_step_count 等 |
| Global Step | ✅ 恢复 | 从路径名解析 |
| DataLoader 位置 | ✅ 恢复 | data.pt |
| RNG 状态 | ⚠️ 部分恢复 | 使用 rank 0 的状态，训练第一步后各 rank 自然分叉 |

## 注意事项

1. **首次 resume 较慢**：16 个 rank 都要读同一个 17GB 的 `model_full.pt`，IO 带宽密集。后续在 16 卡上 resume 自己保存的 checkpoint 会走 sharded 路径，速度正常。

2. **磁盘空间**：`model_full.pt` 约为模型参数大小（如 Qwen3-4B ≈ 17GB），optimizer shard 总量约为原来的 2 倍（exp_avg + exp_avg_sq，都是 float32）。

3. **整除要求**：转换要求 `total_param_size` 能被 `target_world_size` 整除（FSDP 会 padding，脚本已处理）。

4. **FSDP 版本**：当前支持 FSDP1。FSDP2 使用不同的 state dict 机制，可能需要额外适配。

5. **转换脚本只需 CPU**：不需要 GPU，在任何有 PyTorch 的环境即可运行。

## 相关文件

- 转换脚本：`scripts/convert_checkpoint_worldsize.py`
- Checkpoint 加载逻辑：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- 训练 resume 入口：`verl/trainer/ppo/ray_trainer.py` (`_load_checkpoint()`)

## 实际验证记录

已在以下场景验证通过：
- 8 卡 (world_size=8) → 16 卡 (world_size=16)
- Qwen3-4B 模型，FSDP1 + ZERO2 (SHARD_GRAD_OP)
- 模型权重、优化器状态、LR scheduler 全部 bit-exact 恢复
- 训练正常从 global_step_25 继续（step 26 开始 rollout）
