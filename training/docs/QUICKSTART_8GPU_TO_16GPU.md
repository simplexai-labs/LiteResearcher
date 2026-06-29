# 🚀 从单机8卡checkpoint恢复到2机16卡训练 - 快速指南

## 核心问题

**单机8卡checkpoint无法直接用于2机16卡训练！**

原因：FSDP checkpoint按GPU数量分片
- 8卡训练 → 8个分片（`model_world_size_8_rank_0-7.pt`）
- 16卡训练 → 需要16个分片（`model_world_size_16_rank_0-15.pt`）

## 解决方案：3步走

### ✅ 步骤1：转换Checkpoint（约5-10分钟）

```bash
cd /share/project/wanli/Search_Agent/verl

# 转换你的checkpoint
bash scripts/convert_checkpoint_8to16.sh \
    checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12
```

**输出**：会自动创建 `global_step_12_16gpu` 目录

### ✅ 步骤2：启动Ray集群

**机器1（Head Node）**：
```bash
cd /share/project/wanli/Search_Agent/verl
bash examples/sglang_multiturn/search_browser/ray_setup_head.sh
# 记录输出的IP，例如: 10.0.0.1
```

**机器2（Worker Node）**：
```bash
cd /share/project/wanli/Search_Agent/verl
bash examples/sglang_multiturn/search_browser/ray_setup_worker.sh 10.0.0.1
```

### ✅ 步骤3：修改训练脚本并启动

编辑 `qwen3_agentloop_resume.sh`：

```bash
# 修改checkpoint路径（注意添加 _16gpu 后缀）
CHECKPOINT_PATH="/share/project/wanli/Search_Agent/verl/checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12_16gpu"

# 修改节点数
trainer.nnodes=2 \
trainer.n_gpus_per_node=8
```

启动训练：
```bash
bash examples/sglang_multiturn/search_browser/qwen3_agentloop_resume.sh
```

## 详细说明

### 为什么需要转换？

FSDP（Fully Sharded Data Parallel）将模型参数分片到每个GPU上：

**单机8卡**：
```
global_step_12/actor/
├── model_world_size_8_rank_0.pt  (GPU 0的分片)
├── model_world_size_8_rank_1.pt  (GPU 1的分片)
├── ...
└── model_world_size_8_rank_7.pt  (GPU 7的分片)
```

**2机16卡**：
```
global_step_12_16gpu/actor/
├── model_world_size_16_rank_0.pt   (机器1 GPU 0)
├── model_world_size_16_rank_1.pt   (机器1 GPU 1)
├── ...
├── model_world_size_16_rank_7.pt   (机器1 GPU 7)
├── model_world_size_16_rank_8.pt   (机器2 GPU 0)
├── ...
└── model_world_size_16_rank_15.pt  (机器2 GPU 7)
```

### 转换做了什么？

1. **加载** 8个分片的模型参数
2. **合并** 成完整的模型状态
3. **重新分片** 成16份
4. **保存** 新的checkpoint

### 转换三个可用的Checkpoint

您有三个 global_step_12，建议选择最佳的进行转换：

```bash
# 选项1: on_policy
bash scripts/convert_checkpoint_8to16.sh \
    checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_on_policy/global_step_12

# 选项2: nokl (无KL loss)
bash scripts/convert_checkpoint_8to16.sh \
    checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12

# 选项3: on_policy_nokl
bash scripts/convert_checkpoint_8to16.sh \
    checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_on_policy_nokl/global_step_12
```

### 如果两台机器不共享存储

转换后需要同步checkpoint到第二台机器：

```bash
# 使用rsync（推荐，支持断点续传）
rsync -avz --progress \
    checkpoints/.../global_step_12_16gpu \
    user@machine2:/share/project/wanli/Search_Agent/verl/checkpoints/.../

# 或使用scp
scp -r checkpoints/.../global_step_12_16gpu \
    user@machine2:/share/project/wanli/Search_Agent/verl/checkpoints/.../
```

## 验证清单

转换后，验证以下内容：

- [ ] 转换后的目录存在：`global_step_12_16gpu`
- [ ] Actor分片文件数量正确：16个 `model_world_size_16_rank_*.pt`
- [ ] 如果有Critic，分片也是16个
- [ ] 两台机器的checkpoint路径一致
- [ ] Ray集群状态正常（`ray status` 显示2个节点）
- [ ] 训练脚本已更新checkpoint路径和nnodes=2

## 监控和调试

### 查看转换进度
```bash
# 转换脚本会实时输出进度
```

### 验证转换结果
```bash
# 检查文件数量
ls checkpoints/.../global_step_12_16gpu/actor/model_world_size_16_rank_*.pt | wc -l
# 应该输出: 16

# 检查大小
du -sh checkpoints/.../global_step_12_16gpu
```

### 训练启动后检查
```bash
# 查看日志
tail -f logs_packing_resume/*.log

# 检查Ray集群
ray status

# 查看GPU使用情况（在两台机器上都执行）
nvidia-smi
```

## 常见错误和解决

### 错误1：FileNotFoundError: model_world_size_16_rank_X.pt

**原因**：使用了8卡checkpoint但未转换

**解决**：按步骤1转换checkpoint

### 错误2：转换脚本报错 "不是8-GPU格式"

**原因**：源checkpoint可能已损坏或格式不对

**解决**：
1. 检查源checkpoint完整性
2. 确认有8个 `model_world_size_8_rank_*.pt` 文件
3. 尝试其他checkpoint

### 错误3：磁盘空间不足

**原因**：转换需要额外空间（约1.2倍原checkpoint大小）

**解决**：
1. 清理不需要的文件
2. 或指定其他磁盘路径：
   ```bash
   bash scripts/convert_checkpoint_8to16.sh \
       /path/to/source \
       /other/disk/target
   ```

### 错误4：两台机器checkpoint不一致

**原因**：第二台机器没有转换后的checkpoint

**解决**：使用rsync同步

## 性能对比

| 指标 | 单机8卡 | 2机16卡 | 提升 |
|------|---------|---------|------|
| GPU数量 | 8 | 16 | 2x |
| 训练吞吐 | 基准 | ~1.8-2.0x | 1.8-2.0x |
| 每步时间 | 基准 | ~0.5-0.55x | 更快 |
| Worker并发 | 16 | 32 | 2x |

## 完整命令总结

```bash
# ========== 准备工作 ==========
cd /share/project/wanli/Search_Agent/verl
conda activate /share/project/wanli/env/verl-v060

# ========== 机器1：转换 + 启动 ==========
# 1. 转换checkpoint
bash scripts/convert_checkpoint_8to16.sh \
    checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12

# 2. 启动Ray Head
bash examples/sglang_multiturn/search_browser/ray_setup_head.sh

# ========== 机器2：连接 ==========
# 3. 连接Ray Worker
bash examples/sglang_multiturn/search_browser/ray_setup_worker.sh <HEAD_IP>

# ========== 机器1：训练 ==========
# 4. 编辑训练脚本（修改CHECKPOINT_PATH和nnodes）
# 5. 启动训练
bash examples/sglang_multiturn/search_browser/qwen3_agentloop_resume.sh
```

## 下一步

训练完成后，新的checkpoint将保存为16-GPU格式，可以直接用于后续的2机16卡训练，无需再次转换。

如果需要回到单机8卡，需要重新转换（16 → 8）。

## 相关文档

- **详细转换指南**：`docs/CHECKPOINT_CONVERSION_GUIDE.md`
- **多节点配置**：`examples/sglang_multiturn/search_browser/MULTINODE_SETUP.md`
- **转换脚本源码**：`verl/utils/checkpoint/convert_fsdp_checkpoint.py`
