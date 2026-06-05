# 单机8卡 → 2机16卡 Checkpoint转换指南

## 问题背景

单机8卡训练保存的FSDP checkpoint包含8个分片文件（`model_world_size_8_rank_0.pt` 到 `model_world_size_8_rank_7.pt`），但2机16卡训练需要16个分片文件（`model_world_size_16_rank_0.pt` 到 `model_world_size_16_rank_15.pt`）。

直接使用8卡checkpoint会导致错误：
```
FileNotFoundError: model_world_size_16_rank_2.pt
```

## 解决方案

使用checkpoint转换工具将8卡checkpoint转换为16卡格式。

## 方法1：使用自动转换脚本（推荐）

### 步骤1：转换Checkpoint

```bash
cd /share/project/wanli/Search_Agent/verl

# 转换 global_step_12
bash scripts/convert_checkpoint_8to16.sh \
    /share/project/wanli/Search_Agent/verl/checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12
```

这会自动创建一个新的checkpoint目录：
```
/share/project/wanli/Search_Agent/verl/checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12_16gpu
```

### 步骤2：同步到第二台机器

如果两台机器不共享存储，需要将转换后的checkpoint复制到第二台机器：

```bash
# 方法1: 使用rsync（推荐）
rsync -avz --progress \
    /path/to/global_step_12_16gpu \
    user@worker_node:/path/to/global_step_12_16gpu

# 方法2: 使用scp
scp -r /path/to/global_step_12_16gpu \
    user@worker_node:/path/to/
```

### 步骤3：修改训练脚本

编辑 `qwen3_agentloop_resume.sh` 或创建新的训练脚本：

```bash
# 使用转换后的checkpoint
CHECKPOINT_PATH="/share/project/wanli/Search_Agent/verl/checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12_16gpu"

# 设置为2节点
trainer.nnodes=2 \
trainer.n_gpus_per_node=8
```

### 步骤4：启动训练

```bash
# 1. 在机器1启动Ray Head
bash examples/sglang_multiturn/search_browser/ray_setup_head.sh

# 2. 在机器2连接Ray Worker
bash examples/sglang_multiturn/search_browser/ray_setup_worker.sh <HEAD_NODE_IP>

# 3. 在机器1启动训练
bash examples/sglang_multiturn/search_browser/qwen3_agentloop_resume.sh
```

## 方法2：手动使用Python脚本

如果需要更细粒度的控制：

```bash
# 转换Actor checkpoint
python3 verl/utils/checkpoint/convert_fsdp_checkpoint.py \
    --source_ckpt_dir /path/to/global_step_12 \
    --target_ckpt_dir /path/to/global_step_12_16gpu \
    --source_world_size 8 \
    --target_world_size 16 \
    --component actor

# 转换Critic checkpoint（如果有）
python3 verl/utils/checkpoint/convert_fsdp_checkpoint.py \
    --source_ckpt_dir /path/to/global_step_12 \
    --target_ckpt_dir /path/to/global_step_12_16gpu \
    --source_world_size 8 \
    --target_world_size 16 \
    --component critic
```

## 转换的三个Checkpoint

您有三个可用的 global_step_12 checkpoint，建议都转换：

```bash
# 转换checkpoint 1: on_policy
bash scripts/convert_checkpoint_8to16.sh \
    /share/project/wanli/Search_Agent/verl/checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_on_policy/global_step_12

# 转换checkpoint 2: nokl
bash scripts/convert_checkpoint_8to16.sh \
    /share/project/wanli/Search_Agent/verl/checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12

# 转换checkpoint 3: on_policy_nokl
bash scripts/convert_checkpoint_8to16.sh \
    /share/project/wanli/Search_Agent/verl/checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_on_policy_nokl/global_step_12
```

## 验证转换结果

### 检查文件数量

```bash
# 源checkpoint（应该有8个）
ls /path/to/global_step_12/actor/model_world_size_8_rank_*.pt | wc -l
# 输出: 8

# 转换后的checkpoint（应该有16个）
ls /path/to/global_step_12_16gpu/actor/model_world_size_16_rank_*.pt | wc -l
# 输出: 16
```

### 检查文件大小

```bash
du -sh /path/to/global_step_12
du -sh /path/to/global_step_12_16gpu
```

转换后的checkpoint大小应该与原checkpoint相近（可能略大，因为有更多的metadata）。

## 常见问题

### Q1: 转换需要多长时间？

取决于checkpoint大小，通常：
- 4B模型：5-10分钟
- 7B模型：10-20分钟

### Q2: 转换需要多少磁盘空间？

需要额外的空间存储转换后的checkpoint，大约是原checkpoint的1.1-1.2倍。

### Q3: 可以在不停止训练的情况下转换吗？

可以。转换工具只读取源checkpoint，不会修改它。

### Q4: 如果转换失败怎么办？

1. 检查源checkpoint是否完整
2. 确保有足够的磁盘空间
3. 查看错误日志
4. 重新运行转换脚本

### Q5: 转换后的checkpoint可以用于单机吗？

不建议。转换后的checkpoint是为16-GPU优化的。如果要回到8-GPU：

```bash
# 反向转换（16 → 8）
bash scripts/convert_checkpoint_8to16.sh \
    /path/to/global_step_12_16gpu \
    /path/to/global_step_12_back_to_8gpu
```

但需要修改脚本中的world_size参数。

### Q6: 可以转换为其他GPU数量吗？

可以，修改脚本中的参数：
- 8 → 4: `--source_world_size 8 --target_world_size 4`
- 8 → 32: `--source_world_size 8 --target_world_size 32`

## 完整工作流程示例

```bash
# ========== 在机器1上 ==========

# 1. 转换checkpoint
cd /share/project/wanli/Search_Agent/verl
bash scripts/convert_checkpoint_8to16.sh \
    checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12

# 2. 如果不共享存储，同步到机器2
rsync -avz --progress \
    checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12_16gpu \
    user@machine2:/share/project/wanli/Search_Agent/verl/checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/

# 3. 启动Ray Head
bash examples/sglang_multiturn/search_browser/ray_setup_head.sh
# 记录输出的IP地址，例如: 10.0.0.1

# ========== 在机器2上 ==========

# 4. 连接到Ray集群
cd /share/project/wanli/Search_Agent/verl
bash examples/sglang_multiturn/search_browser/ray_setup_worker.sh 10.0.0.1

# ========== 回到机器1上 ==========

# 5. 修改训练脚本中的checkpoint路径
# 编辑 qwen3_agentloop_resume.sh，修改:
#   CHECKPOINT_PATH="...../global_step_12_16gpu"
#   trainer.nnodes=2

# 6. 启动训练
bash examples/sglang_multiturn/search_browser/qwen3_agentloop_resume.sh

# 7. 监控训练
tail -f logs_packing_resume/*.log
```

## 重要提示

1. **备份原checkpoint**：转换前建议备份原checkpoint
2. **验证转换**：转换后验证文件数量和大小
3. **路径一致性**：确保两台机器的checkpoint路径完全一致
4. **测试先行**：建议先用小checkpoint测试转换流程

## 相关文档

- `verl/utils/checkpoint/convert_fsdp_checkpoint.py` - 转换脚本源码
- `scripts/convert_checkpoint_8to16.sh` - 自动转换脚本
- `examples/sglang_multiturn/search_browser/MULTINODE_SETUP.md` - 多节点配置
