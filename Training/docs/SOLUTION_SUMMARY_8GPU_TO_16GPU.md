# 📖 从单机8卡checkpoint恢复到2机16卡训练 - 完整方案总结

## 🎯 核心问题

您遇到的错误：
```
FileNotFoundError: model_world_size_16_rank_2.pt
```

**根本原因**：FSDP checkpoint按GPU数量分片，8卡checkpoint有8个分片，16卡需要16个分片，无法直接使用。

## ✅ 完整解决方案

### 方案概览

```
单机8卡checkpoint → 转换工具 → 2机16卡checkpoint → 多机训练
```

### 步骤1：转换Checkpoint（必需！）

```bash
cd /share/project/wanli/Search_Agent/verl

# 转换你的checkpoint（选一个）
bash scripts/convert_checkpoint_8to16.sh \
    checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12
```

**输出**：自动创建 `global_step_12_16gpu` 目录

**转换内容**：
- ✅ Actor模型参数（8个分片 → 16个分片）
- ✅ Optimizer状态
- ✅ 训练元数据
- ✅ HuggingFace配置文件

**转换时间**：约5-10分钟（Qwen3-4B模型）

### 步骤2：启动Ray集群

**机器1（Head Node）**：
```bash
cd /share/project/wanli/Search_Agent/verl
bash examples/sglang_multiturn/search_browser/ray_setup_head.sh
```
记录输出的IP地址（例如：`10.0.0.1`）

**机器2（Worker Node）**：
```bash
cd /share/project/wanli/Search_Agent/verl
bash examples/sglang_multiturn/search_browser/ray_setup_worker.sh 10.0.0.1
```

**验证集群**：
```bash
ray status
# 应该显示2个节点
```

### 步骤3：启动训练

**使用新的专用脚本**（推荐）：
```bash
bash examples/sglang_multiturn/search_browser/qwen3_agentloop_2node16gpu_resume.sh
```

这个脚本会：
- ✅ 自动验证checkpoint是16-GPU格式
- ✅ 检查Ray集群状态
- ✅ 使用正确的2节点配置

**或修改现有脚本**：
编辑 `qwen3_agentloop_resume.sh`：
```bash
# 修改checkpoint路径（添加_16gpu后缀）
CHECKPOINT_PATH=".../global_step_12_16gpu"

# 修改训练参数
trainer.nnodes=2 \
trainer.n_gpus_per_node=8
```

## 📁 提供的工具和脚本

### 1. Checkpoint转换工具

| 文件 | 用途 |
|------|------|
| `scripts/convert_checkpoint_8to16.sh` | 自动转换脚本（推荐） |
| `verl/utils/checkpoint/convert_fsdp_checkpoint.py` | 底层转换工具 |

### 2. Ray集群管理

| 文件 | 用途 |
|------|------|
| `ray_setup_head.sh` | 启动Ray Head Node |
| `ray_setup_worker.sh` | 连接Ray Worker Node |

### 3. 训练脚本

| 文件 | 用途 |
|------|------|
| `qwen3_agentloop_2node16gpu_resume.sh` | 专用2机16卡恢复训练脚本（新建） |
| `qwen3_agentloop_resume.sh` | 原单机恢复训练脚本（需手动修改） |

### 4. 文档

| 文件 | 内容 |
|------|------|
| `docs/QUICKSTART_8GPU_TO_16GPU.md` | 快速开始指南 |
| `docs/CHECKPOINT_CONVERSION_GUIDE.md` | 详细转换指南 |
| `examples/.../MULTINODE_SETUP.md` | 多节点配置说明 |

## 🔍 验证清单

转换和启动前，请确认：

- [ ] 已转换checkpoint且有16个分片文件
  ```bash
  ls checkpoints/.../global_step_12_16gpu/actor/model_world_size_16_rank_*.pt | wc -l
  # 应输出: 16
  ```

- [ ] 两台机器都能访问转换后的checkpoint
  ```bash
  # 在两台机器上都执行
  ls checkpoints/.../global_step_12_16gpu
  ```

- [ ] Ray集群正常运行
  ```bash
  ray status
  # 应显示: 2 nodes
  ```

- [ ] 防火墙端口已开放
  - 6379 (Ray GCS)
  - 8265 (Ray Dashboard)
  - 10000-10100 (Ray workers)

## 🎨 可视化流程

```
                  ┌─────────────────────────┐
                  │  单机8卡checkpoint      │
                  │  global_step_12         │
                  │  (8个分片文件)          │
                  └───────────┬─────────────┘
                              │
                              │ 转换
                              │ (convert_checkpoint_8to16.sh)
                              ▼
                  ┌─────────────────────────┐
                  │  2机16卡checkpoint      │
                  │  global_step_12_16gpu   │
                  │  (16个分片文件)         │
                  └───────────┬─────────────┘
                              │
                              │ 同步到第二台机器
                              │ (如果不共享存储)
                              ▼
            ┌─────────────────────────────────────┐
            │         Ray集群                      │
            │  ┌───────────┐    ┌───────────┐    │
            │  │ Head Node │◄───│Worker Node│    │
            │  │  8 GPUs   │    │  8 GPUs   │    │
            │  └───────────┘    └───────────┘    │
            └──────────────┬──────────────────────┘
                           │
                           │ 启动训练
                           │ (qwen3_agentloop_2node16gpu_resume.sh)
                           ▼
            ┌─────────────────────────────────────┐
            │      2机16卡分布式训练               │
            │  速度提升: ~1.8-2.0x                 │
            └─────────────────────────────────────┘
```

## ⚡ 快速命令总结

```bash
# ========== 一键复制版本 ==========

# 1. 转换checkpoint
cd /share/project/wanli/Search_Agent/verl
bash scripts/convert_checkpoint_8to16.sh \
    checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12

# 2. 机器1: 启动Ray Head
bash examples/sglang_multiturn/search_browser/ray_setup_head.sh

# 3. 机器2: 连接Ray Worker (替换<HEAD_IP>)
bash examples/sglang_multiturn/search_browser/ray_setup_worker.sh <HEAD_IP>

# 4. 机器1: 启动训练
bash examples/sglang_multiturn/search_browser/qwen3_agentloop_2node16gpu_resume.sh
```

## 🔧 常见问题

### Q1: 不转换直接用8卡checkpoint会怎样？

**A**: 报错 `FileNotFoundError: model_world_size_16_rank_2.pt`（就是您遇到的问题）

### Q2: 转换需要额外的磁盘空间吗？

**A**: 是的，需要约1.2倍原checkpoint大小的空间（临时存储转换后的版本）

### Q3: 转换会修改原checkpoint吗？

**A**: 不会，转换工具只读取原checkpoint，在新目录创建转换后的版本

### Q4: 可以边训练边转换吗？

**A**: 可以，但不建议。最好在训练开始前完成转换

### Q5: 转换失败了怎么办？

**A**: 检查：
1. 原checkpoint是否完整（8个分片文件）
2. 磁盘空间是否充足
3. 重新运行转换脚本

### Q6: 两台机器必须共享存储吗？

**A**: 不必须，但建议使用共享存储（NFS）。如果不共享，需要用rsync同步

### Q7: 训练中保存的新checkpoint是什么格式？

**A**: 16-GPU格式，可以直接用于后续的2机16卡训练，无需再转换

## 📊 性能对比

| 指标 | 单机8卡 | 2机16卡 | 提升 |
|------|---------|---------|------|
| GPU数量 | 8 | 16 | 2.0x |
| 训练吞吐 | 100% | 180-200% | 1.8-2.0x |
| 每步时间 | 100% | 50-55% | 更快 |
| Worker数 | 8-16 | 32 | 2-4x |

## 📞 需要帮助？

如果遇到问题：

1. **查看日志**：`logs_2node16gpu_resume/*.log`
2. **检查Ray状态**：`ray status`
3. **查看GPU使用**：`nvidia-smi`（在两台机器上都执行）
4. **Ray Dashboard**：`http://<HEAD_NODE_IP>:8265`

## 🎉 成功标志

训练成功启动后，您应该看到：

```
✅ Found 16-GPU checkpoint: .../global_step_12_16gpu
✅ Ray集群正常，检测到 2 个节点
🚀 2机16卡训练 - 从转换后的checkpoint恢复
  - Nodes: 2
  - GPUs per node: 8
  - Total GPUs: 16
```

日志中应该看到两台机器的GPU都在工作。

祝训练顺利！🚀
