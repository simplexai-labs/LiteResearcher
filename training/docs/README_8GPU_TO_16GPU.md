# 🎯 完整解决方案：单机8卡checkpoint → 2机16卡训练

## 问题

您在尝试从单机8卡checkpoint恢复到2机16卡训练时遇到错误：
```
FileNotFoundError: model_world_size_16_rank_2.pt
```

## 根本原因

FSDP checkpoint按GPU数量分片：
- 8卡训练 → 8个分片文件 (`model_world_size_8_rank_0-7.pt`)
- 16卡训练 → 需要16个分片文件 (`model_world_size_16_rank_0-15.pt`)

**不能直接使用8卡checkpoint进行16卡训练！**

## 解决方案（3个命令）

```bash
# 1. 转换checkpoint
bash scripts/convert_checkpoint_8to16.sh checkpoints/.../global_step_12

# 2. 启动Ray集群
# 机器1:
bash examples/sglang_multiturn/search_browser/ray_setup_head.sh
# 机器2:
bash examples/sglang_multiturn/search_browser/ray_setup_worker.sh <HEAD_IP>

# 3. 启动训练
bash examples/sglang_multiturn/search_browser/qwen3_agentloop_2node16gpu_resume.sh
```

## 📚 文档索引

### 快速开始
- **[QUICKSTART_8GPU_TO_16GPU.md](./QUICKSTART_8GPU_TO_16GPU.md)** ⭐
  - 3步快速指南
  - 适合快速上手

### 详细指南
- **[CHECKPOINT_CONVERSION_GUIDE.md](./CHECKPOINT_CONVERSION_GUIDE.md)**
  - Checkpoint转换详细说明
  - 故障排查
  - 常见问题

- **[SOLUTION_SUMMARY_8GPU_TO_16GPU.md](./SOLUTION_SUMMARY_8GPU_TO_16GPU.md)**
  - 完整解决方案总结
  - 可视化流程
  - 性能对比

- **[../examples/sglang_multiturn/search_browser/MULTINODE_SETUP.md](../examples/sglang_multiturn/search_browser/MULTINODE_SETUP.md)**
  - Ray集群配置
  - 网络设置
  - 故障排查

## 🛠️ 工具和脚本

### Checkpoint转换
```bash
scripts/convert_checkpoint_8to16.sh           # 自动转换脚本
verl/utils/checkpoint/convert_fsdp_checkpoint.py  # 底层工具
```

### Ray集群管理
```bash
examples/sglang_multiturn/search_browser/ray_setup_head.sh      # Head Node
examples/sglang_multiturn/search_browser/ray_setup_worker.sh    # Worker Node
```

### 训练脚本
```bash
examples/sglang_multiturn/search_browser/qwen3_agentloop_2node16gpu_resume.sh  # 2机16卡专用（新）
examples/sglang_multiturn/search_browser/qwen3_agentloop_resume.sh             # 单机恢复（需改）
```

## ✅ 关键步骤

### 步骤1：转换Checkpoint（必需）

```bash
cd /share/project/wanli/Search_Agent/verl

# 转换你的checkpoint（选一个）
bash scripts/convert_checkpoint_8to16.sh \
    checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_12

# 验证转换结果
ls checkpoints/.../global_step_12_16gpu/actor/model_world_size_16_rank_*.pt | wc -l
# 应该输出: 16
```

### 步骤2：Ray集群

**机器1（Head Node）**：
```bash
bash examples/sglang_multiturn/search_browser/ray_setup_head.sh
```

**机器2（Worker Node）**：
```bash
bash examples/sglang_multiturn/search_browser/ray_setup_worker.sh <HEAD_NODE_IP>
```

**验证**：
```bash
ray status
# 应显示2个节点
```

### 步骤3：启动训练

```bash
bash examples/sglang_multiturn/search_browser/qwen3_agentloop_2node16gpu_resume.sh
```

## 📊 转换说明

### 转换做什么？
1. 加载8个原始分片
2. 合并为完整模型
3. 重新分成16个分片
4. 保存新checkpoint

### 转换的内容
- ✅ Actor模型参数
- ✅ Critic模型参数（如果有）
- ✅ Optimizer状态
- ✅ 训练元数据
- ✅ HuggingFace配置

### 转换时间和空间
- **时间**：5-10分钟（Qwen3-4B）
- **空间**：需要额外1.2倍原checkpoint大小

## 🔍 验证清单

- [ ] Checkpoint已转换（16个分片文件）
- [ ] 两台机器都能访问转换后的checkpoint
- [ ] Ray集群状态正常（2个节点）
- [ ] 防火墙端口已开放（6379, 8265, 10000-10100）
- [ ] 两台机器的环境和路径一致

## 💡 重要提示

1. **必须转换**：8卡checkpoint不能直接用于16卡训练
2. **路径后缀**：转换后的checkpoint默认添加 `_16gpu` 后缀
3. **数据同步**：如果不用共享存储，需要用rsync同步checkpoint到第二台机器
4. **新checkpoint**：训练中保存的新checkpoint是16-GPU格式，无需再转换

## 🚀 性能提升

| 指标 | 单机8卡 | 2机16卡 | 提升 |
|------|---------|---------|------|
| GPU数量 | 8 | 16 | 2.0x |
| 训练速度 | 基准 | 1.8-2.0x | 1.8-2.0x |
| Worker数 | 16 | 32 | 2.0x |

## 🐛 常见问题

### FileNotFoundError: model_world_size_16_rank_X.pt
**原因**：使用了未转换的8卡checkpoint
**解决**：运行转换脚本

### 转换失败
**检查**：
- 原checkpoint是否完整（8个文件）
- 磁盘空间是否充足
- 权限是否正确

### Ray集群连接失败
**检查**：
- 两台机器能否互相ping通
- 防火墙端口是否开放
- IP地址是否正确

## 📞 获取帮助

遇到问题时：
1. 查看训练日志：`logs_2node16gpu_resume/*.log`
2. 检查Ray状态：`ray status`
3. 查看GPU：`nvidia-smi`
4. Ray Dashboard：`http://<HEAD_NODE_IP>:8265`

## 📖 相关文档

- [QUICKSTART_8GPU_TO_16GPU.md](./QUICKSTART_8GPU_TO_16GPU.md) - 快速开始
- [CHECKPOINT_CONVERSION_GUIDE.md](./CHECKPOINT_CONVERSION_GUIDE.md) - 转换详细指南
- [SOLUTION_SUMMARY_8GPU_TO_16GPU.md](./SOLUTION_SUMMARY_8GPU_TO_16GPU.md) - 完整方案总结

## ✨ 成功指标

训练成功启动后应看到：
```
✅ Found 16-GPU checkpoint: .../global_step_12_16gpu
✅ Ray集群正常，检测到 2 个节点
🚀 2机16卡训练 - 从转换后的checkpoint恢复
  - Total GPUs: 16
  - 预期速度提升: ~1.8-2.0x
```

祝训练顺利！🎉
