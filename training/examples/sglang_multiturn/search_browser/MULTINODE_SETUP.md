# 多节点训练配置方案

## 概述
本方案适用于两台通过内网连接的机器进行多节点训练。使用 Ray 集群来协调多节点训练。

## 架构说明
- **机器1（Head Node）**：作为 Ray 集群的头节点，负责协调和任务分发
- **机器2（Worker Node）**：作为 Ray 集群的工作节点，连接到 Head Node

## 前置要求
1. 两台机器可以通过内网 IP 互相访问
2. 两台机器上都有相同的代码和数据路径
3. 两台机器上都有相同的 conda 环境
4. 防火墙允许以下端口通信：
   - `6379`：Ray GCS 端口
   - `8265`：Ray Dashboard 端口
   - `10000-10100`：Ray 工作端口（建议开放范围）

## 步骤 1：设置 Ray 集群

### 在机器1（Head Node）上执行：

```bash
# 1. 进入项目目录
cd /share/project/wanli/Search_Agent/verl

# 2. 激活 conda 环境
conda activate /share/project/wanli/env/verl-v060

# 3. 停止可能存在的 Ray 进程
ray stop

# 4. 启动 Ray Head Node
# 注意：将 <HEAD_NODE_IP> 替换为机器1的内网 IP 地址
ray start --head \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --port=6379 \
    --node-ip-address=<HEAD_NODE_IP>

# 5. 查看启动信息，记录 GCS 地址（格式：<IP>:6379）
# 例如：10.0.0.1:6379
ray status
```

**重要**：记录下显示的 GCS 地址，格式类似：`10.0.0.1:6379`

### 在机器2（Worker Node）上执行：

```bash
# 1. 进入项目目录
cd /share/project/wanli/Search_Agent/verl

# 2. 激活 conda 环境
conda activate /share/project/wanli/env/verl-v060

# 3. 停止可能存在的 Ray 进程
ray stop

# 4. 连接到 Head Node
# 将 <HEAD_NODE_IP>:6379 替换为机器1的 GCS 地址
ray start --address=<HEAD_NODE_IP>:6379

# 5. 验证连接
ray status
```

### 验证集群状态

在任意一台机器上运行：
```bash
ray status
```

应该看到两台机器都在集群中，例如：
```
======== Autoscaler status ========
Node status
-------------------------------------------------------
Active:
  1 node_xxx (head node)
  1 node_yyy (worker node)
```

## 步骤 2：配置训练脚本

### 方案 A：使用 ray job submit（推荐）

在 Head Node（机器1）上运行训练脚本，使用 `ray job submit` 提交任务：

```bash
# 使用修改后的脚本（见下方）
bash qwen3_agentloop_packing_resume_multinode.sh
```

### 方案 B：直接运行（如果 Ray 已连接）

如果 Ray 集群已正确连接，可以直接运行训练脚本，verl 会自动检测并使用现有集群。

## 步骤 3：修改训练脚本

主要修改点：
1. 将 `trainer.nnodes=1` 改为 `trainer.nnodes=2`
2. 如果使用 `ray job submit`，需要设置 `RAY_ADDRESS`
3. 确保两台机器的数据路径一致

## 注意事项

### 1. 数据路径一致性
确保两台机器上的以下路径一致：
- 训练数据路径
- 模型路径
- checkpoint 路径
- 日志输出路径（或使用共享存储）

### 2. 网络配置
- 确保两台机器可以互相 ping 通
- 检查防火墙规则
- 如果使用 VPN 或特殊网络，确保 Ray 端口可访问

### 3. 资源分配
- 总 GPU 数 = `n_gpus_per_node × nnodes`
- 例如：`n_gpus_per_node=8, nnodes=2` = 16 张 GPU

### 4. 故障排查
如果遇到连接问题：
```bash
# 检查 Ray 状态
ray status

# 查看 Ray 日志
tail -f /tmp/ray/session_latest/logs/*.log

# 重启 Ray 集群
# 在 Worker Node 上：
ray stop
ray start --address=<HEAD_NODE_IP>:6379

# 在 Head Node 上：
ray stop
ray start --head --dashboard-host=0.0.0.0
```

## 性能优化建议

1. **网络优化**：如果使用 InfiniBand 或高速网络，配置 NCCL 使用高速网络
2. **数据加载**：使用共享存储（NFS/GlusterFS）避免数据复制
3. **监控**：访问 `http://<HEAD_NODE_IP>:8265` 查看 Ray Dashboard
