# 多节点训练快速启动指南

## 快速步骤

### 1. 在机器1（Head Node）启动 Ray

```bash
cd /share/project/wanli/Search_Agent/verl
conda activate /share/project/wanli/env/verl-v060

# 获取本机 IP（内网 IP）
HEAD_IP=$(hostname -I | awk '{print $1}')
echo "Head Node IP: $HEAD_IP"

# 停止可能存在的 Ray
ray stop

# 启动 Ray Head
ray start --head --dashboard-host=0.0.0.0 --port=6379 --node-ip-address=$HEAD_IP

# 查看状态，记录 GCS 地址（格式：IP:6379）
ray status
```

**记录下显示的 GCS 地址，例如：`10.0.0.1:6379`**

### 2. 在机器2（Worker Node）连接 Ray

```bash
cd /share/project/wanli/Search_Agent/verl
conda activate /share/project/wanli/env/verl-v060

# 将 <HEAD_IP>:6379 替换为步骤1中记录的地址
ray stop
ray start --address=<HEAD_IP>:6379

# 验证连接
ray status
```

### 3. 在机器1运行训练脚本

```bash
cd /share/project/wanli/Search_Agent/verl
conda activate /share/project/wanli/env/verl-v060

# 方式1：直接运行（推荐，如果 Ray 已连接）
bash examples/sglang_multiturn/search_browser/qwen3_agentloop_packing_resume_multinode.sh

# 方式2：使用 ray job submit（如果需要）
# 先设置 RAY_ADDRESS
# export RAY_ADDRESS="http://<HEAD_IP>:8265"
# bash examples/sglang_multiturn/search_browser/qwen3_agentloop_packing_resume_multinode.sh
```

## 验证清单

- [ ] 两台机器可以互相 ping 通
- [ ] 防火墙已开放端口 6379 和 8265
- [ ] 两台机器的数据路径一致
- [ ] `ray status` 显示两台机器都在集群中
- [ ] 训练脚本中 `trainer.nnodes=2`

## 常见问题

### Q: 如何查看 Ray Dashboard？
A: 在浏览器访问 `http://<HEAD_NODE_IP>:8265`

### Q: Worker Node 连接失败？
A: 检查：
1. Head Node 的 IP 是否正确
2. 防火墙是否开放端口
3. 两台机器是否在同一网络

### Q: 如何停止 Ray 集群？
A: 在每台机器上运行 `ray stop`

### Q: 如何查看训练日志？
A: 日志保存在 `./logs_packing_resume_multinode/` 目录

## 性能提示

- 总 GPU 数 = 8 GPUs/node × 2 nodes = 16 GPUs
- 如果使用共享存储，可以避免数据复制
- 监控 Ray Dashboard 查看资源使用情况
