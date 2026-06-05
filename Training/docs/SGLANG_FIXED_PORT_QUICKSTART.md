# 🎉 SGLang 固定端口配置完成

## ✅ 修改内容

已成功将 SGLang HTTP 服务器配置为使用**固定端口**,从 **30000** 开始。

### 端口分配规则

```
Replica 0 → 端口 30000
Replica 1 → 端口 30001  
Replica 2 → 端口 30002
Replica 3 → 端口 30003
...
Replica N → 端口 3000N
```

## 🚀 快速开始

### 1. 启动训练

```bash
cd /share/project/wanli/Search_Agent/verl
bash examples/sglang_multiturn/search_browser/qwen3_agentloop.sh
```

### 2. 等待服务器启动

训练启动后,等待 1-2 分钟让 SGLang 服务器完全启动。

### 3. 查找 SGLang 服务器

使用自动发现脚本:

```bash
# 默认扫描 30000-30015 (16个端口)
bash scripts/find_sglang_servers.sh

# 如果你有 8 个 GPU
bash scripts/find_sglang_servers.sh 30000 8

# 输出示例:
# 🔍 Searching for SGLang servers...
# ✅ Server #0 found at port 30000
#    📦 Model: checkpoint-533
#    📊 Max Total Tokens: 131072
#    🎯 Max Running Requests: 256
# ✅ Server #1 found at port 30001
#    ...
```

### 4. 监控 SGLang 服务器

```bash
# 监控单个服务器
python scripts/monitor_sglang.py --host localhost --port 30000

# 自动扫描并监控所有服务器 (推荐)
python scripts/monitor_sglang.py --host localhost --port 30000 --scan 8

# 输出示例:
# 🔍 SGLang Server Monitor
# ============================================================
# ✅ Server 0 (localhost:30000)
#    📦 Model: checkpoint-533
#    📈 Metrics:
#       Queue Requests: 12
#       Running Requests: 48
#       KV Cache Usage: 🟡 75.3%
#       Throughput: 1234.5 tokens/s
```

### 5. 手动检查单个服务器

```bash
# 健康检查
curl http://localhost:30000/health_generate

# 获取服务器信息
curl http://localhost:30000/get_server_info | python -m json.tool

# 获取实时 metrics
curl http://localhost:30000/metrics
```

## 📊 端口映射示例

### 场景 1: 8 GPU 单节点 (TP=1, DP=1)

配置:
```bash
trainer.n_gpus_per_node=8
actor_rollout_ref.rollout.tensor_model_parallel_size=1
```

端口映射:
```
GPU 0 → Replica 0 → 端口 30000
GPU 1 → Replica 1 → 端口 30001
GPU 2 → Replica 2 → 端口 30002
GPU 3 → Replica 3 → 端口 30003
GPU 4 → Replica 4 → 端口 30004
GPU 5 → Replica 5 → 端口 30005
GPU 6 → Replica 6 → 端口 30006
GPU 7 → Replica 7 → 端口 30007
```

### 场景 2: 8 GPU 单节点 (TP=2, DP=1)

配置:
```bash
trainer.n_gpus_per_node=8
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

端口映射 (每个 replica 使用 2 个 GPU):
```
GPU 0,1 → Replica 0 → 端口 30000
GPU 2,3 → Replica 1 → 端口 30001
GPU 4,5 → Replica 2 → 端口 30002
GPU 6,7 → Replica 3 → 端口 30003
```

## 🔧 故障排查

### 问题 1: 找不到 SGLang 服务器

**检查步骤**:

1. 确认训练正在运行:
```bash
ps aux | grep main_ppo
```

2. 检查日志中的启动信息:
```bash
grep "SGLangHttpServer attempting to start" logs/*.log
grep "HTTP server started on port" logs/*.log
```

3. 检查是否有错误:
```bash
tail -100 logs/*.log | grep -i "error\|failed"
```

### 问题 2: 端口被占用

如果看到类似错误:
```
Failed to start HTTP server on port 30000
```

**解决方法**:

1. 检查端口占用:
```bash
ss -tlnp | grep 30000
```

2. 端口会自动尝试下一个 (30001, 30002...),查看实际使用的端口:
```bash
bash scripts/find_sglang_servers.sh 30000 16
```

### 问题 3: GPU 数量不足

如果看到错误:
```
ValueError: Total available GPUs X is less than total desired GPUs Y
```

**解决方法**:

修改训练脚本中的 GPU 数量配置:
```bash
# 在 qwen3_agentloop.sh 中修改
trainer.n_gpus_per_node=1  # 改为实际可用的 GPU 数量
```

## 📚 相关文档

- [详细配置说明](./SGLANG_FIXED_PORT_CONFIG.md)
- [SGLang 架构和监控](./SGLANG_ARCHITECTURE_AND_MONITORING.md)
- [监控脚本文档](../scripts/monitor_sglang.py)

## 🎯 关键要点

✅ **固定端口**: 30000 + replica_rank  
✅ **自动发现**: `bash scripts/find_sglang_servers.sh`  
✅ **实时监控**: `python scripts/monitor_sglang.py --port 30000 --scan 8`  
✅ **端口冲突**: 自动尝试下一个端口  
✅ **多节点支持**: 每个节点使用相同的端口范围  

---

*最后更新: 2026-01-13*
