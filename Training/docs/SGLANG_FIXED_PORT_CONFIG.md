# SGLang 固定端口配置说明

## 📋 修改内容

### 1. 固定端口分配策略

修改后,SGLang HTTP 服务器将使用**固定端口**,从 **30000** 开始:

```
Replica 0 → 30000
Replica 1 → 30001
Replica 2 → 30002
...
Replica N → 3000N
```

### 2. 修改的文件

#### `verl/workers/rollout/utils.py`

修改 `run_unvicorn` 函数,添加 `fixed_port` 参数:

```python
async def run_unvicorn(app: FastAPI, server_args, server_address, 
                       max_retries=5, fixed_port: int = None):
    # 如果指定了fixed_port,使用固定端口
    # 否则使用原来的动态端口分配
```

**特性**:
- 如果 `fixed_port` 被指定,直接使用该端口
- 如果端口被占用,自动尝试下一个端口 (fixed_port + 1, fixed_port + 2, ...)
- 最多重试 5 次

#### `verl/workers/rollout/sglang_rollout/async_sglang_server.py`

在 `SGLangHttpServer` 类中:

1. **添加基础端口配置**:
```python
self._base_http_port = 30000  # 基础端口
```

2. **修改服务器启动逻辑**:
```python
# 使用固定端口: 30000 + replica_rank
fixed_http_port = self._base_http_port + self.replica_rank
self._server_port, self._server_task = await run_unvicorn(
    app, server_args, self._server_address, fixed_port=fixed_http_port
)
```

## 🎯 使用方法

### 1. 自动发现 SGLang 服务器

使用新创建的脚本:

```bash
# 默认扫描 30000-30015 (16个端口)
bash scripts/find_sglang_servers.sh

# 指定起始端口和扫描数量
bash scripts/find_sglang_servers.sh 30000 8

# 指定主机
bash scripts/find_sglang_servers.sh 30000 8 192.168.1.100
```

### 2. 使用监控脚本

```bash
# 监控单个服务器
python scripts/monitor_sglang.py --host localhost --port 30000

# 监控多个服务器
python scripts/monitor_sglang.py --hosts localhost:30000,localhost:30001,localhost:30002

# 自动扫描 (推荐)
python scripts/monitor_sglang.py --host localhost --port 30000 --scan 8
```

### 3. 快速检查

```bash
# 检查特定端口
curl http://localhost:30000/health_generate

# 获取服务器信息
curl http://localhost:30000/get_server_info | python -m json.tool

# 获取 metrics
curl http://localhost:30000/metrics
```

## 📊 端口映射示例

### 8 GPU 单节点训练

假设配置:
- `trainer.n_gpus_per_node=8`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=1` (TP=1)
- `actor_rollout_ref.rollout.data_parallel_size=1` (DP=1)

则会创建 **8 个 SGLang Replicas**:

```
GPU 0 → Replica 0 → HTTP Server: 30000
GPU 1 → Replica 1 → HTTP Server: 30001
GPU 2 → Replica 2 → HTTP Server: 30002
GPU 3 → Replica 3 → HTTP Server: 30003
GPU 4 → Replica 4 → HTTP Server: 30004
GPU 5 → Replica 5 → HTTP Server: 30005
GPU 6 → Replica 6 → HTTP Server: 30006
GPU 7 → Replica 7 → HTTP Server: 30007
```

### 8 GPU 单节点训练 (TP=2)

假设配置:
- `trainer.n_gpus_per_node=8`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=2` (TP=2)

则会创建 **4 个 SGLang Replicas** (每个 replica 使用 2 个 GPU):

```
GPU 0,1 → Replica 0 → HTTP Server: 30000
GPU 2,3 → Replica 1 → HTTP Server: 30001
GPU 4,5 → Replica 2 → HTTP Server: 30002
GPU 6,7 → Replica 3 → HTTP Server: 30003
```

### 多节点训练

假设配置:
- `trainer.nnodes=2`
- `trainer.n_gpus_per_node=8`
- TP=1, DP=1

则每个节点创建 8 个 replicas:

**节点 0**:
```
Replica 0-7 → HTTP Servers: 30000-30007
```

**节点 1**:
```
Replica 8-15 → HTTP Servers: 30008-30015
```

## 🔧 故障排查

### 端口被占用

如果端口被占用,服务器会自动尝试下一个端口。查看日志:

```bash
tail -f logs/*.log | grep "Failed to start HTTP server"
```

如果持续失败,可能需要:

1. **检查端口占用**:
```bash
ss -tlnp | grep 30000
```

2. **修改基础端口**:

在 `async_sglang_server.py` 中修改:
```python
self._base_http_port = 31000  # 改为其他端口段
```

### 服务器未启动

如果找不到服务器:

1. **检查训练是否运行**:
```bash
ps aux | grep main_ppo
```

2. **检查 GPU 可用性**:
```bash
nvidia-smi
```

3. **查看错误日志**:
```bash
tail -100 logs/*.log | grep -i "error\|failed"
```

### 验证端口映射

在训练日志中搜索端口信息:

```bash
grep "HTTP server started on port" logs/*.log
grep "SGLangHttpServer attempting to start on fixed port" logs/*.log
```

## 📝 优势

### 修改前 (动态端口)
- ❌ 端口随机分配,难以预测
- ❌ 每次启动端口不同
- ❌ 难以配置防火墙规则
- ❌ 监控脚本需要扫描所有端口

### 修改后 (固定端口)
- ✅ 端口可预测: 30000 + replica_rank
- ✅ 便于监控和调试
- ✅ 易于配置防火墙
- ✅ 直接通过端口号访问特定 replica
- ✅ 支持多节点部署时的端口规划

## 🚀 快速开始

1. **启动训练**:
```bash
bash examples/sglang_multiturn/search_browser/qwen3_agentloop.sh
```

2. **等待服务器启动** (约 1-2 分钟)

3. **查找服务器**:
```bash
bash scripts/find_sglang_servers.sh
```

4. **开始监控**:
```bash
python scripts/monitor_sglang.py --host localhost --port 30000 --scan 8
```

## 📚 相关文档

- [SGLANG_ARCHITECTURE_AND_MONITORING.md](./SGLANG_ARCHITECTURE_AND_MONITORING.md) - SGLang 架构和监控详细指南
- [ROLLOUT_PROGRESS_MONITORING.md](./ROLLOUT_PROGRESS_MONITORING.md) - Rollout 进度监控
- [监控脚本](../scripts/monitor_sglang.py) - Python 监控工具
- [查找脚本](../scripts/find_sglang_servers.sh) - Bash 自动发现工具

---

*最后更新: 2026-01-13*
