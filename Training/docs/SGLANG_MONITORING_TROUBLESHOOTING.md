# SGLang 监控故障诊断与解决方案

> **创建时间**: 2026-01-14 19:33
> **问题**: `check_sglang_servers.sh` 显示所有端口无法连接
> **环境**: VERL v0.6.0 + SGLang 0.5.2

---

## 问题现象

执行 `bash scripts/check_sglang_servers.sh` 后显示：

```
Server  0 (:30000) ... ✗ Not responding
Server  1 (:30001) ... ✗ Not responding
Server  2 (:30002) ... ✗ Not responding
...
Summary: 0 healthy, 8 unhealthy
```

---

## 根本原因分析

### 原因 1: 训练已结束，服务器已关闭 ✅ **主要原因**

**诊断步骤**：

```bash
# 1. 检查训练进程是否在运行
ps aux | grep -E "python.*main_ppo" | grep -v grep

# 2. 检查 SGLang 进程是否在运行
ps aux | grep -E "sglang|SGLang" | grep -v grep

# 3. 检查 Ray 是否在运行
ray status

# 4. 检查日志文件的最后更新时间
stat logs_packing_resume/*.log | grep Modify
```

**当前状态**：
- ❌ 训练进程已停止（无 `main_ppo` 进程）
- ❌ SGLang 服务器已关闭（无 `SGLangHttpServer` 进程）
- ❌ Ray 已停止（`ray status` 返回 "Ray is not running"）
- ❌ 端口无监听（`netstat -tlnp | grep 3000X` 无输出）

**结论**: 训练任务已于 2026-01-14 19:33 结束，所有 SGLang HTTP 服务器随训练进程一起退出。

### 原因 2: 服务器绑定在非 localhost 地址

**症状**: 训练正在运行，但 `localhost` 无法连接

**原因**: VERL 在多节点环境中，SGLang 服务器绑定在节点的实际 IP 地址上，而非 `127.0.0.1`。

**源码位置**: `verl/workers/rollout/sglang_rollout/async_sglang_server.py:94`

```python
self._server_address = ray.util.get_node_ip_address().strip("[]")
# 结果可能是 "172.27.190.39" 而非 "localhost"
```

**解决方案**：

```bash
# 1. 获取本机 IP 地址
hostname -I

# 2. 使用实际 IP 而非 localhost
curl http://172.27.190.39:30000/health_generate

# 3. 修改监控脚本使用实际 IP
python scripts/monitor_sglang.py --host 172.27.190.39 --port 30000
```

### 原因 3: Ray 日志去重隐藏了端口信息

**症状**: 日志中找不到端口号

**原因**: Ray 默认启用日志去重 (`RAY_DEDUP_LOGS=1`)，端口日志被过滤。

**证据**: 日志中有 `[repeated 7x across cluster]` 标记。

**解决方案**：

```bash
# 下次启动训练时禁用日志去重
export RAY_DEDUP_LOGS=0
python -m verl.trainer.main_ppo ...

# 或者在训练脚本中添加
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
```

### 原因 4: 端口分配规则不明确

**问题**: 日志只显示 `replica_rank`，不显示端口号

**原因**: 端口信息在日志中存在但被去重，或 Uvicorn 日志级别设为 "warning"

**端口计算公式**：

```python
# verl/workers/rollout/sglang_rollout/async_sglang_server.py:96
self._base_http_port = 30000
fixed_http_port = self._base_http_port + self.replica_rank

# 结果：
# replica_rank=0 → port 30000
# replica_rank=1 → port 30001
# replica_rank=2 → port 30002
# ...
```

**从日志推算端口**：

```
SGLangHttpServer pid=1141080: replica_rank=1, node_rank=0
→ 端口 = 30000 + 1 = 30001
```

---

## 诊断流程图

```
开始诊断
    ↓
训练是否在运行？
    ├─ 否 → **原因 1**: 训练已结束，服务器已关闭
    │         解决方案: 重启训练或使用已保存的 checkpoint
    │
    └─ 是 → 检查 Ray 是否运行
              ├─ 否 → 启动 Ray: ray start --head
              │
              └─ 是 → 检查端口监听
                    ├─ 无监听 → SGLang 启动失败
                    │         解决方案: 查看日志中的错误信息
                    │
                    └─ 有监听 → 尝试连接
                          ├─ localhost 失败 → **原因 2**: 绑定在非 localhost 地址
                          │                     解决方案: 使用实际 IP 地址
                          │
                          └─ 实际 IP 成功 → 监控正常
```

---

## 完整诊断脚本

保存为 `scripts/diagnose_sglang.sh`:

```bash
#!/bin/bash

echo "=================================="
echo "  SGLang 服务器诊断工具"
echo "=================================="
echo

# 1. 检查训练进程
echo "[1] 检查训练进程..."
TRAINING_PID=$(ps aux | grep -E "python.*main_ppo" | grep -v grep | awk '{print $2}')
if [ -z "$TRAINING_PID" ]; then
    echo "  ❌ 训练进程未运行"
    echo "  → 所有 SGLang 服务器已关闭"
    exit 1
else
    echo "  ✅ 训练进程运行中 (PID: $TRAINING_PID)"
fi
echo

# 2. 检查 Ray 状态
echo "[2] 检查 Ray 状态..."
if ray status &>/dev/null; then
    echo "  ✅ Ray 正在运行"
else
    echo "  ❌ Ray 未运行"
    echo "  → 启动 Ray: ray start --head"
    exit 1
fi
echo

# 3. 检查 SGLang 进程
echo "[3] 检查 SGLang 进程..."
SGLANG_PIDS=$(ps aux | grep -E "SGLangHttpServer" | grep -v grep | awk '{print $2}')
if [ -z "$SGLANG_PIDS" ]; then
    echo "  ❌ SGLang 进程未找到"
    echo "  → 训练可能仍在初始化"
else
    echo "  ✅ 找到 $(echo "$SGLANG_PIDS" | wc -w) 个 SGLang 进程"
fi
echo

# 4. 检查端口监听
echo "[4] 检查端口监听 (30000-30007)..."
HOST_IP=$(hostname -I | awk '{print $1}')
echo "  主机 IP: $HOST_IP"
echo

for port in 30000 30001 30002 30003 30004 30005 30006 30007; do
    if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
        PID=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1)
        echo "  ✅ 端口 $port: 监听中 (PID: $PID)"

        # 尝试连接
        if curl -s http://$HOST_IP:$port/health_generate &>/dev/null; then
            echo "     → HTTP 响应: 正常"
        else
            echo "     → HTTP 响应: 异常"
        fi
    else
        echo "  ❌ 端口 $port: 未监听"
    fi
done
echo

# 5. 检查 localhost 连接
echo "[5] 检查 localhost 连接..."
for port in 30000 30001 30002 30003; do
    if curl -s --connect-timeout 1 http://localhost:$port/health_generate &>/dev/null; then
        echo "  ✅ localhost:$port 可连接"
        exit 0
    fi
done
echo "  ❌ localhost 无法连接"
echo "  → 尝试使用实际 IP: $HOST_IP"
echo

# 6. 检查最近的日志
echo "[6] 检查最近的日志..."
LATEST_LOG=$(ls -t logs_packing_resume/*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "  日志文件: $LATEST_LOG"
    echo
    echo "  最近的 SGLangHttpServer 消息:"
    grep "SGLangHttpServer" "$LATEST_LOG" | grep "replica_rank" | tail -3 | sed 's/^/    /'
    echo
    echo "  最近的端口信息:"
    grep -E "port.*3000|HTTP server started" "$LATEST_LOG" | tail -3 | sed 's/^/    /' || echo "    (未找到端口日志，可能被 Ray 去重)"
else
    echo "  ❌ 未找到日志文件"
fi
echo

echo "=================================="
echo "  诊断完成"
echo "=================================="
```

**使用方法**：

```bash
bash scripts/diagnose_sglang.sh
```

---

## 实时监控方案

### 方案 1: 在训练运行时监控

**前提**: 训练必须正在运行

```bash
# 1. 获取主机 IP
HOST_IP=$(hostname -I | awk '{print $1}')

# 2. 使用实际 IP 监控
python scripts/monitor_sglang.py \
    --host $HOST_IP \
    --port 30000 \
    --scan 8 \
    --interval 5

# 3. 或者直接 curl
curl -s http://$HOST_IP:30000/get_server_info | jq '.internal_states[] | {num_running_reqs, token_usage, cache_hit_rate}'
```

### 方案 2: 修改 `check_sglang_servers.sh`

**问题**: 原脚本只检查 `localhost`

**修复**: 编辑 `scripts/check_sglang_servers.sh`，将：

```bash
HOST="localhost"
```

改为：

```bash
HOST=$(hostname -I | awk '{print $1}')  # 使用实际 IP
```

或添加参数支持：

```bash
#!/bin/bash
HOST=${1:-"localhost"}  # 支持传参，默认 localhost
```

使用：

```bash
# 使用 localhost
bash scripts/check_sglang_servers.sh

# 使用实际 IP
bash scripts/check_sglang_servers.sh 172.27.190.39
```

### 方案 3: 集成到训练日志

**在训练脚本中添加端口输出**：

```python
# verl/trainer/main_ppo.py 或启动脚本中
import os
import socket

def print_server_addresses():
    host_ip = socket.gethostbyname(socket.gethostname())
    base_port = 30000

    print("\n" + "="*60)
    print("SGLang 服务器地址:")
    print("="*60)
    for i in range(num_replicas):  # 从配置获取
        port = base_port + i
        print(f"  Replica {i}: http://{host_ip}:{port}")
        print(f"    健康检查: curl http://{host_ip}:{port}/health_generate")
        print(f"    监控指标: curl http://{host_ip}:{port}/metrics")
    print("="*60 + "\n")

# 在 AgentLoopManager 初始化后调用
print_server_addresses()
```

### 方案 4: 使用 Ray Dashboard

**启动 Dashboard**：

```bash
# 在训练启动时添加
ray start --head --port=8265 --dashboard-host=0.0.0.0

# 访问
http://<节点IP>:8265
```

**可查看**：
- Ray Actor 状态 (SGLangHttpServer, AgentLoopWorker)
- GPU 使用情况
- 任务日志（包括端口信息）

---

## 调试端口问题的技巧

### 技巧 1: 从日志推算端口

```bash
# 查找 replica_rank 信息
grep "SGLangHttpServer.*replica_rank" logs_packing_resume/*.log | grep -v repeated

# 计算端口
# replica_rank=0 → 30000
# replica_rank=1 → 30001
# ...
```

### 技巧 2: 使用 netstat 找到所有监听端口

```bash
# 查找所有 Python 进程监听的端口
netstat -tlnp | grep python | grep -E "3000[0-9]"

# 或使用 ss
ss -tlnp | grep python | grep -E "3000[0-9]"
```

### 技巧 3: 测试所有可能的主机

```bash
# 测试 localhost
for port in 30000 30001 30002 30003 30004 30005 30006 30007; do
    curl -s http://localhost:$port/health_generate && echo "localhost:$port OK"
done

# 测试实际 IP
HOST_IP=$(hostname -I | awk '{print $1}')
for port in 30000 30001 30002 30003 30004 30005 30006 30007; do
    curl -s http://$HOST_IP:$port/health_generate && echo "$HOST_IP:$port OK"
done

# 测试 127.0.0.1
for port in 30000 30001 30002 30003 30004 30005 30006 30007; do
    curl -s http://127.0.0.1:$port/health_generate && echo "127.0.0.1:$port OK"
done
```

### 技巧 4: 查看 Ray Actor 的 IP 地址

```python
import ray

if ray.is_initialized():
    # 获取所有节点
    nodes = ray.nodes()
    for node in nodes:
        if node["Alive"]:
            print(f"Node ID: {node['NodeID']}")
            print(f"IP Address: {node['NodeManagerAddress']}")
```

---

## 预防措施

### 措施 1: 启动时记录端口信息

**在训练脚本开头添加**：

```bash
#!/bin/bash

# 记录主机 IP 和端口
HOST_IP=$(hostname -I | awk '{print $1}')
BASE_PORT=30000

echo "============================================" | tee -a ports.log
echo "训练启动时间: $(date)" | tee -a ports.log
echo "主机 IP: $HOST_IP" | tee -a ports.log
echo "SGLang 服务器端口:" | tee -a ports.log
for i in {0..7}; do
    echo "  Replica $i: http://$HOST_IP:$((BASE_PORT + i))" | tee -a ports.log
done
echo "============================================" | tee -a ports.log
echo

# 启动训练
python -m verl.trainer.main_ppo "$@"
```

### 措施 2: 使用固定端口配置

**在配置文件中明确指定**：

```yaml
# config/rollout/sglang_rollout.yaml
actor_rollout_ref:
  rollout:
    engine_kwargs:
      sglang:
        base_http_port: 30000  # 明确指定
```

### 措施 3: 禁用 Ray 日志去重（用于调试）

```bash
export RAY_DEDUP_LOGS=0
export RAY_LOG_TO_STDERR=true

python -m verl.trainer.main_ppo ...
```

### 措施 4: 添加健康检查脚本

**创建 `scripts/wait_for_sglang.sh`**：

```bash
#!/bin/bash

HOST=${1:-"localhost"}
BASE_PORT=${2:-30000}
NUM_REPLICAS=${3:-8}
TIMEOUT=${4:-300}  # 5 分钟

echo "等待 SGLang 服务器启动..."
echo "主机: $HOST"
echo "端口范围: $BASE_PORT - $((BASE_PORT + NUM_REPLICAS - 1))"

START_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo "❌ 超时：$TIMEOUT 秒内未检测到 SGLang 服务器"
        exit 1
    fi

    HEALTHY=0
    for i in $(seq 0 $((NUM_REPLICAS - 1))); do
        PORT=$((BASE_PORT + i))
        if curl -s http://$HOST:$PORT/health_generate &>/dev/null; then
            echo "  ✅ 端口 $PORT: 就绪"
            HEALTHY=$((HEALTHY + 1))
        fi
    done

    if [ $HEALTHY -gt 0 ]; then
        echo "✅ 检测到 $HEALTHY 个健康的 SGLang 服务器"
        exit 0
    fi

    echo "⏳ 等待中... ($((ELAPSED))s/$TIMEOUT)"
    sleep 5
done
```

**使用**：

```bash
# 在启动训练后立即运行
bash scripts/wait_for_sglang.sh 172.27.190.39 30000 8

# 或在训练脚本中使用
python -m verl.trainer.main_ppo ... &
TRAIN_PID=$!
bash scripts/wait_for_sglang.sh || { kill $TRAIN_PID; exit 1; }
```

---

## 快速参考

### 常用命令

| 命令 | 说明 |
|------|------|
| `ps aux \| grep SGLangHttpServer` | 检查 SGLang 进程 |
| `netstat -tlnp \| grep 3000` | 检查端口监听 |
| `curl http://$HOST_IP:30000/health_generate` | 测试连接 |
| `curl http://$HOST_IP:30000/get_server_info` | 获取服务器信息 |
| `curl http://$HOST_IP:30000/metrics` | 获取 Prometheus 指标 |
| `python scripts/monitor_sglang.py --host $HOST_IP --scan 8` | 监控所有服务器 |
| `bash scripts/diagnose_sglang.sh` | 运行完整诊断 |

### 端口分配规则

```
replica_rank → 端口
0            → 30000
1            → 30001
2            → 30002
3            → 30003
4            → 30004
5            → 30005
6            → 30006
7            → 30007
```

### IP 地址说明

| 地址 | 说明 | 何时可用 |
|------|------|---------|
| `localhost` | 本地回环 | 仅本地访问 |
| `127.0.0.1` | 本地回环 | 仅本地访问 |
| `hostname -I` | 实际 IP (如 172.27.190.39) | 可从其他节点访问 |
| `0.0.0.0` | 监听所有接口 | SGLang 绑定地址 |

---

## 总结

**当前问题**：所有端口无法连接

**主要原因**：训练已结束，SGLang 服务器已关闭

**解决方案**：
1. ✅ 重启训练任务
2. ✅ 使用实际 IP 地址（而非 localhost）
3. ✅ 在训练运行时执行监控脚本
4. ✅ 禁用 Ray 日志去重以查看端口信息

**长期改进**：
- 修改 `check_sglang_servers.sh` 支持自定义 IP
- 在训练启动时自动输出服务器地址
- 使用 `wait_for_sglang.sh` 确保服务器就绪

---

**相关文档**：
- [SGLANG_STARTUP_AND_MONITORING_GUIDE.md](./SGLANG_STARTUP_AND_MONITORING_GUIDE.md)
- [SGLANG_ARCHITECTURE_AND_MONITORING.md](./SGLANG_ARCHITECTURE_AND_MONITORING.md)

**最后更新**: 2026-01-14 19:33
