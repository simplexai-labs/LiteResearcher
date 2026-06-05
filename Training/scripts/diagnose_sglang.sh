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
