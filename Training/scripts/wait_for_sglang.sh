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
