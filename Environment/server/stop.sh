#!/bin/bash
# LiteResearcher 检索后端停止脚本

echo "🛑 停止 LiteResearcher 检索后端..."

# 1) 停止检索服务（端口 8018）
if lsof -Pi :8018 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    PID=$(lsof -t -i:8018)
    echo "🔍 检索服务 PID: $PID，正在停止..."
    kill "$PID" 2>/dev/null || true
    sleep 2
    kill -9 "$PID" 2>/dev/null || true
    echo "✅ 检索服务已停止"
else
    echo "ℹ️  未发现检索服务运行在 8018"
fi

# 2) 停止 embedding worker(s)
WPIDS=$(pgrep -f "embedding_worker.py" || true)
if [ -n "$WPIDS" ]; then
    echo "🔍 embedding worker PID: $WPIDS，正在停止..."
    kill $WPIDS 2>/dev/null || true
    sleep 1
    kill -9 $WPIDS 2>/dev/null || true
    echo "✅ embedding worker 已停止"
else
    echo "ℹ️  未发现 embedding worker 运行"
fi

echo ""
echo "✅ 所有服务已停止"
