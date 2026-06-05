#!/bin/bash
# DISKANN RAG服务停止脚本

echo "🛑 停止DISKANN RAG服务..."

# 查找并停止DISKANN服务
if lsof -Pi :8018 -sTCP:LISTEN -t >/dev/null ; then
    echo "🔍 发现DISKANN服务运行在端口8018"
    PID=$(lsof -t -i:8018)
    echo "   进程ID: $PID"
    echo "   正在停止..."
    kill $PID
    sleep 2

    # 检查是否还在运行
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  正常停止失败，尝试强制停止..."
        kill -9 $PID
        sleep 1
    fi

    if ! ps -p $PID > /dev/null 2>&1; then
        echo "✅ DISKANN服务已停止"
    else
        echo "❌ 无法停止进程 $PID"
        exit 1
    fi
else
    echo "ℹ️  未发现DISKANN服务运行"
fi

echo ""
echo "✅ 所有服务已停止"
