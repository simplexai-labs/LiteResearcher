#!/bin/bash
# DISKANN RAG服务启动脚本

echo "🚀 启动DISKANN RAG服务..."
echo "📈 索引类型: DISKANN (适用于千万级数据)"
echo "🎯 向量精度: FP32 (DISKANN要求)"
echo ""

# 创建logs目录
mkdir -p logs

# 检查端口是否被占用
if lsof -Pi :8018 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  端口8018已被占用"
    echo "   使用以下命令查看占用进程: lsof -i:8018"
    echo "   或使用stop.sh停止服务"
    exit 1
fi

# 启动DISKANN RAG服务
echo "🔧 启动DISKANN RAG服务 (端口8018)..."
nohup python local_rag_diskann_server.py > logs/diskann_server.log 2>&1 &
DISKANN_PID=$!

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 3

# 检查服务状态
if ps -p $DISKANN_PID > /dev/null; then
    echo "✅ DISKANN RAG服务启动成功！"
    echo ""
    echo "📊 服务信息:"
    echo "   DISKANN RAG服务:  http://localhost:8018"
    echo "   API文档:          http://localhost:8018/docs"
    echo "   健康检查:         http://localhost:8018/health"
    echo "   进程ID:           $DISKANN_PID"
    echo ""
    echo "📋 日志文件:"
    echo "   DISKANN服务: logs/diskann_server.log"
    echo ""
    echo "🔍 查看日志:"
    echo "   tail -f logs/diskann_server.log"
    echo ""
    echo "🛑 停止服务:"
    echo "   bash stop.sh"
    echo ""

    # 等待服务完全启动
    echo "⏳ 等待服务完全初始化（加载模型和集合）..."
    sleep 5

    # 测试健康检查
    echo "🔍 测试服务健康..."
    if curl -s http://localhost:8018/health > /dev/null; then
        echo "✅ 服务健康检查通过"
        echo ""
        echo "🎉 DISKANN RAG服务已准备就绪！"
    else
        echo "⚠️  服务可能还在初始化中，请稍后再试"
        echo "   查看日志: tail -f logs/diskann_server.log"
    fi
else
    echo "❌ DISKANN RAG服务启动失败"
    echo "   查看日志: cat logs/diskann_server.log"
    exit 1
fi
