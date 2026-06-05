#!/bin/bash
# LiteResearcher 检索后端启动脚本
# 启动顺序：1) embedding worker(s)  2) 检索服务
#
# 依赖：本机或可达的 Redis（默认 127.0.0.1:6379）、已导入数据的 Milvus。
# 环境变量（可选）：
#   REDIS_HOST / REDIS_PORT   Redis 地址
#   MILVUS_URI                Milvus 地址
#   SEARCH_COLLECTION         检索集合名
#   BGE_MODEL_PATH            BGE-M3 模型路径
#   EMBED_WORKERS             启动多少个 embedding worker（默认 1）

set -e
cd "$(dirname "$0")"
mkdir -p logs

EMBED_WORKERS="${EMBED_WORKERS:-1}"

echo "🚀 启动 LiteResearcher 检索后端"
echo "   Redis:  ${REDIS_HOST:-127.0.0.1}:${REDIS_PORT:-6379}"
echo "   Milvus: ${MILVUS_URI:-http://localhost:19530}"
echo ""

# 1) 启动 embedding worker(s)
for i in $(seq 1 "$EMBED_WORKERS"); do
    echo "🤖 启动 embedding worker #$i ..."
    nohup python embedding_worker.py > "logs/embedding_worker_$i.log" 2>&1 &
    echo "   PID: $!  日志: logs/embedding_worker_$i.log"
done

echo "⏳ 等待 worker 加载模型..."
sleep 8

# 2) 启动检索服务（端口 8018）
if lsof -Pi :8018 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  端口 8018 已被占用，请先 bash stop.sh"
    exit 1
fi

echo "🔧 启动检索服务 (端口 8018) ..."
nohup python local_rag_server.py > logs/rag_server.log 2>&1 &
echo "   PID: $!  日志: logs/rag_server.log"

echo ""
echo "⏳ 等待服务初始化（连接 Milvus + 加载集合）..."
sleep 6

if curl -s http://localhost:8018/health > /dev/null 2>&1; then
    echo "✅ 检索服务健康检查通过"
    echo ""
    echo "🎉 启动完成！"
    echo "   检索服务:  http://localhost:8018"
    echo "   API 文档:  http://localhost:8018/docs"
    echo "   健康检查:  http://localhost:8018/health"
else
    echo "⚠️  服务可能还在初始化，请查看 logs/rag_server.log"
fi
