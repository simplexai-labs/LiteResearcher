#!/bin/bash

# ================= 配置 =================
ROUTER_URL="http://127.0.0.1:7000"
NEW_NODE_IP="172.26.79.145"

# 根据您之前的 curl 结果提取的模型路径
MODEL_ID="/share/project/wanli/model/Qwen3-4B-Instruct-2507-FP8"
# =======================================

echo "🚀 开始使用 /workers 接口注册新节点..."
echo "📍 模型 ID: $MODEL_ID"

# 循环注册端口 30001 到 30004
for i in {0..3}; do
    PORT=$((30001 + i))
    WORKER_URL="http://${NEW_NODE_IP}:${PORT}"
    
    echo -n "👉 正在注册 ${WORKER_URL} ... "
    
    # 构造 JSON 数据，必须包含 url 和 model_id
    # 注意：shell 变量引用要小心引号
    JSON_DATA=$(cat <<EOF
{
  "url": "${WORKER_URL}",
  "model_id": "${MODEL_ID}"
}
EOF
)

    # 发送请求
    RESPONSE=$(curl -s -X POST "${ROUTER_URL}/workers" \
      -H "Content-Type: application/json" \
      -d "$JSON_DATA")
    
    echo "Router 响应: $RESPONSE"
done

echo "========================================"
echo "✅ 注册请求已发送"

# 验证环节：列出当前 Worker
echo "🔎 正在验证 Router 中的 Worker 列表..."
curl -s "${ROUTER_URL}/workers" | grep "${NEW_NODE_IP}" && echo "🎉 新节点已出现在列表中！" || echo "⚠️ 列表中未找到新节点，请检查 Router 日志。"