#!/bin/bash

set -e
ulimit -n 65535
# ==============================================================================
# 禁用代理（避免干扰本地服务启动）
# ==============================================================================
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,172.24.178.0/24"
echo "已禁用所有代理"

# ==============================================================================
# Router 配置
# ==============================================================================
ROUTER_PORT="7000"

# 主节点 IP（请根据实际情况修改）
MAIN_NODE_IP=$(hostname -I | awk '{print $1}')
echo "主节点 IP: $MAIN_NODE_IP"

# 副节点 IP（请根据实际情况修改）
SUB_NODE_IP="172.24.37.23"  # 修改为你的副节点 IP
echo "副节点 IP: $SUB_NODE_IP"

# ==============================================================================
# 日志配置
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/log_sglang"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ROUTER_LOG="${LOG_DIR}/router_${TIMESTAMP}.log"

# ==============================================================================
# 构建所有 Worker URLs
# ==============================================================================
echo "========================================="
echo "构建 Worker URLs..."
echo "========================================="

# 用数组存储所有 worker URLs
WORKER_URLS=()

# 添加主节点的 8 个 workers（端口 30001-30008）
for i in {4..7}; do
    PORT=$((30001 + i))
    WORKER_URLS+=("http://${MAIN_NODE_IP}:${PORT}")
done

# # # 添加副节点的 4 个 workers（端口 30001-30004）
# for i in {4..7}; do
#     PORT=$((30001 + i))
#     WORKER_URLS+=("http://${SUB_NODE_IP}:${PORT}")
# done

# echo "所有 Worker URLs:"
# printf '  %s\n' "${WORKER_URLS[@]}"
# echo ""

# ==============================================================================
# 启动 Router
# ==============================================================================
echo "========================================="
echo "启动 Router..."
echo "========================================="

# 构建 --worker-urls 参数（注意是复数形式）
# 方式：所有 worker URLs 放在一个 --worker-urls 参数后面
python -m sglang_router.launch_router \
    --host 0.0.0.0 \
    --port "$ROUTER_PORT" \
    --worker-urls "${WORKER_URLS[@]}" \
    --policy cache_aware \
    --prometheus-port 9091 \
    --max-concurrent-requests 8000 \
    --queue-size 15000 \
    2>&1 | tee -a "$ROUTER_LOG"

echo "========================================="
echo "Router 启动完成！"
echo "Router 地址: http://${MAIN_NODE_IP}:${ROUTER_PORT}"
echo "========================================="
