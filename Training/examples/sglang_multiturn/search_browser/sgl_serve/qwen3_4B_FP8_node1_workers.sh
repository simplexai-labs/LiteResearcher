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
# 主节点 Workers 配置（8卡）
# ==============================================================================
export CUDA_HOME=/usr/local/cuda-12.6 && echo "CUDA_HOME 已设置为: $CUDA_HOME"

SUMMARY_MODEL_PATH="/share/project/wanli/model/Qwen3-4B-Instruct-2507-FP8"
SUMMARY_MAX_MODEL_LEN="50000"
SUMMARY_GPU_MEMORY_UTILIZATION="0.92"

# 获取主节点 IP
MAIN_NODE_IP=$(hostname -I | awk '{print $1}')
echo "主节点 IP: $MAIN_NODE_IP"

# ==============================================================================
# 日志配置
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/log_sglang"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORKER_LOG="${LOG_DIR}/main_node_workers_${TIMESTAMP}.log"

# ==============================================================================
# 启动主节点的 8 个 worker（端口 30001-30008）
# ==============================================================================
echo "========================================="
echo "启动主节点 8 个 worker..."
echo "========================================="

for i in {0..7}; do
    GPU_ID=$i
    PORT=$((30001 + i))
    
    echo "启动 Worker GPU:$GPU_ID Port:$PORT"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m sglang.launch_server \
        --model-path "$SUMMARY_MODEL_PATH" \
        --host 0.0.0.0 \
        --port $PORT \
        --mem-fraction-static "$SUMMARY_GPU_MEMORY_UTILIZATION" \
        --context-length "$SUMMARY_MAX_MODEL_LEN" \
        --attention-backend flashinfer \
        --sampling-backend flashinfer \
        --kv-cache-dtype fp8_e5m2 \
        --chunked-prefill-size 8192 \
        --trust-remote-code \
        2>&1 | tee -a "${WORKER_LOG}" &
done

echo "========================================="
echo "主节点 8 个 worker 已启动"
echo "Worker 端口: 30001-30008"
echo "日志同时输出到终端和文件: $WORKER_LOG"
echo "========================================="
echo ""
echo "提示: Workers 将在后台运行"
echo "      等待约30秒让所有 workers 完全初始化后,再启动 Router"
echo ""

# 保持脚本运行，可以查看日志
wait
