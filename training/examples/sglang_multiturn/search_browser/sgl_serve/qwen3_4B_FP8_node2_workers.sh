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
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
echo "已禁用所有代理"

# ==============================================================================
# 副节点 Workers 配置（4卡）
# ==============================================================================
export CUDA_HOME=/usr/local/cuda-12.6 && echo "CUDA_HOME 已设置为: $CUDA_HOME"

SUMMARY_MODEL_PATH="/share/project/wanli/model/Qwen3-4B-Instruct-2507-FP8"
SUMMARY_MAX_MODEL_LEN="50000"
SUMMARY_GPU_MEMORY_UTILIZATION="0.92"

# 获取副节点 IP
SUB_NODE_IP=$(hostname -I | awk '{print $1}')
echo "副节点 IP: $SUB_NODE_IP"

# ==============================================================================
# 日志配置
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/log_sglang"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORKER_LOG="${LOG_DIR}/sub_node_workers_${TIMESTAMP}.log"

# ==============================================================================
# 启动副节点的 4 个 worker（使用后四张卡 GPU 4,5,6,7，端口 30001-30004）
# ==============================================================================
echo "========================================="
echo "启动副节点 4 个 worker（使用 GPU 4,5,6,7）..."
echo "========================================="

for i in {4..7}; do
    GPU_ID=$((0 + i))  # 使用 GPU 4,5,6,7
    PORT=$((30001 + i))  # 端口 30001-30004
    
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
echo "副节点 4 个 worker 已启动（GPU 4,5,6,7）"
echo "Worker 端口: 30001-30004"
echo "日志同时输出到终端和文件: $WORKER_LOG"
echo "========================================="
echo ""
echo "提示: Workers 将在后台运行"
echo "      等待主节点也启动完 workers 后,在主节点启动 Router"
echo ""

# 保持脚本运行，可以查看日志
wait
