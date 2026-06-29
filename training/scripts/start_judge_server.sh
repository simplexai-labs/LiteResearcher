#!/bin/bash
# 启动vLLM Judge服务器
#
# 使用说明：
# 1. 修改下面的配置参数
# 2. 在单独的终端运行此脚本
# 3. 保持此终端运行，不要关闭

set -e

# ============================================================================
# 配置部分
# ============================================================================

# Judge模型配置
JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # 可以改为3B以节省显存
GPU_ID="0"  # 使用哪个GPU运行judge

# 服务配置
HOST="0.0.0.0"  # 允许远程访问，如果只本地用可以改为127.0.0.1
PORT="8000"

# 性能配置
TENSOR_PARALLEL_SIZE=1  # TP大小，通常1就够
MAX_MODEL_LEN=2048      # 最大序列长度，judge不需要太长
GPU_MEMORY_UTIL=0.5     # GPU显存利用率，留一半给训练

# 环境配置
CONDA_ENV="/share/project/wanli/env/verl-v060"

# ============================================================================
# 启动服务
# ============================================================================

echo "================================================================"
echo "Starting vLLM Judge Server"
echo "================================================================"
echo "Model: $JUDGE_MODEL"
echo "GPU: $GPU_ID"
echo "Host: $HOST:$PORT"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo "================================================================"
echo

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# 检查GPU是否可用
if ! nvidia-smi -i $GPU_ID > /dev/null 2>&1; then
    echo "✗ Error: GPU $GPU_ID is not available"
    nvidia-smi
    exit 1
fi

echo "Starting vLLM server on GPU $GPU_ID..."
echo "Press Ctrl+C to stop the server"
echo

# 启动vLLM服务
CUDA_VISIBLE_DEVICES=$GPU_ID python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code

# 如果脚本到这里说明服务被停止了
echo
echo "================================================================"
echo "vLLM Judge Server stopped"
echo "================================================================"
