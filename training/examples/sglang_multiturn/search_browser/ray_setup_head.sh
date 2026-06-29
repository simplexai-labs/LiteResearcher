#!/bin/bash
# Quick-start Ray Head Node (multi-node training entry).
# Run this on the head node BEFORE launching stage1/stage2 scripts.

PROJECT_DIR="${PROJECT_DIR:-/share/project/wanli/Search_Agent/verl}"
CONDA_ENV="${CONDA_ENV:-/share/project/wanli/env/verl-v060}"

cd "$PROJECT_DIR" || { echo "PROJECT_DIR not found: $PROJECT_DIR"; exit 1; }
# shellcheck disable=SC1090
[ -d "$CONDA_ENV" ] && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"

# 获取本机IP地址
# 尝试多种方式获取IP
HEAD_NODE_IP=""

# 方法1: 使用hostname -I
if command -v hostname &> /dev/null; then
    HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
fi

# 方法2: 如果方法1失败，尝试ip命令
if [ -z "$HEAD_NODE_IP" ] && command -v ip &> /dev/null; then
    HEAD_NODE_IP=$(ip route get 8.8.8.8 | grep -oP 'src \K\S+')
fi

# 方法3: 手动设置（如果自动检测失败）
if [ -z "$HEAD_NODE_IP" ]; then
    echo "⚠️  无法自动检测IP地址"
    echo "请手动设置 HEAD_NODE_IP，例如："
    echo "  export HEAD_NODE_IP=10.0.0.1"
    echo "  然后重新运行此脚本"
    exit 1
fi

echo "========================================"
echo "🚀 启动 Ray Head Node"
echo "========================================"
echo "检测到的IP地址: $HEAD_NODE_IP"
echo ""

# 停止现有的Ray进程
echo "1. 停止现有的Ray进程..."
ray stop 2>/dev/null || true
sleep 2

# 启动Ray Head Node
echo "2. 启动Ray Head Node..."
ray start --head \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --port=6379 \
    --node-ip-address=$HEAD_NODE_IP

# 等待Ray完全启动
sleep 3

# 显示状态
echo ""
echo "========================================"
echo "✅ Ray Head Node 启动成功！"
echo "========================================"
echo ""
ray status
echo ""
echo "📋 重要信息："
echo "  - GCS地址: $HEAD_NODE_IP:6379"
echo "  - Dashboard地址: http://$HEAD_NODE_IP:8265"
echo ""
echo "📝 下一步："
echo "  1. 在机器2（Worker Node）上运行："
echo "     bash examples/sglang_multiturn/search_browser/ray_setup_worker.sh $HEAD_NODE_IP"
echo ""
echo "  2. Or manually on the worker node:"
echo "     cd \$PROJECT_DIR"
echo "     conda activate \$CONDA_ENV"
echo "     ray stop"
echo "     ray start --address=$HEAD_NODE_IP:6379"
echo ""
echo "  3. 验证集群状态（在任意机器上）："
echo "     ray status"
echo ""
echo "  4. 启动训练（在Head Node上）："
echo "     bash examples/sglang_multiturn/search_browser/qwen3_agentloop_resume_multinode_from_step12.sh"
echo "========================================"
