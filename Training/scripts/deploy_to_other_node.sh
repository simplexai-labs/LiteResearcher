#!/bin/bash
# 一键部署到另一台机器

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

OTHER_NODE="172.27.244.90"
PROJECT_DIR="/share/project/wanli/Search_Agent/verl"

echo -e "${GREEN}========================================"
echo "  📤 部署到另一台机器"
echo "========================================${NC}"
echo ""
echo "目标节点: $OTHER_NODE"
echo "项目目录: $PROJECT_DIR"
echo ""

# 检查连接
echo "🔍 检查SSH连接..."
if ssh -o ConnectTimeout=5 "$OTHER_NODE" "echo ok" &>/dev/null; then
    echo -e "${GREEN}✅ SSH连接正常${NC}"
else
    echo -e "${RED}❌ 无法连接到 $OTHER_NODE${NC}"
    exit 1
fi

echo ""
echo "📋 准备复制脚本文件..."
echo ""

# 确保目标机器上有scripts目录
ssh "$OTHER_NODE" "mkdir -p $PROJECT_DIR/scripts $PROJECT_DIR/logs"

# 复制脚本
SCRIPTS=(
    "checkpoint_sync_daemon.sh"
    "dual_node_sync_guide.sh"
    "check_ray_storage.sh"
    "find_ray_checkpoints.sh"
)

for script in "${SCRIPTS[@]}"; do
    echo "  📄 复制 $script..."
    scp "$PROJECT_DIR/scripts/$script" "$OTHER_NODE:$PROJECT_DIR/scripts/" >/dev/null 2>&1
    ssh "$OTHER_NODE" "chmod +x $PROJECT_DIR/scripts/$script"
done

echo ""
echo -e "${GREEN}✅ 复制完成！${NC}"
echo ""

echo "========================================"
echo "  🚀 在另一台机器上启动"
echo "========================================"
echo ""
echo "执行以下命令:"
echo ""
echo -e "${YELLOW}ssh $OTHER_NODE${NC}"
echo ""
echo "# 查看指南"
echo "cd $PROJECT_DIR && bash scripts/dual_node_sync_guide.sh"
echo ""
echo "# 在tmux中启动"
echo "tmux new -s checkpoint_sync"
echo "cd $PROJECT_DIR && bash scripts/checkpoint_sync_daemon.sh \"qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_48k_nokl\" 180"
echo ""
echo "# Detach: Ctrl+B 然后 D"
echo ""
echo "========================================"
