#!/bin/bash
# Checkpoint救援脚本
# 用于从Ray临时目录中找到并复制checkpoint到共享存储

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;36m'
NC='\033[0m' # No Color

echo "========================================"
echo "  🚑 Checkpoint 救援工具"
echo "========================================"
echo ""

# 根据日志信息,Ray session ID
RAY_SESSION="session_2026-01-18_16-43-00_310323_835393"
RAY_PKG="runtime_resources/working_dir_files/_ray_pkg_b5354c25bae8975d"

# 目标checkpoint路径(相对于Ray工作目录)
RELATIVE_CKPT_PATH="checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_48k_nokl"

# 目标存储路径(共享存储)
TARGET_BASE="/share/project/wanli/Search_Agent/verl/checkpoints"
TARGET_PATH="$TARGET_BASE/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_48k_nokl"

echo "📋 当前训练信息:"
echo "   Ray Session: $RAY_SESSION"
echo "   目标路径: $TARGET_PATH"
echo ""

# 1. 在所有可能的位置搜索checkpoint
echo "🔍 步骤1: 搜索checkpoint..."
echo ""

FOUND_PATHS=()

# 搜索本机
LOCAL_RAY_PATH="/tmp/ray/$RAY_SESSION/$RAY_PKG/$RELATIVE_CKPT_PATH"
if [ -d "$LOCAL_RAY_PATH" ]; then
    echo -e "${GREEN}✅ 在本机找到: $LOCAL_RAY_PATH${NC}"
    FOUND_PATHS+=("$LOCAL_RAY_PATH")
else
    echo -e "${YELLOW}⚠️  本机未找到${NC}"
fi

# 搜索另一台机器 (需要ssh访问)
REMOTE_HOST="172.27.244.90"  # 从日志中看到的另一台机器IP
REMOTE_RAY_PATH="/tmp/ray/$RAY_SESSION/$RAY_PKG/$RELATIVE_CKPT_PATH"

echo ""
echo "🌐 检查远程机器 ($REMOTE_HOST)..."
if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$REMOTE_HOST" "[ -d '$REMOTE_RAY_PATH' ]" 2>/dev/null; then
    echo -e "${GREEN}✅ 在远程机器找到: $REMOTE_RAY_PATH${NC}"
    FOUND_PATHS+=("$REMOTE_HOST:$REMOTE_RAY_PATH")
else
    echo -e "${YELLOW}⚠️  远程机器未找到或无法访问${NC}"
fi

echo ""
echo "========================================"

if [ ${#FOUND_PATHS[@]} -eq 0 ]; then
    echo -e "${RED}❌ 错误: 未找到任何checkpoint！${NC}"
    echo ""
    echo "💡 提示:"
    echo "   1. 确认训练是否已保存checkpoint (检查日志中的 'Saved model to')"
    echo "   2. Ray session可能已经过期"
    echo "   3. 尝试手动搜索:"
    echo "      find /tmp/ray -name 'global_step_*' -type d 2>/dev/null"
    exit 1
fi

# 2. 列出找到的checkpoint
echo ""
echo "📦 步骤2: 列出找到的checkpoint..."
echo ""

for path in "${FOUND_PATHS[@]}"; do
    if [[ $path == *:* ]]; then
        # 远程路径
        HOST="${path%%:*}"
        REMOTE_PATH="${path#*:}"
        echo "📍 远程 ($HOST):"
        ssh "$HOST" "find '$REMOTE_PATH' -name 'global_step_*' -type d 2>/dev/null | sort" || true
    else
        # 本地路径
        echo "📍 本机:"
        find "$path" -name 'global_step_*' -type d 2>/dev/null | sort
    fi
done

echo ""
echo "========================================"

# 3. 询问是否复制
echo ""
echo "🚀 步骤3: 复制checkpoint到共享存储"
echo ""
echo "目标路径: $TARGET_PATH"
echo ""
read -p "是否开始复制? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消操作"
    exit 0
fi

# 创建目标目录
mkdir -p "$TARGET_PATH"
echo -e "${BLUE}✅ 创建目标目录${NC}"

# 4. 复制checkpoint
echo ""
echo "📥 开始复制..."
echo ""

COPY_SUCCESS=0

for path in "${FOUND_PATHS[@]}"; do
    if [[ $path == *:* ]]; then
        # 从远程复制
        HOST="${path%%:*}"
        REMOTE_PATH="${path#*:}"
        echo "📡 从远程机器复制 ($HOST)..."
        
        # 使用rsync复制
        rsync -avz --progress "$HOST:$REMOTE_PATH/" "$TARGET_PATH/" && COPY_SUCCESS=1
    else
        # 从本地复制
        echo "📂 从本机复制..."
        rsync -av --progress "$path/" "$TARGET_PATH/" && COPY_SUCCESS=1
    fi
done

echo ""
echo "========================================"

if [ $COPY_SUCCESS -eq 1 ]; then
    echo -e "${GREEN}✅ 复制完成！${NC}"
    echo ""
    echo "📊 checkpoint信息:"
    
    # 统计checkpoint
    CKPT_COUNT=$(find "$TARGET_PATH" -name 'global_step_*' -type d | wc -l)
    echo "   总共: $CKPT_COUNT 个checkpoint"
    
    # 列出checkpoint
    echo ""
    echo "📋 可用checkpoint:"
    find "$TARGET_PATH" -name 'global_step_*' -type d | sort | while read ckpt; do
        STEP=$(basename "$ckpt" | sed 's/global_step_//')
        SIZE=$(du -sh "$ckpt" 2>/dev/null | cut -f1)
        echo "   - global_step_$STEP (大小: $SIZE)"
    done
    
    echo ""
    echo "💡 Resume训练时使用:"
    LATEST_CKPT=$(find "$TARGET_PATH" -name 'global_step_*' -type d | sort -V | tail -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "   trainer.resume_from_path=\"$LATEST_CKPT\""
    fi
else
    echo -e "${RED}❌ 复制失败！${NC}"
    exit 1
fi

echo ""
echo "========================================"
echo "完成!"
