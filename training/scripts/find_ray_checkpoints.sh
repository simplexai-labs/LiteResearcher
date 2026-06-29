#!/bin/bash
# 快速查找当前运行训练的checkpoint位置

echo "🔍 正在搜索Ray临时目录中的checkpoint..."
echo ""

# 在本机搜索
echo "📍 本机 ($(hostname)):"
CKPTS=$(find /tmp/ray -name "global_step_*" -type d 2>/dev/null | head -20)

if [ -n "$CKPTS" ]; then
    echo "$CKPTS" | while read ckpt; do
        STEP=$(basename "$ckpt" | sed 's/global_step_//')
        SIZE=$(du -sh "$ckpt" 2>/dev/null | cut -f1)
        MTIME=$(stat -c %y "$ckpt" 2>/dev/null | cut -d' ' -f1,2 | cut -d'.' -f1)
        echo "  ✅ Step $STEP | 大小: $SIZE | 修改时间: $MTIME"
        echo "     路径: $ckpt"
    done
else
    echo "  ❌ 未找到"
fi

echo ""
echo "📍 远程机器 (172.27.244.90):"
ssh -o ConnectTimeout=3 172.27.244.90 'find /tmp/ray -name "global_step_*" -type d 2>/dev/null | head -20' 2>/dev/null | while read ckpt; do
    STEP=$(basename "$ckpt" | sed 's/global_step_//')
    echo "  ✅ Step $STEP"
    echo "     路径: $ckpt"
done || echo "  ⚠️  无法访问远程机器"

echo ""
echo "========================================"
echo "💡 复制命令:"
echo "   bash scripts/rescue_checkpoints.sh"
