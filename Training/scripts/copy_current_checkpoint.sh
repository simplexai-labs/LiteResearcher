#!/bin/bash
# 快速复制当前训练的checkpoint到共享存储

set -e

# 最新的checkpoint位置 (从搜索结果中获取)
SOURCE="/tmp/ray/session_2026-01-18_16-43-00_310323_835393/runtime_resources/working_dir_files/_ray_pkg_b5354c25bae8975d/checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_48k_nokl"

# 目标位置 (共享存储)
TARGET="/share/project/wanli/Search_Agent/verl/checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_48k_nokl"

echo "========================================"
echo "  📥 复制Checkpoint到共享存储"
echo "========================================"
echo ""
echo "源路径: $SOURCE"
echo "目标路径: $TARGET"
echo ""

# 检查源路径
if [ ! -d "$SOURCE" ]; then
    echo "❌ 错误: 源路径不存在!"
    echo "   请先运行: bash scripts/find_ray_checkpoints.sh"
    exit 1
fi

# 创建目标目录
mkdir -p "$TARGET"
echo "✅ 创建目标目录"

# 复制checkpoint
echo ""
echo "📥 开始复制..."
echo ""

rsync -av --progress "$SOURCE/" "$TARGET/"

echo ""
echo "========================================"
echo "✅ 复制完成！"
echo ""

# 显示结果
CKPT_COUNT=$(find "$TARGET" -name 'global_step_*' -type d | wc -l)
echo "📊 Checkpoint统计:"
echo "   总数: $CKPT_COUNT"
echo ""

find "$TARGET" -name 'global_step_*' -type d | sort -V | while read ckpt; do
    STEP=$(basename "$ckpt" | sed 's/global_step_//')
    SIZE=$(du -sh "$ckpt" 2>/dev/null | cut -f1)
    echo "   ✅ global_step_$STEP (大小: $SIZE)"
    echo "      路径: $ckpt"
done

echo ""
echo "💡 Resume训练时使用:"
LATEST=$(find "$TARGET" -name 'global_step_*' -type d | sort -V | tail -1)
if [ -n "$LATEST" ]; then
    echo "   trainer.resume_mode=resume_path \\"
    echo "   trainer.resume_from_path=\"$LATEST\""
fi

echo ""
echo "========================================"
