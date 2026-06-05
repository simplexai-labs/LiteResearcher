#!/bin/bash
# 将单机8卡的checkpoint转换为2机16卡可用的格式
# 用法: bash convert_checkpoint_8to16.sh <source_checkpoint_path> [target_checkpoint_path]

set -e

# 激活环境
cd /share/project/wanli/Search_Agent/verl
conda activate /share/project/wanli/env/verl-v060

# 参数检查
if [ -z "$1" ]; then
    echo "❌ 错误: 缺少源checkpoint路径"
    echo ""
    echo "用法: bash $0 <source_checkpoint_path> [target_checkpoint_path]"
    echo ""
    echo "示例:"
    echo "  bash $0 /path/to/global_step_12"
    echo "  bash $0 /path/to/global_step_12 /path/to/global_step_12_16gpu"
    echo ""
    exit 1
fi

SOURCE_CKPT=$1
TARGET_CKPT=${2:-"${SOURCE_CKPT}_16gpu"}

echo "========================================"
echo "🔄 FSDP Checkpoint 转换工具"
echo "从 8-GPU → 16-GPU"
echo "========================================"
echo ""
echo "源Checkpoint: $SOURCE_CKPT"
echo "目标Checkpoint: $TARGET_CKPT"
echo ""

# 验证源checkpoint存在
if [ ! -d "$SOURCE_CKPT" ]; then
    echo "❌ 错误: 源checkpoint不存在: $SOURCE_CKPT"
    exit 1
fi

# 检查是否是8-GPU的checkpoint
if [ ! -d "$SOURCE_CKPT/actor" ]; then
    echo "❌ 错误: 未找到actor目录: $SOURCE_CKPT/actor"
    exit 1
fi

MODEL_FILES=$(ls "$SOURCE_CKPT/actor"/model_world_size_8_rank_*.pt 2>/dev/null | wc -l)
if [ "$MODEL_FILES" -ne 8 ]; then
    echo "❌ 错误: 源checkpoint不是8-GPU格式 (找到 $MODEL_FILES 个文件，期望8个)"
    echo "提示: 请确保checkpoint是从单机8卡训练保存的"
    exit 1
fi

echo "✅ 验证通过: 源checkpoint是8-GPU格式"
echo ""

# 检查目标目录
if [ -d "$TARGET_CKPT" ]; then
    echo "⚠️  警告: 目标目录已存在: $TARGET_CKPT"
    read -p "是否覆盖? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消转换"
        exit 0
    fi
    rm -rf "$TARGET_CKPT"
fi

# 转换Actor checkpoint
echo "📦 步骤 1/3: 转换Actor checkpoint..."
python3 verl/utils/checkpoint/convert_fsdp_checkpoint.py \
    --source_ckpt_dir "$SOURCE_CKPT" \
    --target_ckpt_dir "$TARGET_CKPT" \
    --source_world_size 8 \
    --target_world_size 16 \
    --component actor

if [ $? -ne 0 ]; then
    echo "❌ Actor转换失败"
    exit 1
fi
echo "✅ Actor checkpoint转换完成"
echo ""

# 转换Critic checkpoint（如果存在）
if [ -d "$SOURCE_CKPT/critic" ]; then
    echo "📦 步骤 2/3: 转换Critic checkpoint..."
    python3 verl/utils/checkpoint/convert_fsdp_checkpoint.py \
        --source_ckpt_dir "$SOURCE_CKPT" \
        --target_ckpt_dir "$TARGET_CKPT" \
        --source_world_size 8 \
        --target_world_size 16 \
        --component critic
    
    if [ $? -ne 0 ]; then
        echo "❌ Critic转换失败"
        exit 1
    fi
    echo "✅ Critic checkpoint转换完成"
else
    echo "⏭️  步骤 2/3: 跳过Critic (不存在)"
fi
echo ""

# 复制其他文件
echo "📦 步骤 3/3: 复制元数据和配置文件..."

# 复制顶层文件
for file in "$SOURCE_CKPT"/*; do
    if [ -f "$file" ]; then
        cp "$file" "$TARGET_CKPT/"
    fi
done

# 复制huggingface模型文件（如果存在）
if [ -d "$SOURCE_CKPT/actor/huggingface" ]; then
    mkdir -p "$TARGET_CKPT/actor/huggingface"
    cp -r "$SOURCE_CKPT/actor/huggingface"/* "$TARGET_CKPT/actor/huggingface/"
fi

if [ -d "$SOURCE_CKPT/critic/huggingface" ]; then
    mkdir -p "$TARGET_CKPT/critic/huggingface"
    cp -r "$SOURCE_CKPT/critic/huggingface"/* "$TARGET_CKPT/critic/huggingface/"
fi

echo "✅ 元数据复制完成"
echo ""

# 验证转换结果
echo "🔍 验证转换结果..."
TARGET_ACTOR_FILES=$(ls "$TARGET_CKPT/actor"/model_world_size_16_rank_*.pt 2>/dev/null | wc -l)
if [ "$TARGET_ACTOR_FILES" -eq 16 ]; then
    echo "✅ 验证通过: 找到16个Actor模型分片"
else
    echo "⚠️  警告: Actor分片数量不符 (期望16个，实际$TARGET_ACTOR_FILES个)"
fi

# 显示文件大小
SOURCE_SIZE=$(du -sh "$SOURCE_CKPT" | cut -f1)
TARGET_SIZE=$(du -sh "$TARGET_CKPT" | cut -f1)

echo ""
echo "========================================"
echo "✅ 转换完成！"
echo "========================================"
echo "源Checkpoint大小: $SOURCE_SIZE"
echo "目标Checkpoint大小: $TARGET_SIZE"
echo ""
echo "目标Checkpoint保存在:"
echo "  $TARGET_CKPT"
echo ""
echo "📝 下一步:"
echo "  1. 确保两台机器都能访问转换后的checkpoint"
echo "  2. 在训练脚本中使用新的checkpoint路径:"
echo "     CHECKPOINT_PATH=\"$TARGET_CKPT\""
echo "  3. 启动2机16卡训练"
echo "========================================"
