#!/bin/bash
# 将 FSDP checkpoint 转换为 HuggingFace 格式
# 用法: bash scripts/convert_my_checkpoint_to_hf.sh

set -e

# 激活环境
echo "🔧 激活 conda 环境..."
source /opt/conda/etc/profile.d/conda.sh
conda activate verl-v060

# 切换到项目根目录
cd /share/project/wanli/Search_Agent/verl

# ==================== 配置区域 ====================

# 源 FSDP checkpoint 的 actor 目录（必需）
# 指向 checkpoint 的 actor 子目录
SOURCE_ACTOR_DIR="checkpoints/qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_32k_nokl/global_step_18/actor"

# 输出的 HuggingFace 模型目录（必需）
# 这是转换后的模型保存路径，可以直接用 AutoModelForCausalLM.from_pretrained() 加载
TARGET_HF_DIR="checkpoints/qwen3_hf_model/global_step_18"

# ==================== 执行区域 ====================

echo "========================================"
echo "🔄 FSDP → HuggingFace 转换工具"
echo "========================================"
echo ""
echo "📂 源 Checkpoint:"
echo "   $SOURCE_ACTOR_DIR"
echo ""
echo "💾 输出路径:"
echo "   $TARGET_HF_DIR"
echo ""

# 验证源 checkpoint 存在
if [ ! -d "$SOURCE_ACTOR_DIR" ]; then
    echo "❌ 错误: 源 checkpoint 目录不存在"
    echo "   路径: $SOURCE_ACTOR_DIR"
    exit 1
fi

# 检查是否有 FSDP 分片文件
MODEL_FILES=$(ls "$SOURCE_ACTOR_DIR"/model_world_size_*_rank_*.pt 2>/dev/null | wc -l)
if [ "$MODEL_FILES" -eq 0 ]; then
    echo "❌ 错误: 未找到 FSDP 模型分片文件"
    echo "   期望格式: model_world_size_*_rank_*.pt"
    exit 1
fi

echo "✅ 找到 $MODEL_FILES 个 FSDP 分片文件"
echo ""

# 创建输出目录
mkdir -p "$TARGET_HF_DIR"

# 执行转换
echo "🚀 开始转换..."
echo ""

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$SOURCE_ACTOR_DIR" \
    --target_dir "$TARGET_HF_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 转换失败！"
    exit 1
fi

echo ""
echo "========================================"
echo "✅ 转换完成！"
echo "========================================"
echo ""
echo "📂 HuggingFace 模型保存在:"
echo "   $(pwd)/$TARGET_HF_DIR"
echo ""
echo "📊 文件大小:"
du -sh "$TARGET_HF_DIR"
echo ""

# 显示输出目录内容
echo "📁 输出目录内容:"
ls -lh "$TARGET_HF_DIR"
echo ""

echo "🚀 使用方法:"
echo ""
echo "Python 代码加载:"
echo "  from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "  model = AutoModelForCausalLM.from_pretrained('$(pwd)/$TARGET_HF_DIR')"
echo "  tokenizer = AutoTokenizer.from_pretrained('$(pwd)/$TARGET_HF_DIR')"
echo ""
echo "或者使用相对路径:"
echo "  model = AutoModelForCausalLM.from_pretrained('$TARGET_HF_DIR')"
echo ""
echo "使用 vLLM 部署:"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "    --model $(pwd)/$TARGET_HF_DIR \\"
echo "    --tensor-parallel-size 8"
echo ""
echo "========================================"
