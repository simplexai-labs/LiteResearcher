#!/bin/bash
# Checkpoint位置检查脚本
# 用于验证多节点训练时checkpoint是否保存在正确位置

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Checkpoint 位置检查工具"
echo "========================================"
echo ""

# 1. 检查预期的checkpoint目录
EXPECTED_DIR="/share/project/wanli/Search_Agent/verl/checkpoints"
echo "📂 检查预期checkpoint目录: $EXPECTED_DIR"

if [ -d "$EXPECTED_DIR" ]; then
    echo -e "${GREEN}✅ 目录存在${NC}"
    
    # 列出最近的checkpoint
    echo ""
    echo "📋 最近的checkpoint:"
    find "$EXPECTED_DIR" -name "global_step_*" -type d -mtime -7 | head -10
    
    # 统计checkpoint数量
    CKPT_COUNT=$(find "$EXPECTED_DIR" -name "global_step_*" -type d | wc -l)
    echo ""
    echo "📊 总共找到 $CKPT_COUNT 个checkpoint"
else
    echo -e "${RED}❌ 目录不存在！${NC}"
fi

echo ""
echo "========================================"

# 2. 检查Ray临时目录中是否有误存的checkpoint
echo "🔍 检查Ray临时目录中的checkpoint (不应该有)..."
RAY_TMP_DIR="/tmp/ray"

if [ -d "$RAY_TMP_DIR" ]; then
    WRONG_CKPTS=$(find "$RAY_TMP_DIR" -name "global_step_*" -type d 2>/dev/null | head -5)
    
    if [ -n "$WRONG_CKPTS" ]; then
        echo -e "${RED}⚠️  警告: 在Ray临时目录中发现checkpoint!${NC}"
        echo -e "${YELLOW}这些checkpoint应该在共享目录中,而不是临时目录!${NC}"
        echo ""
        echo "$WRONG_CKPTS"
        echo ""
        echo -e "${YELLOW}建议: 修改启动脚本使用绝对路径${NC}"
    else
        echo -e "${GREEN}✅ 未在Ray临时目录中发现checkpoint (正常)${NC}"
    fi
else
    echo "ℹ️  Ray临时目录不存在或已清理"
fi

echo ""
echo "========================================"

# 3. 显示最新checkpoint的详细信息
LATEST_CKPT=$(find "$EXPECTED_DIR" -name "global_step_*" -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -n "$LATEST_CKPT" ]; then
    echo "📦 最新checkpoint详情:"
    echo "   路径: $LATEST_CKPT"
    echo ""
    
    # 检查actor目录
    if [ -d "$LATEST_CKPT/actor" ]; then
        ACTOR_FILES=$(ls "$LATEST_CKPT/actor" 2>/dev/null | wc -l)
        MODEL_FILES=$(ls "$LATEST_CKPT/actor"/model_world_size_*_rank_*.pt 2>/dev/null | wc -l)
        echo "   📁 Actor目录: 共 $ACTOR_FILES 个文件"
        echo "   🔢 模型分片: $MODEL_FILES 个"
        
        # 推断world_size
        if [ $MODEL_FILES -gt 0 ]; then
            FIRST_MODEL=$(ls "$LATEST_CKPT/actor"/model_world_size_*_rank_*.pt 2>/dev/null | head -1)
            WORLD_SIZE=$(echo "$FIRST_MODEL" | grep -oP 'world_size_\K[0-9]+')
            echo "   🌍 World Size: $WORLD_SIZE"
            
            # 验证完整性
            if [ $MODEL_FILES -eq $WORLD_SIZE ]; then
                echo -e "   ${GREEN}✅ 所有rank的模型文件都存在${NC}"
            else
                echo -e "   ${RED}❌ 警告: 模型文件不完整 (预期$WORLD_SIZE个,实际$MODEL_FILES个)${NC}"
            fi
        fi
    else
        echo -e "   ${RED}❌ Actor目录不存在${NC}"
    fi
    
    # 检查huggingface目录
    if [ -d "$LATEST_CKPT/actor/huggingface" ]; then
        echo -e "   ${GREEN}✅ HuggingFace配置存在${NC}"
    else
        echo -e "   ${YELLOW}⚠️  HuggingFace配置缺失${NC}"
    fi
    
    # 显示大小
    SIZE=$(du -sh "$LATEST_CKPT" 2>/dev/null | cut -f1)
    echo "   💾 总大小: $SIZE"
else
    echo -e "${YELLOW}⚠️  未找到任何checkpoint${NC}"
fi

echo ""
echo "========================================"
echo "检查完成!"
echo ""
