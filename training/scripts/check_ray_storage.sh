#!/bin/bash
# Ray临时存储空间分析报告

echo "========================================"
echo "  📊 Ray存储空间分析报告"
echo "========================================"
echo ""

# 1. 总体存储情况
echo "1️⃣  总体存储情况"
echo "----------------------------------------"
df -h / | grep -E "Filesystem|overlay|/$" || df -h /tmp
echo ""

# 2. Ray目录占用
echo "2️⃣  Ray目录占用详情"
echo "----------------------------------------"
if [ -d /tmp/ray ]; then
    RAY_SIZE=$(du -sh /tmp/ray 2>/dev/null | cut -f1)
    echo "Ray总大小: $RAY_SIZE"
    echo ""
    
    # 按session统计
    echo "各Session占用:"
    find /tmp/ray -maxdepth 1 -type d -name "session_*" 2>/dev/null | while read session; do
        SIZE=$(du -sh "$session" 2>/dev/null | cut -f1)
        NAME=$(basename "$session")
        echo "  - $NAME: $SIZE"
    done
else
    echo "Ray目录不存在"
fi
echo ""

# 3. Checkpoint占用
echo "3️⃣  Checkpoint占用详情"
echo "----------------------------------------"
CKPTS=$(find /tmp/ray -name "global_step_*" -type d 2>/dev/null)

if [ -n "$CKPTS" ]; then
    TOTAL_SIZE=0
    COUNT=0
    
    echo "$CKPTS" | while read ckpt; do
        SIZE=$(du -sh "$ckpt" 2>/dev/null | cut -f1)
        BYTES=$(du -sb "$ckpt" 2>/dev/null | cut -f1)
        STEP=$(basename "$ckpt" | sed 's/global_step_//')
        MTIME=$(stat -c %y "$ckpt" 2>/dev/null | cut -d'.' -f1)
        
        echo "  ✅ Step $STEP | 大小: $SIZE | 时间: $MTIME"
        COUNT=$((COUNT + 1))
        TOTAL_SIZE=$((TOTAL_SIZE + BYTES))
    done
    
    # 计算总大小
    if [ $COUNT -gt 0 ]; then
        echo ""
        echo "  总计: $COUNT 个checkpoint"
        TOTAL_GB=$((TOTAL_SIZE / 1024 / 1024 / 1024))
        echo "  总大小: ~${TOTAL_GB}GB"
    fi
else
    echo "未找到checkpoint"
fi
echo ""

# 4. 磁盘空间风险评估
echo "4️⃣  磁盘空间风险评估"
echo "----------------------------------------"
USAGE=$(df / 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//')
if [ -z "$USAGE" ]; then
    USAGE=$(df /tmp | tail -1 | awk '{print $5}' | sed 's/%//')
fi

AVAIL=$(df -h / 2>/dev/null | tail -1 | awk '{print $4}')
if [ -z "$AVAIL" ]; then
    AVAIL=$(df -h /tmp | tail -1 | awk '{print $4}')
fi

echo "当前使用率: ${USAGE}%"
echo "剩余空间: $AVAIL"
echo ""

if [ "$USAGE" -gt 90 ]; then
    echo "🔴 风险等级: 严重 (>90%)"
    echo "   建议: 立即清理Ray临时文件或迁移checkpoint"
elif [ "$USAGE" -gt 80 ]; then
    echo "🟡 风险等级: 警告 (80-90%)"
    echo "   建议: 尽快迁移checkpoint,监控磁盘使用"
elif [ "$USAGE" -gt 70 ]; then
    echo "🟢 风险等级: 注意 (70-80%)"
    echo "   建议: 启用自动checkpoint迁移"
else
    echo "🟢 风险等级: 正常 (<70%)"
    echo "   建议: 继续监控"
fi
echo ""

# 5. 预估空间需求
echo "5️⃣  空间需求预估"
echo "----------------------------------------"

# 单个checkpoint大小(基于现有checkpoint)
if [ -n "$CKPTS" ]; then
    FIRST_CKPT=$(echo "$CKPTS" | head -1)
    CKPT_SIZE_BYTES=$(du -sb "$FIRST_CKPT" 2>/dev/null | cut -f1)
    CKPT_SIZE_GB=$((CKPT_SIZE_BYTES / 1024 / 1024 / 1024))
    
    echo "单个checkpoint大小: ~${CKPT_SIZE_GB}GB"
    echo ""
    echo "预估空间需求 (基于save_freq=2):"
    echo "  - 保留最近2个: ~$((CKPT_SIZE_GB * 2))GB"
    echo "  - 保留最近5个: ~$((CKPT_SIZE_GB * 5))GB"
    echo "  - 保留最近10个: ~$((CKPT_SIZE_GB * 10))GB"
else
    echo "无法预估(未找到checkpoint)"
fi
echo ""

# 6. 建议操作
echo "6️⃣  建议操作"
echo "----------------------------------------"
echo "如果磁盘空间紧张:"
echo ""
echo "1. 启动自动checkpoint迁移:"
echo "   bash scripts/start_checkpoint_sync.sh 180"
echo ""
echo "2. 手动迁移现有checkpoint:"
echo "   bash scripts/copy_current_checkpoint.sh"
echo ""
echo "3. 清理旧的Ray session:"
echo "   find /tmp/ray -name 'session_*' -mtime +1 -exec rm -rf {} +"
echo ""
echo "4. 设置checkpoint保留数量:"
echo "   trainer.max_actor_ckpt_to_keep=3"
echo ""

echo "========================================"
