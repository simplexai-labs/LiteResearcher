#!/bin/bash
# Checkpoint实时迁移守护进程
# 监控Ray临时目录中的checkpoint并自动迁移到共享存储

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;36m'
NC='\033[0m'

# 配置
RAY_TMP_BASE="/tmp/ray"
TARGET_BASE="/share/project/wanli/Search_Agent/verl/checkpoints"
CHECK_INTERVAL=300  # 检查间隔(秒), 默认5分钟
LOG_FILE="./logs/checkpoint_sync.log"

# 从命令行参数读取配置
EXPERIMENT_NAME="${1:-qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_48k_nokl}"
CHECK_INTERVAL="${2:-300}"

# 创建日志目录
mkdir -p "$(dirname "$LOG_FILE")"

# 日志函数 - 同时输出到终端和文件
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log_success() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1"
    echo -e "${GREEN}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1"
    echo -e "${RED}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

log_info() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $1"
    echo -e "${BLUE}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

log_warn() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1"
    echo -e "${YELLOW}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

# 已迁移checkpoint记录
MIGRATED_FILE="./logs/migrated_checkpoints.txt"
touch "$MIGRATED_FILE"

# 检查checkpoint是否已迁移
is_migrated() {
    local ckpt_path="$1"
    grep -Fxq "$ckpt_path" "$MIGRATED_FILE" 2>/dev/null
}

# 标记checkpoint为已迁移
mark_migrated() {
    local ckpt_path="$1"
    echo "$ckpt_path" >> "$MIGRATED_FILE"
}

# 迁移单个checkpoint
migrate_checkpoint() {
    local source_path="$1"
    local ckpt_name=$(basename "$source_path")  # global_step_X
    local target_path="$TARGET_BASE/$EXPERIMENT_NAME/$ckpt_name"
    
    log_info "发现checkpoint: $ckpt_name"
    log_info "源路径: $source_path"
    log_info "目标路径: $target_path"
    
    # 检查目标是否已存在
    if [ -d "$target_path" ]; then
        log_info "目标路径已存在，跳过（避免重复复制）"
        # 不重复标记，避免计数重复增加
        return 0
    fi
    
    # 检查是否已迁移过
    if is_migrated "$source_path"; then
        log_info "已迁移过,跳过"
        return 0
    fi
    
    # 检查checkpoint是否完整(等待所有文件写入完成)
    local file_count=$(find "$source_path" -type f 2>/dev/null | wc -l)
    if [ $file_count -lt 10 ]; then
        log_warn "文件数量过少($file_count),可能正在写入,等待下次检查"
        return 0
    fi
    
    # 等待2分钟确保写入完成
    log_info "等待2分钟确保checkpoint完全写入..."
    sleep 120
    
    # 再次检查文件数量是否增加(确认写入完成)
    local new_file_count=$(find "$source_path" -type f 2>/dev/null | wc -l)
    if [ $new_file_count -ne $file_count ]; then
        log_warn "文件仍在增加($file_count -> $new_file_count),等待下次检查"
        return 0
    fi
    
    # 创建目标目录
    mkdir -p "$target_path"
    
    # 开始复制（使用rsync，保留原文件）
    log_info "开始复制 (文件数: $file_count)..."
    
    # 使用rsync复制
    if rsync -a --info=progress2 "$source_path/" "$target_path/" >> "$LOG_FILE" 2>&1; then
        # 验证复制
        local src_size=$(du -sb "$source_path" 2>/dev/null | cut -f1)
        local dst_size=$(du -sb "$target_path" 2>/dev/null | cut -f1)
        
        if [ "$src_size" -eq "$dst_size" ]; then
            log_success "复制成功! 大小: $(du -sh "$target_path" | cut -f1)"
            mark_migrated "$source_path"
            
            # 注意：不删除源文件，保留在Ray临时目录
            log_info "源文件已保留在Ray临时目录中"
        else
            log_error "大小不匹配! 源:$src_size 目标:$dst_size"
            return 1
        fi
    else
        log_error "复制失败!"
        return 1
    fi
}

# 检查磁盘空间
check_disk_space() {
    local usage=$(df /tmp 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ -z "$usage" ]; then
        usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    fi
    
    if [ "$usage" -gt 90 ]; then
        log_error "磁盘空间不足! 使用率: ${usage}%"
        return 1
    elif [ "$usage" -gt 80 ]; then
        log_warn "磁盘空间告警! 使用率: ${usage}%"
    fi
    
    return 0
}

# 主循环
main() {
    # 检测节点标识
    local hostname=$(hostname)
    local node_ip=$(hostname -I | awk '{print $1}')
    
    log_success "======================================"
    log_success "  Checkpoint实时迁移守护进程启动"
    log_success "======================================"
    log_info "节点信息: $hostname ($node_ip)"
    log_info "实验名称: $EXPERIMENT_NAME"
    log_info "检查间隔: ${CHECK_INTERVAL}秒"
    log_info "目标路径: $TARGET_BASE/$EXPERIMENT_NAME"
    log_info "日志文件: $LOG_FILE"
    log_info ""
    log_info "💡 提示: 在tmux中运行,按 Ctrl+B 然后 D 可以detach"
    log_info ""
    
    local iteration=0
    
    while true; do
        iteration=$((iteration + 1))
        log_info ""
        log_info "========== 第 $iteration 次检查 [$(date '+%H:%M:%S')] =========="
        log_info ""
        
        # 检查磁盘空间
        check_disk_space
        
        # 只搜索最新的Ray session中的checkpoint
        # 找到最新的session
        latest_session=$(find "$RAY_TMP_BASE" -maxdepth 1 -type d -name "session_*" 2>/dev/null | sort -V | tail -1)
        
        if [ -z "$latest_session" ]; then
            log_info "未找到Ray session"
        else
            local session_name=$(basename "$latest_session")
            log_info "检查最新session: $session_name"
            
            # 只在最新session中搜索checkpoint (只匹配global_step_*)
            local found_ckpts=$(find "$latest_session" -type d -name "global_step_*" -path "*/$EXPERIMENT_NAME/global_step_*" 2>/dev/null | sort -V)
            
            if [ -z "$found_ckpts" ]; then
                log_info "未找到新checkpoint (正常 - 等待训练保存)"
            else
                local ckpt_count=$(echo "$found_ckpts" | wc -l)
                log_info "找到 $ckpt_count 个checkpoint"
                
                # 复制每个checkpoint
                echo "$found_ckpts" | while read ckpt_path; do
                    if [ -d "$ckpt_path" ]; then
                        migrate_checkpoint "$ckpt_path" || true
                    fi
                done
            fi
        fi
        
        # 显示当前Ray存储使用情况
        local ray_size=$(du -sh "$RAY_TMP_BASE" 2>/dev/null | cut -f1)
        log_info "Ray临时目录大小: $ray_size"
        
        # 显示已迁移checkpoint统计
        local migrated_count=$(wc -l < "$MIGRATED_FILE" 2>/dev/null || echo 0)
        if [ $migrated_count -gt 0 ]; then
            log_success "已成功复制checkpoint数: $migrated_count"
        fi
        
        # 显示目标目录情况
        if [ -d "$TARGET_BASE/$EXPERIMENT_NAME" ]; then
            local target_ckpts=$(find "$TARGET_BASE/$EXPERIMENT_NAME" -name 'global_step_*' -type d 2>/dev/null | wc -l)
            if [ $target_ckpts -gt 0 ]; then
                log_success "共享存储中checkpoint数: $target_ckpts"
                # 列出已有的checkpoint
                find "$TARGET_BASE/$EXPERIMENT_NAME" -name 'global_step_*' -type d 2>/dev/null | while read ckpt; do
                    local step=$(basename "$ckpt" | sed 's/global_step_//')
                    local size=$(du -sh "$ckpt" 2>/dev/null | cut -f1)
                    log_info "  ✓ global_step_$step ($size)"
                done
            fi
        fi
        
        log_info ""
        log_info "⏳ 等待 ${CHECK_INTERVAL}秒 后进行下次检查..."
        log_info "   (按 Ctrl+C 停止守护进程)"
        
        sleep "$CHECK_INTERVAL"
    done
}

# 信号处理
trap 'log_warn "收到终止信号,守护进程退出"; exit 0' SIGINT SIGTERM

# 启动
main
