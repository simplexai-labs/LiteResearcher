#!/bin/bash
# 快速启动checkpoint同步守护进程

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# 实验名称(从训练配置中读取)
EXPERIMENT="qwen3_deepresearch_tis_rl/local_rag_only_temp_0.7_length_48k_nokl"

# 检查间隔(秒)
# 推荐: 180秒(3分钟) - 300秒(5分钟)
INTERVAL="${1:-300}"

echo "========================================"
echo "  🚀 启动Checkpoint实时同步守护进程"
echo "========================================"
echo ""
echo "实验: $EXPERIMENT"
echo "间隔: ${INTERVAL}秒"
echo ""
echo "日志: ./logs/checkpoint_sync.log"
echo "进程监控: ps aux | grep checkpoint_sync_daemon"
echo ""
echo "停止守护进程: pkill -f checkpoint_sync_daemon"
echo ""
echo "========================================"
echo ""

# 后台运行守护进程
nohup bash "$SCRIPT_DIR/checkpoint_sync_daemon.sh" "$EXPERIMENT" "$INTERVAL" > /dev/null 2>&1 &

DAEMON_PID=$!
echo "✅ 守护进程已启动 (PID: $DAEMON_PID)"
echo ""
echo "💡 查看日志:"
echo "   tail -f logs/checkpoint_sync.log"
echo ""
echo "💡 查看进程:"
echo "   ps aux | grep $DAEMON_PID"
echo ""
