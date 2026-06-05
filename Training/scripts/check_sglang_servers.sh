#!/bin/bash
# SGLang 服务器快速检查脚本
#
# 用法:
#   ./scripts/check_sglang_servers.sh [起始端口] [服务器数量] [主机]
#
# 示例:
#   ./scripts/check_sglang_servers.sh              # 检查 localhost:30000 的 8 个服务器
#   ./scripts/check_sglang_servers.sh 30000 4      # 检查 4 个服务器
#   ./scripts/check_sglang_servers.sh 30000 8 10.0.0.1  # 检查远程主机

START_PORT=${1:-30000}
NUM_SERVERS=${2:-8}
HOST=${3:-localhost}

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           SGLang Server Health Check                        ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Host: $HOST"
echo "║  Ports: $START_PORT - $((START_PORT + NUM_SERVERS - 1))"
echo "║  Servers: $NUM_SERVERS"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

HEALTHY=0
UNHEALTHY=0

for i in $(seq 0 $((NUM_SERVERS - 1))); do
    PORT=$((START_PORT + i))
    printf "Server %2d (:%d) ... " "$i" "$PORT"
    
    # 检查健康状态
    HEALTH_RESP=$(curl -s --connect-timeout 2 --max-time 5 "http://${HOST}:${PORT}/health_generate" 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        echo -e "\033[32m✓ OK\033[0m"
        ((HEALTHY++))
        
        # 获取服务器信息
        INFO=$(curl -s --max-time 5 "http://${HOST}:${PORT}/get_server_info" 2>/dev/null)
        if [ -n "$INFO" ]; then
            MAX_REQS=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('max_running_requests', 'N/A'))" 2>/dev/null)
            MAX_TOKENS=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('max_total_num_tokens', 'N/A'))" 2>/dev/null)
            
            echo "         ├─ Max Running Requests: $MAX_REQS"
            echo "         └─ Max Total Tokens: $MAX_TOKENS"
        fi
        
        # 尝试获取 metrics
        METRICS=$(curl -s --max-time 5 "http://${HOST}:${PORT}/metrics" 2>/dev/null)
        if [ -n "$METRICS" ] && [[ "$METRICS" != *"404"* ]]; then
            QUEUE=$(echo "$METRICS" | grep "^sglang_num_queue_reqs" | awk '{print $2}')
            RUNNING=$(echo "$METRICS" | grep "^sglang_num_running_reqs" | awk '{print $2}')
            USAGE=$(echo "$METRICS" | grep "^sglang_token_usage" | awk '{print $2}')
            
            if [ -n "$QUEUE" ] || [ -n "$RUNNING" ]; then
                echo "         └─ 📊 Queue: ${QUEUE:-N/A}, Running: ${RUNNING:-N/A}, KV Usage: ${USAGE:-N/A}"
            fi
        fi
    else
        echo -e "\033[31m✗ Not responding\033[0m"
        ((UNHEALTHY++))
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Summary: $HEALTHY healthy, $UNHEALTHY unhealthy"
echo "════════════════════════════════════════════════════════════"

# 返回退出码
if [ $UNHEALTHY -gt 0 ]; then
    exit 1
fi
exit 0
