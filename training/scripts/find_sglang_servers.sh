#!/bin/bash
# SGLang服务器自动发现脚本
# 用于查找所有运行中的SGLang HTTP服务器

START_PORT=${1:-30000}
NUM_PORTS=${2:-16}
HOST=${3:-localhost}

echo "🔍 Searching for SGLang servers..."
echo "   Host: $HOST"
echo "   Port range: $START_PORT - $((START_PORT + NUM_PORTS - 1))"
echo "=================================================="

FOUND_COUNT=0

for i in $(seq 0 $((NUM_PORTS - 1))); do
    PORT=$((START_PORT + i))
    
    # 尝试访问 /health_generate 端点
    if timeout 2 curl -s "http://${HOST}:${PORT}/health_generate" > /dev/null 2>&1; then
        echo ""
        echo "✅ Server #${i} found at port ${PORT}"
        
        # 获取服务器详细信息
        INFO=$(timeout 2 curl -s "http://${HOST}:${PORT}/get_server_info" 2>/dev/null)
        if [ -n "$INFO" ]; then
            MODEL=$(echo "$INFO" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data.get('model_path', 'N/A').split('/')[-1])" 2>/dev/null)
            MAX_TOKENS=$(echo "$INFO" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data.get('max_total_num_tokens', 'N/A'))" 2>/dev/null)
            MAX_REQS=$(echo "$INFO" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data.get('max_running_requests', 'N/A'))" 2>/dev/null)
            
            echo "   📦 Model: $MODEL"
            echo "   📊 Max Total Tokens: $MAX_TOKENS"
            echo "   🎯 Max Running Requests: $MAX_REQS"
            
            # 尝试获取metrics
            METRICS=$(timeout 2 curl -s "http://${HOST}:${PORT}/metrics" 2>/dev/null)
            if [ -n "$METRICS" ]; then
                TOKEN_USAGE=$(echo "$METRICS" | grep "sglang_token_usage" | awk '{print $2}')
                QUEUE_REQS=$(echo "$METRICS" | grep "sglang_num_queue_reqs" | awk '{print $2}')
                RUNNING_REQS=$(echo "$METRICS" | grep "sglang_num_running_reqs" | awk '{print $2}')
                
                if [ -n "$TOKEN_USAGE" ]; then
                    PERCENT=$(echo "$TOKEN_USAGE * 100" | bc 2>/dev/null || echo "N/A")
                    echo "   💾 KV Cache Usage: ${PERCENT}%"
                fi
                if [ -n "$QUEUE_REQS" ]; then
                    echo "   📥 Queue Requests: ${QUEUE_REQS}"
                fi
                if [ -n "$RUNNING_REQS" ]; then
                    echo "   🔄 Running Requests: ${RUNNING_REQS}"
                fi
            fi
        fi
        
        FOUND_COUNT=$((FOUND_COUNT + 1))
    fi
done

echo ""
echo "=================================================="
if [ $FOUND_COUNT -eq 0 ]; then
    echo "❌ No SGLang servers found"
    echo ""
    echo "💡 Troubleshooting:"
    echo "   1. Check if training is running: ps aux | grep main_ppo"
    echo "   2. Check logs: tail -f logs/*.log"
    echo "   3. Try wider port range: $0 30000 32"
else
    echo "✅ Found $FOUND_COUNT SGLang server(s)"
fi
echo ""
