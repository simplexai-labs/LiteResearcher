#!/bin/bash
#
# ==============================================================================
# SGLang Router 优化配置 - 防止高并发熔断
# ==============================================================================
# 基于源码分析：sglang_router-0.2.4
# 熔断器参数：cb_failure_threshold, cb_success_threshold, cb_timeout_duration_secs, cb_window_duration_secs
# ==============================================================================

set -e
ulimit -n 65535

# ==============================================================================
# 禁用代理（避免干扰本地服务启动）
# ==============================================================================
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,172.24.178.0/24"
echo "✅ 已禁用所有代理"

# ==============================================================================
# Router 配置（高并发优化版）
# ==============================================================================
ROUTER_PORT="7000"

# 主节点 IP（请根据实际情况修改）
MAIN_NODE_IP=$(hostname -I | awk '{print $1}')
echo "主节点 IP: $MAIN_NODE_IP"

# 副节点 IP（请根据实际情况修改）
SUB_NODE_IP="172.24.43.83"  # 修改为你的副节点 IP
echo "副节点 IP: $SUB_NODE_IP"

# ==============================================================================
# 关键优化配置
# ==============================================================================
# 来源：/share/project/wanli/env/sglang/lib/python3.10/site-packages/sglang_router/router_args.py

# 1. 并发控制配置
ROUTER_MAX_CONCURRENT=16000       # ⬆️ 从 8000 增加到 16000，处理更高并发
ROUTER_QUEUE_SIZE=30000           # ⬆️⬆️ 从 15000 增加到 30000，大幅增加队列缓冲
ROUTER_QUEUE_TIMEOUT=600          # 队列超时时间（秒）- 保持不变

# 2. 熔断器配置（Circuit Breaker - 关键！）
# 默认值：cb_failure_threshold=10, cb_timeout_duration_secs=60
# 优化策略：提高阈值，缩短恢复时间
CB_FAILURE_THRESHOLD=500           # ⬆️ 从 10 增加到 50，需要更多连续失败才触发熔断
CB_SUCCESS_THRESHOLD=10            # 熔断恢复成功阈值（默认 3，降低到 2 加快恢复）
CB_TIMEOUT_DURATION_SECS=30       # ⬇️ 从 60 减少到 30，加快熔断恢复
CB_WINDOW_DURATION_SECS=120       # 熔断窗口期（秒）- 保持默认

# 3. 健康检查配置（提前发现问题）
HEALTH_FAILURE_THRESHOLD=5        # ⬆️ 从 3 增加到 5，避免误判
HEALTH_SUCCESS_THRESHOLD=2        # 恢复健康阈值（默认 2）
HEALTH_CHECK_TIMEOUT_SECS=10      # ⬆️ 从 5 增加到 10，避免超时误判
HEALTH_CHECK_INTERVAL_SECS=30     # ⬇️ 从 60 减少到 30，更频繁检查

# 4. 重试配置（提高容错性）
RETRY_MAX_RETRIES=20               # 最大重试次数（默认 5）- 保持不变
RETRY_INITIAL_BACKOFF_MS=500       # 初始退避时间（毫秒）- 保持不变
RETRY_MAX_BACKOFF_MS=300000        # 最大退避时间（毫秒）- 保持不变

# 5. 缓存感知路由配置（可选优化）
CACHE_THRESHOLD=0.3               # 缓存阈值（0.0-1.0）- 保持默认
BALANCE_ABS_THRESHOLD=64          # 负载均衡绝对阈值 - 保持默认
BALANCE_REL_THRESHOLD=1.5         # 负载均衡相对阈值 - 保持默认

# ==============================================================================
# 日志配置
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/log_sglang"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ROUTER_LOG="${LOG_DIR}/router_${TIMESTAMP}.log"

# ==============================================================================
# 构建所有 Worker URLs
# ==============================================================================
echo "========================================="
echo "构建 Worker URLs..."
echo "========================================="

# 用数组存储所有 worker URLs
WORKER_URLS=()

# 添加主节点的 8 个 workers（端口 30001-30008）
for i in {0..7}; do
    PORT=$((30001 + i))
    WORKER_URLS+=("http://127.0.0.1:${PORT}")
done

# # # # 可选：添加副节点的 4 个 workers（端口 30001-30004）
# for i in {4..7}; do
#     PORT=$((30001 + i))
#     WORKER_URLS+=("http://${SUB_NODE_IP}:${PORT}")
# done

echo "✅ Worker URLs 配置完成"
echo "   主节点 Workers: 8 个 (端口 30001-30008)"
# echo "   副节点 Workers: 4 个 (端口 30001-30004)"
echo "   总计: ${#WORKER_URLS[@]} 个 Workers"
echo ""

# ==============================================================================
# 启动 Router
# ==============================================================================
echo "========================================="
echo "启动 Router（高并发优化配置）..."
echo "========================================="
echo ""
echo "配置参数："
echo "  并发控制："
echo "    - max_concurrent_requests: ${ROUTER_MAX_CONCURRENT}"
echo "    - queue_size: ${ROUTER_QUEUE_SIZE}"
echo "    - queue_timeout_secs: ${ROUTER_QUEUE_TIMEOUT}"
echo ""
echo "  熔断器配置："
echo "    - cb_failure_threshold: ${CB_FAILURE_THRESHOLD} (连续 ${CB_FAILURE_THRESHOLD} 次失败才熔断)"
echo "    - cb_success_threshold: ${CB_SUCCESS_THRESHOLD} (恢复成功阈值)"
echo "    - cb_timeout_duration_secs: ${CB_TIMEOUT_DURATION_SECS} (熔断恢复时间)"
echo "    - cb_window_duration_secs: ${CB_WINDOW_DURATION_SECS} (熔断窗口期)"
echo ""
echo "  健康检查配置："
echo "    - health_failure_threshold: ${HEALTH_FAILURE_THRESHOLD}"
echo "    - health_success_threshold: ${HEALTH_SUCCESS_THRESHOLD}"
echo "    - health_check_timeout_secs: ${HEALTH_CHECK_TIMEOUT_SECS}"
echo "    - health_check_interval_secs: ${HEALTH_CHECK_INTERVAL_SECS}"
echo ""
echo "  重试配置："
echo "    - retry_max_retries: ${RETRY_MAX_RETRIES}"
echo "    - retry_initial_backoff_ms: ${RETRY_INITIAL_BACKOFF_MS}"
echo "    - retry_max_backoff_ms: ${RETRY_MAX_BACKOFF_MS}"
echo ""
echo "========================================="
echo ""

# 构建 --worker-urls 参数（注意是复数形式）
# 方式：所有 worker URLs 放在一个 --worker-urls 参数后面
# MCP 配置文件路径（增加连接池大小，解决高并发时 "error sending request" 问题）

python -m sglang_router.launch_router \
    --host 0.0.0.0 \
    --port "$ROUTER_PORT" \
    --worker-urls "${WORKER_URLS[@]}" \
    --policy cache_aware \
    --prometheus-port 9091 \
    --max-concurrent-requests "$ROUTER_MAX_CONCURRENT" \
    --queue-size "$ROUTER_QUEUE_SIZE" \
    --queue-timeout-secs "$ROUTER_QUEUE_TIMEOUT" \
    --cb-failure-threshold "$CB_FAILURE_THRESHOLD" \
    --cb-success-threshold "$CB_SUCCESS_THRESHOLD" \
    --cb-timeout-duration-secs "$CB_TIMEOUT_DURATION_SECS" \
    --cb-window-duration-secs "$CB_WINDOW_DURATION_SECS" \
    --health-failure-threshold "$HEALTH_FAILURE_THRESHOLD" \
    --health-success-threshold "$HEALTH_SUCCESS_THRESHOLD" \
    --health-check-timeout-secs "$HEALTH_CHECK_TIMEOUT_SECS" \
    --health-check-interval-secs "$HEALTH_CHECK_INTERVAL_SECS" \
    --retry-max-retries "$RETRY_MAX_RETRIES" \
    --retry-initial-backoff-ms "$RETRY_INITIAL_BACKOFF_MS" \
    --retry-max-backoff-ms "$RETRY_MAX_BACKOFF_MS" \
    2>&1 | tee -a "$ROUTER_LOG"

echo ""
echo "========================================="
echo "✅ Router 启动完成！"
echo "Router 地址: http://${MAIN_NODE_IP}:${ROUTER_PORT}"
echo "Prometheus 指标: http://${MAIN_NODE_IP}:9091"
echo "日志文件: $ROUTER_LOG"
echo "========================================="
