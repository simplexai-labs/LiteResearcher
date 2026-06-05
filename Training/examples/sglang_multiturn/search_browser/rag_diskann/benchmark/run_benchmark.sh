#!/bin/bash
# 🔥 DISKANN RAG 服务压测快速运行脚本
#
# 使用方式:
#   bash run_benchmark.sh [embedding|milvus|query|rollout|all]
#
# 示例:
#   bash run_benchmark.sh embedding    # 只压测 Embedding 服务
#   bash run_benchmark.sh query        # 只压测完整查询流程
#   bash run_benchmark.sh rollout      # Rollout 模式压测
#   bash run_benchmark.sh all          # 运行所有压测

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==================== 配置 ====================

# 默认并发数和请求数
DEFAULT_CONCURRENCY=100
DEFAULT_TOTAL=1000

# Rollout 模式配置
ROLLOUT_WORKERS=8
ROLLOUT_SAMPLES_PER_WORKER=16
ROLLOUT_TOOLS_PER_SAMPLE=2
ROLLOUT_TURNS=5

# 服务地址
EMBEDDING_URL="http://10.160.199.231:8028/embed"
RAG_URL="http://localhost:8018/search"

# ==================== 函数定义 ====================

print_header() {
    echo ""
    echo "================================================================"
    echo "🔥 $1"
    echo "================================================================"
}

run_embedding_benchmark() {
    print_header "Embedding 服务压测"
    echo "📡 服务地址: $EMBEDDING_URL"
    echo "⚡ 并发数: ${1:-$DEFAULT_CONCURRENCY}"
    echo "🔢 请求数: ${2:-$DEFAULT_TOTAL}"
    echo ""
    
    python benchmark_embedding.py \
        --url "$EMBEDDING_URL" \
        --concurrency "${1:-$DEFAULT_CONCURRENCY}" \
        --total "${2:-$DEFAULT_TOTAL}"
}

run_milvus_benchmark() {
    print_header "Milvus 数据库压测"
    echo "⚡ 并发数: ${1:-$DEFAULT_CONCURRENCY}"
    echo "🔢 请求数: ${2:-$DEFAULT_TOTAL}"
    echo ""
    
    python benchmark_milvus.py \
        --concurrency "${1:-$DEFAULT_CONCURRENCY}" \
        --total "${2:-$DEFAULT_TOTAL}"
}

run_query_benchmark() {
    print_header "完整查询流程压测"
    echo "📡 服务地址: $RAG_URL"
    echo "⚡ 并发数: ${1:-$DEFAULT_CONCURRENCY}"
    echo "🔢 请求数: ${2:-$DEFAULT_TOTAL}"
    echo ""
    
    python benchmark_query.py \
        --url "$RAG_URL" \
        --concurrency "${1:-$DEFAULT_CONCURRENCY}" \
        --total "${2:-$DEFAULT_TOTAL}"
}

run_rollout_benchmark() {
    print_header "Rollout 模式压测"
    echo "📡 服务地址: $RAG_URL"
    echo "👥 Workers: $ROLLOUT_WORKERS"
    echo "📊 Samples/Worker: $ROLLOUT_SAMPLES_PER_WORKER"
    echo "🔧 Tools/Sample: $ROLLOUT_TOOLS_PER_SAMPLE"
    echo "🔄 Turns: $ROLLOUT_TURNS"
    total=$((ROLLOUT_WORKERS * ROLLOUT_SAMPLES_PER_WORKER * ROLLOUT_TOOLS_PER_SAMPLE * ROLLOUT_TURNS))
    echo "🔢 总请求数: $total"
    echo ""
    
    python benchmark_query.py \
        --url "$RAG_URL" \
        --rollout-mode \
        --workers "$ROLLOUT_WORKERS" \
        --samples-per-worker "$ROLLOUT_SAMPLES_PER_WORKER" \
        --tools-per-sample "$ROLLOUT_TOOLS_PER_SAMPLE" \
        --turns "$ROLLOUT_TURNS"
}

run_all_benchmarks() {
    echo "🚀 运行所有压测..."
    echo ""
    
    run_embedding_benchmark "$1" "$2"
    echo ""
    
    run_milvus_benchmark "$1" "$2"
    echo ""
    
    run_query_benchmark "$1" "$2"
    echo ""
    
    print_header "所有压测完成!"
}

show_usage() {
    echo "🔥 DISKANN RAG 服务压测工具"
    echo ""
    echo "使用方式:"
    echo "  bash run_benchmark.sh [命令] [并发数] [请求数]"
    echo ""
    echo "命令:"
    echo "  embedding    压测 Embedding 服务"
    echo "  milvus       压测 Milvus 数据库"
    echo "  query        压测完整查询流程"
    echo "  rollout      Rollout 模式压测 (模拟训练场景)"
    echo "  all          运行所有压测"
    echo "  help         显示帮助信息"
    echo ""
    echo "示例:"
    echo "  bash run_benchmark.sh embedding          # 默认配置压测 Embedding"
    echo "  bash run_benchmark.sh query 200 5000     # 200 并发, 5000 请求"
    echo "  bash run_benchmark.sh rollout            # Rollout 模式"
    echo "  bash run_benchmark.sh all 100 1000       # 运行所有压测"
    echo ""
    echo "配置修改:"
    echo "  编辑脚本顶部的配置变量来修改默认值"
}

# ==================== 主逻辑 ====================

case "${1:-help}" in
    embedding)
        run_embedding_benchmark "$2" "$3"
        ;;
    milvus)
        run_milvus_benchmark "$2" "$3"
        ;;
    query)
        run_query_benchmark "$2" "$3"
        ;;
    rollout)
        run_rollout_benchmark
        ;;
    all)
        run_all_benchmarks "$2" "$3"
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "❌ 未知命令: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac

