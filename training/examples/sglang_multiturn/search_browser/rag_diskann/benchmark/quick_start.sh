cd /share/project/wanli/Search_Agent/verl/examples/sglang_multiturn/search_browser/rag_diskann/benchmark

# 1. Embedding 服务压测
python benchmark_embedding.py -c 100 -n 1000

# 2. Milvus 数据库压测
python benchmark_milvus.py -c 100 -n 1000

# 3. 完整查询流程压测
python benchmark_query.py -c 100 -n 1000

# 4. 🔥 Rollout 模式压测（模拟真实训练场景）
python benchmark_query.py --rollout-mode --workers 8 --samples-per-worker 16 --tools-per-sample 2 --turns 5

# 或使用快速脚本
bash run_benchmark.sh query
bash run_benchmark.sh rollout
bash run_benchmark.sh all