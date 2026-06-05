#!/bin/bash
# 快速测试rollout monitor是否正常工作

set -e

echo "=========================================="
echo "测试 Rollout Progress Monitor"
echo "=========================================="
echo ""

# 激活环境
cd /share/project/wanli/Search_Agent/verl
conda activate /share/project/wanli/env/verl-v060

echo "1️⃣  测试Python语法..."
python3 -m py_compile verl/experimental/agent_loop/agent_loop.py
python3 -m py_compile verl/utils/rollout_progress.py
echo "✅ 语法检查通过"
echo ""

echo "2️⃣  测试模块导入..."
python3 << 'EOF'
import sys
sys.path.insert(0, '/share/project/wanli/Search_Agent/verl')

try:
    from verl.utils.rollout_progress import RolloutProgressMonitor
    print("✅ RolloutProgressMonitor 导入成功")
    
    # 测试基本功能
    monitor = RolloutProgressMonitor(
        total_samples=32,
        step=0,
        enable_progress_bar=False,  # 测试环境不显示进度条
        enable_logging=False,
        worker_id=0,
        total_workers=2,
        global_total_samples=64
    )
    print("✅ RolloutProgressMonitor 实例化成功")
    
    # 测试统计功能
    stats = monitor.get_stats()
    print(f"✅ 统计信息获取成功: {len(stats)} 个指标")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ 所有基础测试通过！")
EOF

echo ""
echo "3️⃣  检查配置参数..."
grep -A 2 "NUM_WORKERS=" examples/sglang_multiturn/search_browser/qwen3_agentloop.sh
echo ""

echo "=========================================="
echo "✅ 测试完成！可以开始训练了"
echo "=========================================="
echo ""
echo "运行命令:"
echo "  ./examples/sglang_multiturn/search_browser/qwen3_agentloop.sh"
echo ""
echo "预期输出:"
echo "  [Step 1] Worker 0/16 Rollout (Global: 256 samples): 0%|  | 0/16"
echo "  [Step 1] Worker 1/16 Rollout (Global: 256 samples): 0%|  | 0/16"
echo "  ..."
echo ""
