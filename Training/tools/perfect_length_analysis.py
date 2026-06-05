#!/usr/bin/env python3
"""工具完全正常的轨迹长度分析"""
import json
from pathlib import Path

correct_file = Path("/share/project/wanli/Search_Agent/verl/rollout_trajectory/qwen3_deepresearch_tis_rl_stage2_onpolicy_bs128_all_rag-temp_1_length_48k/browse_error_analysis/trajectories_with_errors_correct.jsonl")

lengths = []
turns = []
browses = []

with open(correct_file, 'r') as f:
    for line_num, line in enumerate(f):
        if line_num >= 5000:
            break

        try:
            data = json.loads(line)
            output = data['full_trajectory']['output']

            lengths.append(len(output))

            import re
            browses = len(re.findall(r'\{"name"\s*:\s*"visit"', output))
            assis = len(re.findall(r'\nassistant', output))

            turns.append(assis)
            browses.append(browses)

        except Exception as e:
            pass

print("=" * 100)
print("工具完全正常的轨迹长度分布详情")
print("=" * 100)
print(f"样本数: {len(lengths)}")
print()

# 统计分布
bins = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 120000, 140000]

print("长度区间 | 数量 | 占比 | 累计占比")
print("-" * 100)

cumsum = 0
for i in range(len(bins)-1):
    start = bins[i]
    end = bins[i+1]
    count = sum(1 for l in lengths if start <= l < end)
    pct = count / len(lengths) * 100
    cumsum += pct

    bar_width = max(1, int(pct / 2))
    bar = '█' * bar_width
    print(f"{start//1000:5.0f}k-{end//1000:5.0f}k | {count:5,} | {pct:4.1f}% | {bar} {cumsum:4.1f}%")

print()
print("统计摘要:")
mean_length = sum(lengths) / len(lengths)
median_length = sorted(lengths)[len(lengths)//2]
print(f"  平均长度: {mean_length:.0f} 字符")
print(f"  中位数: {median_length:.0f} 字符")
print(f"  最小值: {min(lengths):.0f} 字符")
print(f"  最大值: {max(lengths):.0f} 字符")

# 计算平均轮次和平均浏览（从lengths重新统计）
total_turns = 0
total_browses = 0

for line_num, line in enumerate(open(correct_file)):
    if line_num >= 5000:
        break

    try:
        data = json.loads(line)
        output = data['full_trajectory']['output']

        total_turns += len(re.findall(r'\nassistant', output))
        total_browses += len(re.findall(r'\{"name"\s*:\s*"visit"', output))
    except:
        pass

avg_turns = total_turns / len(lengths)
avg_browses = total_browses / len(lengths)

print(f"  平均轮次: {avg_turns:.1f}")
print(f"  平均浏览: {avg_browses:.1f}")
