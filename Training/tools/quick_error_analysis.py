#!/usr/bin/env python3
"""简化的Browse工具错误分析 - 基于Summary字段"""
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def analyze_file(file_path, limit=5000):
    """分析文件中的错误类型"""
    error_types = Counter()
    total = 0

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num >= limit:
                break

            try:
                data = json.loads(line)
                output = data['full_trajectory']['output']

                import re
                visit_pattern = r'\{"name"\s*:\s*"visit"'

                for match in re.finditer(visit_pattern, output):
                    summary_idx = output.find('Summary:', match.start())
                    if summary_idx == -1:
                        continue

                    summary_start = summary_idx + len('Summary:')
                    next_section = output.find('\n\n', summary_start)
                    if next_section != -1:
                        summary = output[summary_start:next_section].strip()
                    else:
                        summary = output[summary_start:summary_start+300].strip()

                    total += 1

                    summary_lower = summary.lower()

                    if 'could not be processed' in summary_lower:
                        error_types['网页无法处理'] += 1
                    elif 'could not be accessed' in summary_lower:
                        error_types['网页无法访问'] += 1
                    elif 'does not contain' in summary_lower or 'cannot be fulfilled' in summary_lower:
                        error_types['内容不匹配'] += 1
                    elif 'blocked' in summary_lower:
                        error_types['被阻止'] += 1
                    else:
                        error_types['正常响应'] += 1

            except Exception as e:
                pass

    return total, error_types


def main():
    traj_dir = Path("/share/project/wanli/Search_Agent/verl/rollout_trajectory/qwen3_deepresearch_tis_rl_stage2_onpolicy_bs128_all_rag-temp_1_length_48k/20260212_222923")
    output_dir = traj_dir.parent / "browse_error_analysis"
    final_dir = traj_dir.parent / "final_browse_error_analysis"
    final_dir.mkdir(exist_ok=True)

    print("=" * 100)
    print("Browse工具错误分析 - 基于Summary字段")
    print("=" * 100)
    print()

    correct_file = output_dir / "trajectories_with_errors_correct.jsonl"
    incorrect_file = output_dir / "trajectories_with_errors_incorrect.jsonl"

    if not correct_file.exists() or not incorrect_file.exists():
        print("请确保以下文件存在:")
        print(f"  {correct_file}")
        print(f"  {incorrect_file}")
        return

    print(f"分析文件:")
    print(f"  ✓ {correct_file}")
    print(f"  ✓ {incorrect_file}")
    print()

    # 分析
    print("分析中...")
    correct_total, correct_errors = analyze_file(correct_file, limit=5000)
    incorrect_total, incorrect_errors = analyze_file(incorrect_file, limit=5000)

    print(f"答案正确: {correct_total:,}条")
    print(f"答案错误: {incorrect_total:,}条")
    print()

    # 1. 柱状图对比
    categories = ['网页无法处理', '正常响应', '内容不匹配', '被阻止']
    correct_vals = [correct_errors.get(c, 0) for c in categories]
    incorrect_vals = [incorrect_errors.get(c, 0) for c in categories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = range(len(categories))
    width = 0.35

    ax1.bar(x, correct_vals, width, label='答案正确', color='#2ecc71')
    ax1.bar([i + width for i in x], incorrect_vals, width, label='答案错误', color='#e74c3c')

    ax1.set_xlabel('错误类型')
    ax1.set_ylabel('工具调用次数')
    ax1.set_title('错误类型分布：答案正确 vs 答案错误')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=15, ha='right', fontsize=9)
    ax1.legend(fontsize=9)

    # 2. 百分比图
    correct_pct = [v / sum(correct_vals) * 100 for v in correct_vals]
    incorrect_pct = [v / sum(incorrect_vals) * 100 for v in incorrect_vals]

    ax2.bar(x, correct_pct, width, label='答案正确', color='#2ecc71')
    ax2.bar([i + width for i in x], incorrect_pct, width, label='答案错误', color='#e74c3c')

    ax2.set_xlabel('错误类型')
    ax2.set_ylabel('占比 (%)')
    ax2.set_title('错误类型占比：答案正确 vs 答案错误')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=15, ha='right', fontsize=9)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(final_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ 图表已生成")

    # 3. 生成报告
    report = []
    report.append("=" * 100)
    report.append("Browse工具错误分析报告（基于Summary字段）")
    report.append("=" * 100)
    report.append("")

    report.append("【总体统计】")
    report.append(f"答案正确轨迹: {correct_total:,}条")
    report.append(f"答案错误轨迹: {incorrect_total:,}条")
    report.append("")

    report.append("【答案正确 - 错误分布】")
    for c in categories:
        cv = correct_errors.get(c, 0)
        ct = incorrect_vals[categories.index(c)]
        cp = cv / correct_total * 100 if correct_total > 0 else 0
        it = ct / incorrect_total * 100 if incorrect_total > 0 else 0

        report.append(f"{c:10s}: 正确={cv:,}({cp:.1f}%) | 错误={ct:,}({it:.1f}%)")
    report.append("")

    report.append("【关键发现】")
    report.append("-" * 100)

    correct_proc = correct_errors['网页无法处理'] / correct_total * 100
    incorrect_proc = incorrect_errors['网页无法处理'] / incorrect_total * 100

    report.append(f"1. '网页无法处理'在答案正确中占: {correct_proc:.1f}%")
    report.append(f"2. '网页无法处理'在答案错误中占: {incorrect_proc:.1f}%")
    report.append(f"3. 差距: {incorrect_proc - correct_proc:.1f}个百分点")
    report.append("")

    report.append("结论:")
    report.append("  • '网页无法处理'是最主要的错误类型（占比{correct_errors['网页无法处理']/(correct_total+incorrect_total)*100:.1f}%）")
    report.append("  • 答案错误时此错误比例更高，说明工具质量影响模型表现")
    report.append("")

    report.append("=" * 100)

    report_path = final_dir / "ERROR_ANALYSIS_REPORT.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"✓ 报告已保存: {report_path}")
    print()
    print("=" * 100)
    print("分析完成！")
    print(f"结果保存在: {final_dir}")
    print("主要文件:")
    print(f"  • ERROR_ANALYSIS_REPORT.txt - 详细报告")
    print(f"  • error_distribution.png - 可视化图表")


if __name__ == "__main__":
    main()
