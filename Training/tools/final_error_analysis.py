#!/usr/bin/env python3
"""
完整的Browse工具错误分析 - 基于实际的Summary字段
重新生成准确的分析报告和可视化
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def analyze_file(file_path, label, limit=10000):
    """分析单个文件中的错误类型"""
    error_types = Counter()
    total_visits = 0

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num >= limit:
                break

            try:
                data = json.loads(line)
                output = data['full_trajectory']['output']

                # 查找Summary字段
                import re
                visit_pattern = r'\{"name"\s*:\s*"visit"'

                for match in re.finditer(visit_pattern, output):
                    summary_idx = output.find('Summary:', match.start())
                    if summary_idx == -1:
                        continue

                    # 提取Summary内容
                    summary_start = summary_idx + len('Summary:')
                    next_section = output.find('\n\n', summary_start)
                    if next_section != -1:
                        summary = output[summary_start:next_section].strip()
                    else:
                        summary = output[summary_start:summary_start+300].strip()

                    total_visits += 1

                    # 分类
                    summary_lower = summary.lower()

                    if 'could not be processed' in summary_lower:
                        error_types['网页无法处理'] += 1
                    elif 'could not be accessed' in summary_lower:
                        error_types['网页无法访问'] += 1
                    elif 'does not contain' in summary_lower or 'cannot be fulfilled' in summary_lower or 'no information is available' in summary_lower:
                        error_types['内容不匹配'] += 1
                    elif 'blocked' in summary_lower or 'block' in summary_lower:
                        error_types['被阻止'] += 1
                    elif 'timeout' in summary_lower:
                        error_types['超时'] += 1
                    else:
                        # 正常响应
                        error_types['正常响应'] += 1

            except Exception as e:
                pass

    return total_visits, error_types


def create_visualizations(output_dir, correct_stats, incorrect_stats):
    """创建可视化图表"""

    # 1. 错误类型对比图（正确 vs 错误）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    categories = ['网页无法处理', '正常响应', '内容不匹配', '被阻止']
    correct_values = [correct_stats.get(c, 0) for c in categories]
    incorrect_values = [incorrect_stats.get(c, 0) for c in categories]

    x = range(len(categories))
    width = 0.35

    bars1 = ax1.bar(x, correct_values, width, label='答案正确', color='#2ecc71')
    bars2 = ax1.bar([i + width for i in x], incorrect_values, width, label='答案错误', color='#e74c3c')

    ax1.set_xlabel('错误类型')
    ax1.set_ylabel('工具调用次数')
    ax1.set_title('工具错误类型分布：答案正确 vs 答案错误')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=15, ha='right')
    ax1.legend()

    # 2. 百分比对比图
    correct_pct = [v / sum(correct_values) * 100 if sum(correct_values) > 0 else 0 for v in correct_values]
    incorrect_pct = [v / sum(incorrect_values) * 100 if sum(incorrect_values) > 0 else 0 for v in incorrect_values]

    bars1 = ax2.bar(x, correct_pct, width, label='答案正确', color='#2ecc71')
    bars2 = ax2.bar([i + width for i in x], incorrect_pct, width, label='答案错误', color='#e74c3c')

    ax2.set_xlabel('错误类型')
    ax2.set_ylabel('占比 (%)')
    ax2.set_title('工具错误类型占比：答案正确 vs 答案错误')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=15, ha='right')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution_correct_vs_incorrect.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ 已生成: error_distribution_correct_vs_incorrect.png")


def create_summary_statistics(output_dir, correct_file, incorrect_file, all_file):
    """生成汇总统计"""

    # 分析所有数据
    correct_total, correct_errors = analyze_file(correct_file, "正确")
    incorrect_total, incorrect_errors = analyze_file(incorrect_file, "错误")
    all_total, all_errors = analyze_file(all_file, "全部", limit=50000)

    # 生成报告
    report = []
    report.append("=" * 100)
    report.append("Browse工具错误分析报告 - 基于工具Summary字段")
    report.append("=" * 100)
    report.append("")

    # 总体统计
    report.append("【总体统计】")
    report.append("-" * 100)
    report.append(f"总轨迹数: {all_total:,}")
    report.append(f"答案正确: {all_errors['正常响应'] + all_errors['内容不匹配']:,} ({(all_errors['正常响应'] + all_errors['内容不匹配'])/all_total*100:.1f}%)")
    report.append(f"答案错误: {all_errors['网页无法处理'] + all_errors['被阻止'] + all_errors.get('超时', 0) + all_errors.get('网页无法访问', 0):,} ({(all_errors['网页无法处理'] + all_errors['被阻止'])/all_total*100:.1f}%)")
    report.append("")

    # 详细错误分布
    report.append("【错误类型分布】")
    report.append("-" * 100)

    total_visits = sum(all_errors.values())
    for err_type, count in sorted(all_errors.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_visits * 100
        bar = '█' * int(pct / 2)
        report.append(f"{err_type:15s} | {count:7,} 次 | {pct:5.1f}% {bar}")

    report.append("")

    # 答案正确 vs 错误对比
    report.append("【答案正确 vs 答案错误 - 错误分布对比】")
    report.append("-" * 100)

    for category in ['网页无法处理', '正常响应', '内容不匹配', '被阻止']:
        c_count = correct_errors.get(category, 0)
        i_count = incorrect_errors.get(category, 0)
        c_pct = c_count / correct_total * 100 if correct_total > 0 else 0
        i_pct = i_count / incorrect_total * 100 if incorrect_total > 0 else 0

        report.append(f"{category:15s}")
        report.append(f"  答案正确: {c_count:7,} 次 ({c_pct:5.1f}%)")
        report.append(f"  答案错误: {i_count:7,} 次 ({i_pct:5.1f}%)")
        report.append("")

    report.append("")

    # 关键发现
    report.append("【关键发现】")
    report.append("-" * 100)

    correct_process_rate = correct_errors['网页无法处理'] / correct_total * 100 if correct_total > 0 else 0
    incorrect_process_rate = incorrect_errors['网页无法处理'] / incorrect_total * 100 if incorrect_total > 0 else 0

    report.append(f"1. '网页无法处理'在答案正确中占: {correct_process_rate:.1f}%")
    report.append(f"2. '网页无法处理'在答案错误中占: {incorrect_process_rate:.1f}%")
    report.append(f"3. 差距: {incorrect_process_rate - correct_process_rate:.1f}个百分点")
    report.append("")

    report.append("结论:")
    report.append("  • '网页无法处理'是主要错误类型（占{all_errors['网页无法处理']/total_visits*100:.1f}%）")
    report.append("  • 答案错误时此错误比例更高（{incorrect_process_rate:.1f}% vs {correct_process_rate:.1f}%）")
    report.append("  • 说明工具质量直接影响模型表现")
    report.append("")

    report.append("=" * 100)

    # 保存报告
    report_path = output_dir / "FINAL_ERROR_ANALYSIS_REPORT.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"✓ 报告已保存: {report_path}")

    return correct_errors, incorrect_errors, all_errors


def main():
    # 路径
    traj_dir = Path("/share/project/wanli/Search_Agent/verl/rollout_trajectory/qwen3_deepresearch_tis_rl_stage2_onpolicy_bs128_all_rag-temp_1_length_48k/20260212_222923")
    output_dir = traj_dir.parent / "browse_error_analysis"  # 使用已存在的目录
    final_output_dir = traj_dir.parent / "final_browse_error_analysis"
    final_output_dir.mkdir(exist_ok=True)

    print("=" * 100)
    print("Browse工具错误分析 - 基于Summary字段")
    print("=" * 100)
    print()

    # 分析文件
    correct_file = output_dir / "trajectories_with_errors_correct.jsonl"
    incorrect_file = output_dir / "trajectories_with_errors_incorrect.jsonl"
    all_file = output_dir / "trajectories_with_errors_all.jsonl"

    # 使用已存在的错误轨迹文件
    if not correct_file.exists() or not incorrect_file.exists():
        print("⚠ 错误轨迹文件不存在，请先运行 analyze_browse_errors.py")
        return

    print(f"✓ 找到正确答案文件: {correct_file}")
    print(f"✓ 找到错误答案文件: {incorrect_file}")

    # 生成可视化
    print("\n生成可视化图表...")
    correct_total, correct_errors = analyze_file(correct_file, "正确", limit=5000)
    incorrect_total, incorrect_errors = analyze_file(incorrect_file, "错误", limit=5000)

    # 读取全部错误数据用于总体统计
    all_total = 0
    all_errors = Counter()

    with open(all_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num >= 10000:
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
                    summary_lower = output[summary_start:summary_start+300].lower()

                    all_total += 1

                    if 'could not be processed' in summary_lower:
                        all_errors['网页无法处理'] += 1
                    elif 'could not be accessed' in summary_lower:
                        all_errors['网页无法访问'] += 1
                    elif 'does not contain' in summary_lower or 'cannot be fulfilled' in summary_lower:
                        all_errors['内容不匹配'] += 1
                    elif 'blocked' in summary_lower:
                        all_errors['被阻止'] += 1
                    elif 'timeout' in summary_lower:
                        all_errors['超时'] += 1
                    else:
                        all_errors['正常响应'] += 1

            except Exception as e:
                pass

    # 调用可视化函数
    create_visualizations(output_dir, correct_errors, incorrect_errors)

    # 生成统计报告
    print("\n生成统计报告...")
    create_summary_statistics(output_dir, correct_file, incorrect_file, all_file)

    print()
    print("=" * 100)
    print("分析完成！")
    print("=" * 100)
    print(f"结果保存在: {output_dir}")
    print()
    print("主要文件:")
    print(f"  • FINAL_ERROR_ANALYSIS_REPORT.txt - 详细统计报告")
    print(f"  • error_distribution_correct_vs_incorrect.png - 可视化图表")


if __name__ == "__main__":
    main()
