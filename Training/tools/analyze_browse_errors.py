#!/usr/bin/env python3
"""
Analyze rollout trajectories for browse tool errors.
Separates results by correct/incorrect answers and generates statistics.
"""
import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import re
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import pandas as pd

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from text using regex."""
    tool_calls = []
    # Match <invoke>...</invoke> blocks
    pattern = r'<invoke>\s*(\{.*?\})\s*</invoke>'
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        try:
            tool_call = json.loads(match.group(1))
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            pass
    return tool_calls


def parse_tool_responses(text: str) -> List[str]:
    """Parse tool responses from text."""
    # Match </invoke> blocks which contain tool responses
    pattern = r'</invoke>\s*(\{.*?\})\s*(?=<invoke>|$)'
    matches = re.finditer(pattern, text, re.DOTALL)
    responses = []
    for match in matches:
        try:
            response = json.loads(match.group(1))
            responses.append(response)
        except json.JSONDecodeError:
            pass
    return responses


def count_browse_errors_in_output(output: str) -> Dict[str, int]:
    """Count browse tool errors in an output string."""
    error_count = 0
    errors_by_type = defaultdict(int)
    errors_by_turn = []

    # Parse the output for tool calls and responses
    # The output contains multiple turns of assistant responses and tool responses

    # Split by assistant turns (marked by assistant responses)
    lines = output.split('\n')
    current_tool_call = None
    turn_count = 0

    for i, line in enumerate(lines):
        # Look for browse tool calls
        if 'visit' in str(line) or '"name": "visit"' in str(line):
            # Find the next few lines for the tool response
            context = '\n'.join(lines[i:min(i+50, len(lines))])

            # Check for common browse errors
            error_indicators = [
                ('timeout', 'Timeout'),
                ('failed to fetch', 'Fetch Failed'),
                ('connection error', 'Connection Error'),
                ('http error', 'HTTP Error'),
                ('404', 'Not Found'),
                ('403', 'Forbidden'),
                ('500', 'Server Error'),
                ('ssl', 'SSL Error'),
                ('dns', 'DNS Error'),
                ('blocked', 'Blocked'),
                ('rate limit', 'Rate Limited'),
                ('invalid url', 'Invalid URL'),
                ('unable to access', 'Access Denied'),
                ('cannot access', 'Access Denied'),
                ('failed to load', 'Load Failed'),
                ('error', 'Generic Error'),
            ]

            for error_pattern, error_name in error_indicators:
                if error_pattern in context.lower():
                    error_count += 1
                    errors_by_type[error_name] += 1
                    errors_by_turn.append(turn_count)
                    break

        # Count turns (assistant outputs)
        if line.strip().startswith('assistant') or '<answer>' in line:
            turn_count += 1

    return {
        'total_errors': error_count,
        'errors_by_type': dict(errors_by_type),
        'errors_by_turn': errors_by_turn
    }


def analyze_single_trajectory(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single trajectory for browse errors."""
    output = data.get('output', '')
    input_str = data.get('input', '')
    correct = data.get('correct', None)
    score = data.get('score', 0)
    pred_ans = data.get('pred_ans', '')
    step = data.get('step', 0)

    # Count browse errors
    error_info = count_browse_errors_in_output(output)

    # Count total tool calls
    tool_call_pattern = r'<invoke>\s*\{'
    total_tool_calls = len(re.findall(tool_call_pattern, output))

    # Count browse tool calls specifically
    browse_pattern = r'"name":\s*"visit"'
    total_browse_calls = len(re.findall(browse_pattern, output))

    # Count search tool calls
    search_pattern = r'"name":\s*"search"'
    total_search_calls = len(re.findall(search_pattern, output))

    # Count total turns
    assistant_turns = output.count('<answer>') + output.count('<invoke>')

    return {
        'index': data.get('global_index', 0),
        'step': step,
        'correct': correct,
        'score': score,
        'total_errors': error_info['total_errors'],
        'errors_by_type': error_info['errors_by_type'],
        'errors_by_turn': error_info['errors_by_turn'],
        'total_tool_calls': total_tool_calls,
        'total_browse_calls': total_browse_calls,
        'total_search_calls': total_search_calls,
        'assistant_turns': assistant_turns,
        'question': data.get('gts', {}).get('target', [''])[0] if isinstance(data.get('gts', {}).get('target'), list) else '',
        'pred_answer': pred_ans,
        'has_error': error_info['total_errors'] > 0
    }


def create_output_directory(base_path: Path) -> Path:
    """Create output directory for analysis results."""
    output_dir = base_path.parent / "browse_error_analysis"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def save_error_trajectories(output_dir: Path, with_errors: List[Dict], correct_only: bool = None):
    """Save trajectories with errors to separate files."""
    if correct_only is True:
        filename = "trajectories_with_errors_correct.jsonl"
    elif correct_only is False:
        filename = "trajectories_with_errors_incorrect.jsonl"
    else:
        filename = "trajectories_with_errors_all.jsonl"

    filepath = output_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        for traj in with_errors:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')

    print(f"Saved {len(with_errors)} trajectories to {filepath}")


def create_visualizations(output_dir: Path, all_results: List[Dict]):
    """Create visualization plots."""
    df = pd.DataFrame(all_results)

    # Separate by correctness
    df_correct = df[df['correct'] == True]
    df_incorrect = df[df['correct'] == False]

    # 1. Error distribution by correctness
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    error_counts = df['has_error'].value_counts()
    axes[0].pie([error_counts.get(False, 0), error_counts.get(True, 0)],
                labels=['No Errors', 'Has Errors'],
                colors=['#2ecc71', '#e74c3c'],
                autopct='%1.1f%%',
                startangle=90)
    axes[0].set_title('Distribution of Trajectories with Browse Errors')

    # Error count by correctness
    correct_with_errors = df_correct['has_error'].sum()
    incorrect_with_errors = df_incorrect['has_error'].sum()
    correct_total = len(df_correct)
    incorrect_total = len(df_incorrect)

    x = ['Correct', 'Incorrect']
    y1 = [correct_with_errors, incorrect_with_errors]
    y2 = [correct_total - correct_with_errors, incorrect_total - incorrect_with_errors]

    axes[1].bar(x, y2, label='No Errors', color='#2ecc71')
    axes[1].bar(x, y1, bottom=y2, label='Has Errors', color='#e74c3c')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Browse Errors by Answer Correctness')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Error types distribution
    all_error_types = defaultdict(int)
    for result in all_results:
        for error_type, count in result['errors_by_type'].items():
            all_error_types[error_type] += count

    if all_error_types:
        fig, ax = plt.subplots(figsize=(12, 6))
        error_types = list(all_error_types.keys())
        error_counts = list(all_error_types.values())

        # Sort by count
        sorted_data = sorted(zip(error_types, error_counts), key=lambda x: x[1], reverse=True)
        error_types, error_counts = zip(*sorted_data)

        bars = ax.barh(error_types, error_counts, color='#e74c3c')
        ax.set_xlabel('Count')
        ax.set_title('Browse Error Types Distribution')
        ax.invert_yaxis()

        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, error_counts)):
            ax.text(count + 0.5, i, str(count), va='center')

        plt.tight_layout()
        plt.savefig(output_dir / 'error_types.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Error count distribution
    df_with_errors = df[df['has_error'] == True]
    if not df_with_errors.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overall error count distribution
        error_counts = df_with_errors['total_errors'].values
        axes[0].hist(error_counts, bins=range(1, max(error_counts) + 2), edgecolor='black', color='#e74c3c')
        axes[0].set_xlabel('Number of Errors')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Error Counts per Trajectory')

        # Error count by correctness
        correct_errors = df_correct[df_correct['has_error']]['total_errors'].values
        incorrect_errors = df_incorrect[df_incorrect['has_error']]['total_errors'].values

        bins = range(1, max(max(error_counts), 1) + 2)
        axes[1].hist(correct_errors, bins=bins, alpha=0.5, label='Correct', color='#2ecc71', edgecolor='black')
        axes[1].hist(incorrect_errors, bins=bins, alpha=0.5, label='Incorrect', color='#e74c3c', edgecolor='black')
        axes[1].set_xlabel('Number of Errors')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Counts by Answer Correctness')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'error_count_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Browse call statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Browse calls by correctness
    avg_browse_correct = df_correct['total_browse_calls'].mean()
    avg_browse_incorrect = df_incorrect['total_browse_calls'].mean()

    axes[0].bar(['Correct', 'Incorrect'], [avg_browse_correct, avg_browse_incorrect],
                 color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[0].set_ylabel('Average Browse Calls')
    axes[0].set_title('Average Browse Tool Calls by Correctness')

    # Error rate (errors / browse calls)
    def safe_divide(a, b):
        return a / b if b > 0 else 0

    df['error_rate'] = df.apply(lambda row: safe_divide(row['total_errors'], row['total_browse_calls']), axis=1)
    df_correct['error_rate'] = df_correct.apply(lambda row: safe_divide(row['total_errors'], row['total_browse_calls']), axis=1)
    df_incorrect['error_rate'] = df_incorrect.apply(lambda row: safe_divide(row['total_errors'], row['total_browse_calls']), axis=1)

    avg_error_rate_correct = df_correct['error_rate'].mean()
    avg_error_rate_incorrect = df_incorrect['error_rate'].mean()

    axes[1].bar(['Correct', 'Incorrect'], [avg_error_rate_correct, avg_error_rate_incorrect],
                 color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[1].set_ylabel('Error Rate (Errors/Browse Calls)')
    axes[1].set_title('Browse Error Rate by Correctness')

    plt.tight_layout()
    plt.savefig(output_dir / 'browse_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Turn statistics
    fig, ax = plt.subplots(figsize=(10, 6))

    avg_turns_correct = df_correct['assistant_turns'].mean()
    avg_turns_incorrect = df_incorrect['assistant_turns'].mean()

    x = ['Correct', 'Incorrect']
    y = [avg_turns_correct, avg_turns_incorrect]

    bars = ax.bar(x, y, color=['#2ecc71', '#e74c3c'], edgecolor='black')
    ax.set_ylabel('Average Assistant Turns')
    ax.set_title('Average Number of Turns by Correctness')

    for bar, value in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'turn_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_statistics_report(output_dir: Path, all_results: List[Dict]):
    """Create a detailed statistics report."""
    df = pd.DataFrame(all_results)

    # Separate by correctness
    df_correct = df[df['correct'] == True]
    df_incorrect = df[df['correct'] == False]

    report = []
    report.append("=" * 80)
    report.append("BROWSE TOOL ERROR ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total trajectories: {len(df)}")
    report.append(f"Correct answers: {len(df_correct)} ({len(df_correct)/len(df)*100:.1f}%)")
    report.append(f"Incorrect answers: {len(df_incorrect)} ({len(df_incorrect)/len(df)*100:.1f}%)")
    report.append("")

    # Error statistics
    report.append("BROWSE ERROR STATISTICS")
    report.append("-" * 80)

    total_with_errors = df['has_error'].sum()
    total_without_errors = len(df) - total_with_errors

    report.append(f"Trajectories with browse errors: {total_with_errors} ({total_with_errors/len(df)*100:.1f}%)")
    report.append(f"Trajectories without browse errors: {total_without_errors} ({total_without_errors/len(df)*100:.1f}%)")
    report.append("")

    # Error breakdown by correctness
    correct_with_errors = df_correct['has_error'].sum()
    incorrect_with_errors = df_incorrect['has_error'].sum()

    report.append("ERRORS BY CORRECTNESS")
    report.append("-" * 80)
    report.append(f"Correct trajectories with errors: {correct_with_errors} ({correct_with_errors/len(df_correct)*100:.1f}% of correct)")
    report.append(f"Incorrect trajectories with errors: {incorrect_with_errors} ({incorrect_with_errors/len(df_incorrect)*100:.1f}% of incorrect)")
    report.append("")

    # Error count statistics
    df_with_errors = df[df['has_error'] == True]

    report.append("ERROR COUNT STATISTICS")
    report.append("-" * 80)

    for label, subset in [('All', df_with_errors), ('Correct', df_correct[df_correct['has_error']]), ('Incorrect', df_incorrect[df_incorrect['has_error']])]:
        if len(subset) > 0:
            report.append(f"\n{label}:")
            report.append(f"  Mean errors per trajectory: {subset['total_errors'].mean():.2f}")
            report.append(f"  Median errors per trajectory: {subset['total_errors'].median():.1f}")
            report.append(f"  Max errors in a trajectory: {subset['total_errors'].max()}")
            report.append(f"  Min errors in a trajectory: {subset['total_errors'].min()}")

    report.append("")

    # Error types
    all_error_types = defaultdict(int)
    for result in all_results:
        for error_type, count in result['errors_by_type'].items():
            all_error_types[error_type] += count

    if all_error_types:
        report.append("ERROR TYPES DISTRIBUTION")
        report.append("-" * 80)
        for error_type, count in sorted(all_error_types.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {error_type}: {count}")

    report.append("")

    # Tool call statistics
    report.append("TOOL CALL STATISTICS")
    report.append("-" * 80)

    for label, subset in [('All', df), ('Correct', df_correct), ('Incorrect', df_incorrect)]:
        report.append(f"\n{label}:")
        report.append(f"  Mean browse calls: {subset['total_browse_calls'].mean():.2f}")
        report.append(f"  Mean search calls: {subset['total_search_calls'].mean():.2f}")
        report.append(f"  Mean total tool calls: {subset['total_tool_calls'].mean():.2f}")

        if 'total_errors' in subset.columns and subset['total_browse_calls'].sum() > 0:
            error_rate = subset['total_errors'].sum() / subset['total_browse_calls'].sum() * 100
            report.append(f"  Overall error rate: {error_rate:.1f}%")

    report.append("")

    # Turn statistics
    report.append("TURN STATISTICS")
    report.append("-" * 80)

    for label, subset in [('All', df), ('Correct', df_correct), ('Incorrect', df_incorrect)]:
        report.append(f"\n{label}:")
        report.append(f"  Mean assistant turns: {subset['assistant_turns'].mean():.2f}")
        report.append(f"  Median assistant turns: {subset['assistant_turns'].median():.1f}")

    report.append("")
    report.append("=" * 80)

    # Write report to file
    report_path = output_dir / "statistics_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\nReport saved to {report_path}")

    # Also print to console
    print('\n'.join(report))


def analyze_trajectories(trajectory_dir: Path):
    """Main analysis function."""
    print(f"Analyzing trajectories in: {trajectory_dir}")

    # Find all JSONL files
    jsonl_files = sorted(trajectory_dir.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")

    if not jsonl_files:
        print("No JSONL files found!")
        return

    # Create output directory
    output_dir = create_output_directory(trajectory_dir)
    print(f"Output directory: {output_dir}")

    # Analyze all trajectories
    all_results = []
    trajectories_with_errors = []
    trajectories_with_errors_correct = []
    trajectories_with_errors_incorrect = []

    for jsonl_file in jsonl_files:
        print(f"\nProcessing {jsonl_file.name}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        data = json.loads(line)
                        result = analyze_single_trajectory(data)
                        all_results.append(result)

                        # Save full trajectory data if it has errors
                        if result['has_error']:
                            result['full_trajectory'] = data
                            trajectories_with_errors.append(result)

                            if result['correct']:
                                trajectories_with_errors_correct.append(result)
                            elif result['correct'] is False:
                                trajectories_with_errors_incorrect.append(result)

                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num + 1}: {e}")

    print(f"\nAnalyzed {len(all_results)} trajectories total")

    # Generate statistics
    create_statistics_report(output_dir, all_results)

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(output_dir, all_results)

    # Save trajectories with errors
    print("\nSaving trajectories with errors...")
    save_error_trajectories(output_dir, trajectories_with_errors)
    save_error_trajectories(output_dir, trajectories_with_errors_correct, correct_only=True)
    save_error_trajectories(output_dir, trajectories_with_errors_incorrect, correct_only=False)

    # Save detailed CSV
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "analysis_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to {csv_path}")

    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    trajectory_dir = Path("/share/project/wanli/Search_Agent/verl/rollout_trajectory/qwen3_deepresearch_tis_rl_stage2_onpolicy_bs128_all_rag-temp_1_length_48k/20260212_222923")

    if not trajectory_dir.exists():
        print(f"Error: Directory {trajectory_dir} does not exist")
        exit(1)

    analyze_trajectories(trajectory_dir)
