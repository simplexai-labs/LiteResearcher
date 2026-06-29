#!/usr/bin/env python3
"""
分析 JSONL 轨迹文件的统计脚本
- 统计正确率
- 统计使用LLM judge的比例
- 按method分类统计
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List


def safe_load_jsonl(line: str) -> Dict[str, Any]:
    """安全地加载JSONL行，处理JavaScript风格的true/false/null"""
    # 将JavaScript风格的值替换为 Python 风格
    line = line.replace(': true', ': True')
    line = line.replace(': false', ': False')
    line = line.replace(':true', ':True')
    line = line.replace(':false', ':False')
    line = line.replace(': null', ': None')
    line = line.replace(':null', ':None')
    
    try:
        # 先尝试标准JSON解析
        return json.loads(line.replace('True', 'true').replace('False', 'false').replace('None', 'null'))
    except json.JSONDecodeError:
        # 如果失败，使用eval（仅用于修复true/false/null问题）
        try:
            return eval(line)
        except Exception as e:
            print(f"⚠️  警告: 无法解析行: {e}")
            return {}


def analyze_jsonl(jsonl_path: Path) -> None:
    """分析JSONL文件"""
    
    if not jsonl_path.exists():
        print(f"❌ 错误: 文件不存在: {jsonl_path}")
        return
    
    print(f"📂 正在分析文件: {jsonl_path}")
    print("=" * 80)
    
    # 统计数据
    total = 0
    correct_count = 0
    incorrect_count = 0
    unknown_count = 0
    
    # 按method统计
    method_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'incorrect': 0,
        'unknown': 0
    })
    
    # LLM Judge相关统计
    llm_judge_count = 0
    llm_judge_correct_count = 0
    llm_judge_methods = set()
    
    # 收集所有数据
    data_list: List[Dict[str, Any]] = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = safe_load_jsonl(line)
                if not data:
                    continue
                
                data_list.append(data)
                total += 1
                
                # 基本统计
                correct = data.get('correct')
                method = data.get('method', 'unknown')
                raw_response = data.get('raw_response')
                
                # 统计正确性
                if correct is True:
                    correct_count += 1
                    method_stats[method]['correct'] += 1
                elif correct is False:
                    incorrect_count += 1
                    method_stats[method]['incorrect'] += 1
                else:
                    unknown_count += 1
                    method_stats[method]['unknown'] += 1
                
                method_stats[method]['total'] += 1
                
                # 检查是否使用LLM judge（有raw_response字段且不为空）
                if raw_response and raw_response.strip():
                    llm_judge_count += 1
                    llm_judge_methods.add(method)
                    if correct is True:
                        llm_judge_correct_count += 1
                
            except Exception as e:
                print(f"⚠️  警告: 第{line_num}行解析失败: {e}")
                continue
    
    # 打印统计结果
    print(f"\n📊 总体统计")
    print("-" * 80)
    print(f"总样本数: {total}")
    print(f"正确数量: {correct_count} ({correct_count/total*100:.2f}%)" if total > 0 else "正确数量: 0")
    print(f"错误数量: {incorrect_count} ({incorrect_count/total*100:.2f}%)" if total > 0 else "错误数量: 0")
    print(f"未知数量: {unknown_count} ({unknown_count/total*100:.2f}%)" if total > 0 else "未知数量: 0")
    
    print(f"\n🤖 LLM Judge 使用情况")
    print("-" * 80)
    print(f"使用LLM Judge的样本数: {llm_judge_count} ({llm_judge_count/total*100:.2f}%)" if total > 0 else "使用LLM Judge的样本数: 0")
    print(f"LLM Judge正确数: {llm_judge_correct_count} ({llm_judge_correct_count/llm_judge_count*100:.2f}%)" if llm_judge_count > 0 else "LLM Judge正确数: 0")
    print(f"使用LLM Judge的methods: {', '.join(sorted(llm_judge_methods)) if llm_judge_methods else '无'}")
    
    print(f"\n📋 按Method分类统计")
    print("-" * 80)
    print(f"{'Method':<20} {'总数':<8} {'正确':<8} {'错误':<8} {'未知':<8} {'正确率':<10}")
    print("-" * 80)
    
    for method in sorted(method_stats.keys()):
        stats = method_stats[method]
        total_m = stats['total']
        correct_m = stats['correct']
        incorrect_m = stats['incorrect']
        unknown_m = stats['unknown']
        acc_m = correct_m / total_m * 100 if total_m > 0 else 0
        
        print(f"{method:<20} {total_m:<8} {correct_m:<8} {incorrect_m:<8} {unknown_m:<8} {acc_m:<10.2f}%")
    
    # 详细分析
    if data_list:
        print(f"\n📈 详细分析")
        print("-" * 80)
        
        # 平均分数
        scores = [d.get('score', 0) for d in data_list]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"平均分数: {avg_score:.4f}")
        
        # 平均轮数
        assistant_turns = [d.get('assistant_turns', 0) for d in data_list if d.get('assistant_turns')]
        user_turns = [d.get('user_turns', 0) for d in data_list if d.get('user_turns')]
        
        if assistant_turns:
            avg_assistant = sum(assistant_turns) / len(assistant_turns)
            print(f"平均Assistant轮数: {avg_assistant:.2f}")
        
        if user_turns:
            avg_user = sum(user_turns) / len(user_turns)
            print(f"平均User轮数: {avg_user:.2f}")
        
        # 数据来源统计
        sources = defaultdict(int)
        for d in data_list:
            source = d.get('data_source', 'unknown')
            sources[source] += 1
        
        if sources:
            print(f"\n数据来源分布:")
            for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
                print(f"  {source}: {count} ({count/total*100:.2f}%)")
    
    print("\n" + "=" * 80)
    print("✅ 分析完成!")


def main():
    parser = argparse.ArgumentParser(
        description='分析JSONL轨迹文件的统计信息',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python analyze_jsonl.py rollout_trajectory/xxx/4.jsonl
        """
    )
    
    parser.add_argument('jsonl_file', help='输入的JSONL文件路径')
    
    args = parser.parse_args()
    
    jsonl_path = Path(args.jsonl_file)
    analyze_jsonl(jsonl_path)


if __name__ == "__main__":
    main()

