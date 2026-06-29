#!/usr/bin/env python3
"""
Trajectory Visualization Script
可视化Agent Loop的rollout轨迹，按轮次(turn)展示，并保存不重复的数据
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import OrderedDict
import argparse
from datetime import datetime


class TrajectoryVisualizer:
    """轨迹可视化器"""

    def __init__(self, jsonl_file: str):
        self.jsonl_file = jsonl_file
        self.entries = []

    def load_data(self, max_entries: int = None) -> List[Dict[str, Any]]:
        """加载JSONL数据"""
        print(f"📖 加载文件: {self.jsonl_file}")

        entries = []
        seen_outputs = set()  # 用于去重

        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)

                    # 基于output去重
                    output_hash = hash(data.get('output', ''))
                    if output_hash not in seen_outputs:
                        seen_outputs.add(output_hash)
                        entries.append(data)

                        if max_entries and len(entries) >= max_entries:
                            break

                except json.JSONDecodeError as e:
                    print(f"⚠️  警告: 第{i}行JSON解析失败: {e}")
                    continue

        print(f"✅ 加载了 {len(entries)} 条不重复数据 (总共 {i} 行)")
        self.entries = entries
        return entries

    def parse_turns(self, output: str) -> List[Dict[str, Any]]:
        """解析output为多个轮次"""
        turns = []

        # 方法1: 基于 "assistant\n" 分割
        parts = re.split(r'(assistant\n)', output)

        current_turn = []
        for i, part in enumerate(parts):
            if part == "assistant\n":
                if current_turn:
                    # 保存上一个turn
                    turn_text = ''.join(current_turn)
                    turns.append(self._analyze_turn(turn_text, len(turns) + 1))
                current_turn = [part]
            else:
                current_turn.append(part)

        # 保存最后一个turn
        if current_turn:
            turn_text = ''.join(current_turn)
            turns.append(self._analyze_turn(turn_text, len(turns) + 1))

        return turns

    def _analyze_turn(self, text: str, turn_num: int) -> Dict[str, Any]:
        """分析单个轮次的内容"""
        turn_data = {
            'turn_number': turn_num,
            'raw_text': text,
            'has_think': '<think>' in text,
            'has_tool_call': '<tool_call>' in text,
            'has_tool_response': '<tool_response>' in text,
            'has_answer': '<answer>' in text,
        }

        # 提取think内容
        if turn_data['has_think']:
            think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
            if think_match:
                turn_data['think_content'] = think_match.group(1).strip()

        # 提取tool_call内容
        if turn_data['has_tool_call']:
            tool_calls = []
            for match in re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
                tool_call_text = match.group(1).strip()
                try:
                    tool_call_json = json.loads(tool_call_text)
                    tool_calls.append(tool_call_json)
                except json.JSONDecodeError:
                    tool_calls.append({'raw': tool_call_text, 'error': 'Invalid JSON'})
            turn_data['tool_calls'] = tool_calls

        # 提取tool_response内容
        if turn_data['has_tool_response']:
            tool_responses = []
            for match in re.finditer(r'<tool_response>(.*?)</tool_response>', text, re.DOTALL):
                response_text = match.group(1).strip()
                tool_responses.append(response_text)
            turn_data['tool_responses'] = tool_responses

        # 提取answer内容
        if turn_data['has_answer']:
            answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
            if answer_match:
                turn_data['answer_content'] = answer_match.group(1).strip()

        return turn_data

    def visualize_entry(self, entry: Dict[str, Any], entry_idx: int) -> str:
        """可视化单条轨迹"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"📊 Entry #{entry_idx}")
        lines.append("=" * 80)

        # 定义需要特殊处理的字段（不在基本信息中显示）
        special_fields = {'input', 'output', 'gts'}

        # 定义基本信息字段的显示顺序和名称映射
        basic_field_mapping = {
            'step': 'Step',
            'score': 'Score',
            'data_source': 'Data Source',
            'assistant_turns': 'Assistant Turns',
            'user_turns': 'User Turns',
            'pred_ans': 'Predicted Answer',
            'reward': 'Reward',
        }

        # 显示基本信息（按优先级顺序）
        for field, display_name in basic_field_mapping.items():
            if field in entry:
                value = entry[field]
                if value is not None:
                    lines.append(f"{display_name}: {value}")

        # 显示其他额外字段（动态）
        displayed_fields = set(basic_field_mapping.keys()) | special_fields
        extra_fields = {k: v for k, v in entry.items() if k not in displayed_fields}
        if extra_fields:
            lines.append("\n📋 Additional Fields:")
            for key, value in sorted(extra_fields.items()):
                # 限制值的长度以保持可读性
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                lines.append(f"  - {key}: {value_str}")

        # 显示ground truth
        if 'gts' in entry:
            lines.append(f"\nGround Truth: {json.dumps(entry['gts'], ensure_ascii=False)}")

        # 显示input (简化版)
        if 'input' in entry:
            input_text = entry['input']
            # 提取用户问题
            user_match = re.search(r'user\n(.+?)(?:\n|$)', input_text, re.DOTALL)
            if user_match:
                question = user_match.group(1).strip()
                lines.append(f"Question: {question[:200]}...")

        # 解析和显示每一轮
        output = entry.get('output', '')
        turns = self.parse_turns(output)

        lines.append(f"\n🔄 Total Turns: {len(turns)}")
        lines.append("-" * 80)

        for turn in turns:
            lines.append(f"\n🔹 Turn {turn['turn_number']}")
            lines.append("-" * 40)

            # Think部分
            if turn['has_think']:
                think = turn.get('think_content', '')
                lines.append(f"💭 Think: {think[:150]}..." if len(think) > 150 else f"💭 Think: {think}")

            # Tool Call部分
            if turn['has_tool_call']:
                for i, tool_call in enumerate(turn.get('tool_calls', []), 1):
                    if 'error' in tool_call:
                        lines.append(f"🔧 Tool Call {i}: ❌ {tool_call['raw'][:100]}")
                    else:
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('arguments', {})
                        lines.append(f"🔧 Tool Call {i}: {tool_name}")
                        for key, value in tool_args.items():
                            value_str = str(value)[:100]
                            lines.append(f"   - {key}: {value_str}...")

            # Tool Response部分
            if turn['has_tool_response']:
                for i, response in enumerate(turn.get('tool_responses', []), 1):
                    response_preview = response[:200] if len(response) > 200 else response
                    lines.append(f"📥 Tool Response {i}: {response_preview}...")

            # Answer部分
            if turn['has_answer']:
                answer = turn.get('answer_content', '')
                lines.append(f"✅ Answer: {answer}")

        lines.append("\n")
        return '\n'.join(lines)

    def save_structured_data(self, output_file: str, max_entries: int = 200):
        """保存结构化的数据到JSON文件"""
        print(f"\n💾 保存结构化数据到: {output_file}")

        structured_data = []

        for i, entry in enumerate(self.entries[:max_entries], 1):
            output = entry.get('output', '')
            turns = self.parse_turns(output)

            # 基本结构
            structured_entry = {
                'entry_id': i,
                'turns': turns,
                'statistics': {
                    'total_turns': len(turns),
                    'has_tool_calls': any(t.get('has_tool_call') for t in turns),
                    'total_tool_calls': sum(len(t.get('tool_calls', [])) for t in turns),
                    'has_answer': any(t.get('has_answer') for t in turns),
                }
            }

            # 动态添加所有其他字段（除了 output，因为已经解析为 turns）
            for key, value in entry.items():
                if key not in ['output']:  # output 已经被解析为 turns
                    structured_entry[key] = value

            structured_data.append(structured_entry)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存 {len(structured_data)} 条结构化数据")

        # 显示统计信息
        total_turns = sum(e['statistics']['total_turns'] for e in structured_data)
        total_tool_calls = sum(e['statistics']['total_tool_calls'] for e in structured_data)
        entries_with_answer = sum(1 for e in structured_data if e['statistics']['has_answer'])

        print(f"\n📈 统计信息:")
        print(f"  总条目: {len(structured_data)}")
        print(f"  总轮次: {total_turns}")
        print(f"  总工具调用: {total_tool_calls}")
        print(f"  有答案的条目: {entries_with_answer}")
        print(f"  平均每条轮次: {total_turns / len(structured_data):.2f}")

    def generate_html_report(self, output_file: str, max_entries: int = 200):
        """生成HTML可视化报告"""
        print(f"\n🌐 生成HTML报告: {output_file}")

        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <meta charset='utf-8'>",
            "    <title>Trajectory Visualization Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
            "        .entry { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            "        .entry-header { background: #4CAF50; color: white; padding: 10px; border-radius: 4px; margin-bottom: 15px; }",
            "        .turn { margin: 15px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #2196F3; }",
            "        .turn-header { font-weight: bold; color: #2196F3; margin-bottom: 10px; }",
            "        .think { background: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 4px; }",
            "        .tool-call { background: #d1ecf1; padding: 10px; margin: 10px 0; border-radius: 4px; }",
            "        .tool-response { background: #d4edda; padding: 10px; margin: 10px 0; border-radius: 4px; }",
            "        .answer { background: #c3e6cb; padding: 10px; margin: 10px 0; border-radius: 4px; font-weight: bold; }",
            "        .stats { background: #e7f3ff; padding: 10px; margin: 10px 0; border-radius: 4px; }",
            "        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }",
            "        .error { color: red; }",
            "    </style>",
            "</head>",
            "<body>",
            f"    <h1>🔍 Trajectory Visualization Report</h1>",
            f"    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"    <p>Source: {os.path.basename(self.jsonl_file)}</p>",
            f"    <p>Total Entries: {len(self.entries[:max_entries])}</p>",
            "    <hr>",
        ]

        for i, entry in enumerate(self.entries[:max_entries], 1):
            output = entry.get('output', '')
            turns = self.parse_turns(output)

            html_lines.append(f"    <div class='entry' id='entry-{i}'>")
            html_lines.append(f"        <div class='entry-header'>")
            html_lines.append(f"            <h2>📊 Entry #{i}</h2>")

            # 动态显示所有基本字段
            header_parts = []
            if 'step' in entry:
                header_parts.append(f"Step: {entry['step']}")
            if 'score' in entry:
                header_parts.append(f"Score: {entry['score']}")
            if 'data_source' in entry:
                header_parts.append(f"Data Source: <strong>{entry['data_source']}</strong>")

            if header_parts:
                html_lines.append(f"            {' | '.join(header_parts)}")
            html_lines.append(f"        </div>")

            # Statistics and additional fields
            html_lines.append(f"        <div class='stats'>")
            html_lines.append(f"            <strong>📈 Statistics:</strong>")
            html_lines.append(f"            Total Turns: {len(turns)} | ")
            html_lines.append(f"            Tool Calls: {sum(len(t.get('tool_calls', [])) for t in turns)} | ")
            html_lines.append(f"            Has Answer: {'✅' if any(t.get('has_answer') for t in turns) else '❌'}")

            # 动态添加其他字段
            special_fields = {'input', 'output', 'gts', 'step', 'score', 'data_source'}
            additional_info = []

            # 优先显示常见字段
            priority_fields = ['assistant_turns', 'user_turns', 'pred_ans', 'reward']
            for field in priority_fields:
                if field in entry and entry[field] is not None:
                    display_name = field.replace('_', ' ').title()
                    if field == 'pred_ans':
                        display_name = 'Predicted Answer'
                    additional_info.append(f"{display_name}: {entry[field]}")

            # 显示其他额外字段
            for key, value in sorted(entry.items()):
                if key not in special_fields and key not in priority_fields and value is not None:
                    display_name = key.replace('_', ' ').title()
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:50] + "..."
                    additional_info.append(f"{display_name}: {value_str}")

            if additional_info:
                html_lines.append(f"<br>{' | '.join(additional_info)}")

            html_lines.append(f"        </div>")

            # Turns
            for turn in turns:
                html_lines.append(f"        <div class='turn'>")
                html_lines.append(f"            <div class='turn-header'>🔹 Turn {turn['turn_number']}</div>")

                if turn['has_think']:
                    think = turn.get('think_content', '')
                    html_lines.append(f"            <div class='think'>💭 <strong>Think:</strong><br>{self._html_escape(think[:300])}</div>")

                if turn['has_tool_call']:
                    for tool_call in turn.get('tool_calls', []):
                        if 'error' in tool_call:
                            html_lines.append(f"            <div class='tool-call error'>🔧 <strong>Tool Call:</strong> Error parsing JSON</div>")
                        else:
                            tool_name = tool_call.get('name', 'unknown')
                            html_lines.append(f"            <div class='tool-call'>🔧 <strong>Tool Call:</strong> <code>{tool_name}</code><br>")
                            html_lines.append(f"                Arguments: <code>{json.dumps(tool_call.get('arguments', {}), ensure_ascii=False)[:200]}</code>")
                            html_lines.append(f"            </div>")

                if turn['has_tool_response']:
                    for response in turn.get('tool_responses', []):
                        html_lines.append(f"            <div class='tool-response'>📥 <strong>Tool Response:</strong><br>{self._html_escape(response[:300])}</div>")

                if turn['has_answer']:
                    answer = turn.get('answer_content', '')
                    html_lines.append(f"            <div class='answer'>✅ <strong>Answer:</strong> {self._html_escape(answer)}</div>")

                html_lines.append(f"        </div>")

            html_lines.append(f"    </div>")

        html_lines.extend([
            "</body>",
            "</html>"
        ])

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_lines))

        print(f"✅ HTML报告已生成")

    def _html_escape(self, text: str) -> str:
        """HTML转义"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;')
                .replace('\n', '<br>'))


def main():
    parser = argparse.ArgumentParser(
        description='可视化Agent Loop轨迹数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python visualize_trajectory.py rollout_trajectory/xxx/1.jsonl

  # 指定输出目录和最大条目数
  python visualize_trajectory.py rollout_trajectory/xxx/1.jsonl -o output/ -n 100

  # 只生成JSON，不生成HTML
  python visualize_trajectory.py rollout_trajectory/xxx/1.jsonl --no-html
        """
    )

    parser.add_argument('jsonl_file', help='输入的JSONL文件路径')
    parser.add_argument('-o', '--output-dir', default='./trajectory_visualization', help='输出目录 (默认: ./trajectory_visualization)')
    parser.add_argument('-n', '--max-entries', type=int, default=200, help='最大处理条目数 (默认: 200)')
    parser.add_argument('--no-html', action='store_true', help='不生成HTML报告')
    parser.add_argument('--no-json', action='store_true', help='不生成JSON文件')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细输出到终端')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.jsonl_file):
        print(f"❌ 错误: 文件不存在: {args.jsonl_file}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 生成输出文件名
    base_name = Path(args.jsonl_file).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_output = os.path.join(args.output_dir, f"{base_name}_structured_{timestamp}.json")
    html_output = os.path.join(args.output_dir, f"{base_name}_report_{timestamp}.html")
    txt_output = os.path.join(args.output_dir, f"{base_name}_text_{timestamp}.txt")

    print("=" * 80)
    print("🚀 Trajectory Visualization Tool")
    print("=" * 80)

    # 创建可视化器
    visualizer = TrajectoryVisualizer(args.jsonl_file)

    # 加载数据
    entries = visualizer.load_data(max_entries=args.max_entries)

    if not entries:
        print("❌ 没有加载到任何数据")
        sys.exit(1)

    # 保存JSON
    if not args.no_json:
        visualizer.save_structured_data(json_output, max_entries=args.max_entries)

    # 生成HTML报告
    if not args.no_html:
        visualizer.generate_html_report(html_output, max_entries=args.max_entries)

    # 生成文本报告
    if args.verbose:
        print("\n" + "=" * 80)
        print("📄 文本可视化 (前5条)")
        print("=" * 80)

        with open(txt_output, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(entries[:min(5, len(entries))], 1):
                text = visualizer.visualize_entry(entry, i)
                print(text)
                f.write(text + '\n\n')

            # 写入剩余的到文件
            for i, entry in enumerate(entries[5:args.max_entries], 6):
                text = visualizer.visualize_entry(entry, i)
                f.write(text + '\n\n')

        print(f"\n💾 完整文本报告已保存到: {txt_output}")

    print("\n" + "=" * 80)
    print("✅ 完成!")
    print("=" * 80)
    print(f"输出文件:")
    if not args.no_json:
        print(f"  📄 JSON: {json_output}")
    if not args.no_html:
        print(f"  🌐 HTML: {html_output}")
    if args.verbose:
        print(f"  📝 TXT:  {txt_output}")
    print(f"\n打开HTML报告查看:")
    print(f"  xdg-open {html_output}  # Linux")
    print(f"  open {html_output}      # macOS")
    print("=" * 80)


if __name__ == '__main__':
    main()
