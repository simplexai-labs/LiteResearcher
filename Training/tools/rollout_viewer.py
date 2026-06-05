#!/usr/bin/env python3
"""
Rollout JSONL 可视化查看器
用法: python rollout_viewer.py <jsonl_path> [--port PORT]

支持查看 verl 训练过程中生成的 rollout trajectory jsonl 文件
"""
import argparse
import json
import re
import html as html_module
from pathlib import Path
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Rollout JSONL Viewer")

# 全局配置
JSONL_PATH = None
JSONL_DATA = []


def parse_input_to_messages(input_str: str) -> List[Dict[str, str]]:
    """
    将 input 字符串解析为多轮对话消息列表
    
    基于 tool_agent_loop.py 中的对话构建逻辑：
    - system: 系统提示
    - user: 用户问题  
    - assistant: 模型输出 (包含 <think>, <tool_call>, <answer> 等)
    - user + <tool_response>: 工具响应（Qwen chat template 把 tool role 转成 user + <tool_response>）
    
    input 格式示例（Qwen chat template 输出）:
    system
    ... system content ...
    user
    ... user content ...
    assistant
    <think>...</think>
    <tool_call>...</tool_call>user           <-- 注意：可能没有换行！
    <tool_response>...</tool_response>
    assistant
    ...
    """
    messages = []
    
    # Qwen chat template 的 role 标记可能：
    # 1. 在行首后跟换行: "system\n", "user\n", "assistant\n"
    # 2. 紧跟在 </tool_call> 后面: "</tool_call>user\n"
    # 
    # 先预处理：在 </tool_call>user 和 </tool_response>assistant 之间插入换行
    processed = input_str
    processed = re.sub(r'</tool_call>(user)', r'</tool_call>\n\1', processed)
    processed = re.sub(r'</tool_response>(assistant)', r'</tool_response>\n\1', processed)
    
    # 按照 role 标记分割
    # 匹配模式：行首的 system/user/assistant 后跟换行
    pattern = r'^(system|user|assistant)\n'
    
    # 找到所有 role 标记的位置
    role_positions = []
    for match in re.finditer(pattern, processed, re.MULTILINE):
        role_positions.append((match.start(), match.end(), match.group(1)))
    
    if not role_positions:
        # 如果没有找到标准格式，返回整个内容作为 text
        return [{"role": "text", "content": input_str}]
    
    # 根据位置提取每个 role 的内容
    for i, (start, end, role) in enumerate(role_positions):
        # 内容从当前 role 标记结束到下一个 role 标记开始（或字符串结尾）
        if i + 1 < len(role_positions):
            content_end = role_positions[i + 1][0]
        else:
            content_end = len(processed)
        
        content = processed[end:content_end].strip()
        
        # 智能识别 role 类型:
        # 如果是 user role 但内容主要是 <tool_response>，则标记为 tool
        actual_role = role
        if role == 'user':
            # 检查是否是工具响应（Qwen chat template 把 tool role 转成 user + <tool_response>）
            content_stripped = content.strip()
            if content_stripped.startswith('<tool_response>') or \
               (content_stripped.startswith('{') and '<tool_response>' in content_stripped[:100]):
                actual_role = 'tool'
        
        messages.append({"role": actual_role, "content": content})
    
    return messages


def parse_message_parts(content: str) -> List[Tuple[str, str]]:
    """解析消息内容，按顺序提取标签块
    
    基于 tool_agent_loop.py 中的对话构建逻辑，agent 输出格式:
    
    1. assistant 消息格式（模型生成）:
       - <think>思考内容</think> 或 思考内容</think>（Qwen 格式可能没有开始标签）
       - <tool_call>{"name": "xxx", "arguments": {...}}</tool_call>
       - <answer>最终答案</answer>
    
    2. tool 消息格式（工具响应）:
       - <tool_response>{"result": "..."}</tool_response>
       - 可能有多个 tool_response（并行工具调用）
    
    返回: [(type, content), ...] 按出现顺序排列
    """
    parts = []
    
    # 定义所有要匹配的标签模式
    tag_patterns = [
        ('think', r'<think>(.*?)</think>'),
        ('tool_call', r'<tool_call>(.*?)</tool_call>'),
        ('answer', r'<answer>(.*?)</answer>'),
        ('tool_response', r'<tool_response>(.*?)</tool_response>'),
    ]
    
    # 收集所有匹配，记录位置和内容
    all_matches = []
    
    # 特殊处理：Qwen 格式可能只有 </think> 结束标签（思考内容在开头）
    has_think_start = '<think>' in content
    has_think_end = '</think>' in content
    
    if has_think_end and not has_think_start:
        # 没有开始标签，内容从头到 </think> 是思考内容
        think_end_pos = content.find('</think>')
        if think_end_pos > 0:
            think_content = content[:think_end_pos].strip()
            if think_content:
                all_matches.append((0, think_end_pos + len('</think>'), 'think', think_content))
    
    # 匹配所有标准标签
    for tag_type, pattern in tag_patterns:
        for match in re.finditer(pattern, content, re.DOTALL):
            # 跳过已经被特殊处理的 think（如果存在）
            if tag_type == 'think' and has_think_end and not has_think_start:
                continue
            all_matches.append((match.start(), match.end(), tag_type, match.group(1).strip()))
    
    # 按位置排序
    all_matches.sort(key=lambda x: x[0])
    
    # 提取匹配到的内容和未匹配的内容
    last_end = 0
    for start, end, tag_type, tag_content in all_matches:
        # 添加匹配前的未标记内容
        if start > last_end:
            gap_content = content[last_end:start].strip()
            # 清理可能残留的 </think> 标签
            gap_content = gap_content.replace('</think>', '').strip()
            if gap_content:
                parts.append(('other', gap_content))
        
        # 添加标签内容
        if tag_content:
            parts.append((tag_type, tag_content))
        
        last_end = end
    
    # 添加最后的未标记内容
    if last_end < len(content):
        tail_content = content[last_end:].strip()
        if tail_content:
            parts.append(('other', tail_content))
    
    # 如果没有匹配到任何内容，返回整个内容作为 text
    if not parts:
        parts.append(('text', content))
    
    return parts


def linkify_urls(text: str, preserve_newlines: bool = True) -> str:
    """将文本中的URL转换为可点击的超链接
    
    Args:
        text: 要处理的文本
        preserve_newlines: 是否保留换行符（转换为<br>）
    """
    text = html_module.escape(text)
    
    # 将换行符转换为<br>标签（如果需要）
    if preserve_newlines:
        text = text.replace('\n', '<br>')
    
    # Markdown 格式链接
    text = re.sub(
        r'\[([^\]]+)\]\(([^\)]+)\)',
        r'<a href="\2" target="_blank" style="color: #0066cc;">\1</a>',
        text
    )
    
    # 普通 URL
    def replace_url(match):
        url = match.group(0)
        return f'<a href="{url}" target="_blank" style="color: #0066cc;">{url}</a>'
    
    text = re.sub(
        r'(?<!href=")(?<!">)(https?://[^\s<>"\')\]]+)',
        replace_url,
        text
    )
    
    return text


def format_json_html(json_str: str) -> str:
    """格式化JSON字符串为HTML，保留换行符"""
    try:
        obj = json.loads(json_str)
        formatted = json.dumps(obj, indent=2, ensure_ascii=False)
        # 先转义HTML特殊字符
        formatted = formatted.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # 语法高亮
        formatted = re.sub(r'"([^"]+)":', r'<span class="json-key">"\1"</span>:', formatted)
        formatted = re.sub(r': "([^"]*)"', r': <span class="json-string">"\1"</span>', formatted)
        formatted = re.sub(r': (\d+)', r': <span class="json-number">\1</span>', formatted)
        formatted = re.sub(r': (true|false|null)', r': <span class="json-bool">\1</span>', formatted)
        return f'<pre class="json-code">{formatted}</pre>'
    except:
        # 如果不是有效的JSON，直接显示原始内容并保留换行
        escaped = html_module.escape(json_str)
        return f'<pre class="json-code">{escaped}</pre>'


def count_turns_from_conversation(input_str: str, output_str: str) -> tuple:
    """从对话内容中计算轮数
    
    Returns:
        (assistant_turns, user_turns, total_turns)
    """
    full_conversation = input_str + output_str
    messages = parse_input_to_messages(full_conversation)
    
    assistant_turns = sum(1 for m in messages if m['role'] == 'assistant')
    user_turns = sum(1 for m in messages if m['role'] in ('user', 'tool'))
    total_turns = len(messages)
    
    return assistant_turns, user_turns, total_turns


def load_jsonl_data() -> List[Dict]:
    """加载 JSONL 文件数据"""
    global JSONL_DATA
    JSONL_DATA = []
    
    if not JSONL_PATH or not JSONL_PATH.exists():
        return JSONL_DATA
    
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                data['_index'] = idx
                
                # 动态计算轮数（如果数据中没有这些字段）
                if 'assistant_turns' not in data or 'user_turns' not in data:
                    input_str = data.get('input', '')
                    output_str = data.get('output', '')
                    a_turns, u_turns, total = count_turns_from_conversation(input_str, output_str)
                    data['assistant_turns'] = data.get('assistant_turns', a_turns)
                    data['user_turns'] = data.get('user_turns', u_turns)
                    data['total_turns'] = data.get('total_turns', total)
                
                JSONL_DATA.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {idx + 1}: {e}")
    
    return JSONL_DATA


# CSS 样式
COMMON_STYLES = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: #f8fafc; color: #1e293b; padding: 40px 20px; line-height: 1.6; }
.container { max-width: 2200px; margin: 0 auto; width: 95%; }
h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; color: #0f172a; }
.subtitle { color: #64748b; font-size: 14px; margin-bottom: 30px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }
.stat-card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: all 0.2s; }
.stat-card:hover { border-color: #3b82f6; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(59,130,246,0.15); }
.stat-value { font-size: 28px; font-weight: 700; color: #3b82f6; margin-bottom: 4px; font-feature-settings: 'tnum'; }
.stat-label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
.table-container { background: white; border-radius: 12px; border: 1px solid #e2e8f0; overflow-x: auto; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
table { width: 100%; border-collapse: collapse; min-width: 1200px; }
th { background: #f1f5f9; padding: 14px 16px; text-align: left; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 2px solid #e2e8f0; color: #475569; }
td { padding: 14px 16px; border-bottom: 1px solid #f1f5f9; font-size: 14px; }
tr:hover { background: #f8fafc; }
tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; }
.badge-correct { background: #dcfce7; color: #166534; }
.badge-incorrect { background: #fee2e2; color: #991b1b; }
.badge-info { background: #dbeafe; color: #1e40af; }
.badge-warning { background: #fef3c7; color: #92400e; }
.badge-purple { background: #f3e8ff; color: #7c3aed; }
.turn-badge { display: inline-block; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; background: #e0f2fe; color: #0369a1; font-feature-settings: 'tnum'; }
.score-badge { display: inline-block; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 13px; }
.question-cell { max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.answer-cell { max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: 'JetBrains Mono', monospace; font-size: 13px; }
.view-btn { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; text-decoration: none; display: inline-block; font-size: 13px; font-weight: 600; transition: all 0.2s; }
.view-btn:hover { background: linear-gradient(135deg, #2563eb, #1d4ed8); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,130,246,0.3); }
.filter-controls { display: flex; gap: 16px; align-items: center; margin-bottom: 20px; padding: 16px 20px; background: white; border-radius: 12px; border: 1px solid #e2e8f0; flex-wrap: wrap; }
.filter-label { font-size: 13px; font-weight: 600; color: #475569; }
.filter-select { padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 8px; font-size: 14px; background: white; cursor: pointer; min-width: 140px; }
.filter-select:hover { border-color: #3b82f6; }
.filter-select:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,0.15); }
"""

DETAIL_STYLES = """
.back-btn { display: inline-flex; align-items: center; gap: 8px; background: white; color: #475569; padding: 10px 16px; border-radius: 8px; text-decoration: none; font-size: 14px; font-weight: 500; border: 1px solid #e2e8f0; margin-bottom: 24px; transition: all 0.2s; }
.back-btn:hover { border-color: #3b82f6; color: #3b82f6; background: #f8fafc; }
.info-box { background: white; padding: 28px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 28px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.info-box h1 { font-size: 18px; font-weight: 700; color: #0f172a; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 2px solid #e2e8f0; }
.info-grid { display: grid; gap: 14px; }
.info-row { display: grid; grid-template-columns: 160px 1fr; gap: 16px; align-items: start; }
.info-label { font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; padding-top: 2px; }
.info-value { color: #1e293b; font-size: 14px; word-break: break-word; }
.info-value pre { background: #f1f5f9; padding: 12px; border-radius: 8px; overflow-x: auto; font-family: 'JetBrains Mono', monospace; font-size: 13px; white-space: pre-wrap; }
.message { background: white; margin-bottom: 20px; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.message-header { padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e2e8f0; }
.message.system .message-header { background: #f1f5f9; }
.message.user .message-header { background: #fef2f2; border-bottom-color: #fecaca; }
.message.assistant .message-header { background: #eff6ff; border-bottom-color: #bfdbfe; }
.message.tool .message-header { background: #f0fdf4; border-bottom-color: #bbf7d0; }
.role-badge { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; padding: 5px 12px; border-radius: 6px; }
.message.system .role-badge { background: #e2e8f0; color: #475569; }
.message.user .role-badge { background: #fee2e2; color: #991b1b; }
.message.assistant .role-badge { background: #dbeafe; color: #1e40af; }
.message.tool .role-badge { background: #dcfce7; color: #166534; }
.message-body { padding: 20px; }
.content-block { margin-bottom: 14px; border-radius: 8px; border: 1px solid #e2e8f0; overflow: hidden; }
.block-header { padding: 10px 14px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; display: flex; justify-content: space-between; align-items: center; cursor: pointer; user-select: none; transition: opacity 0.2s; }
.block-header:hover { opacity: 0.8; }
.block-content { padding: 16px; line-height: 1.7; word-wrap: break-word; border-top: 1px solid #e2e8f0; font-size: 14px; }
.block-content.collapsed { display: none; }
.expand-icon { font-size: 10px; transition: transform 0.2s; }
.expand-icon.expanded { transform: rotate(180deg); }
.block-think .block-header { background: #eff6ff; color: #1e40af; }
.block-think .block-content { background: #f8fafc; }
.block-tool_call .block-header { background: #faf5ff; color: #7c3aed; }
.block-tool_call .block-content { background: #fafafa; }
.block-tool_response .block-header { background: #f0fdf4; color: #166534; }
.block-tool_response .block-content { background: #fafafa; max-height: 500px; overflow-y: auto; }
.block-answer .block-header { background: #fefce8; color: #a16207; }
.block-answer .block-content { background: #fffef0; font-weight: 500; }
.block-other .block-header { background: #fef2f2; color: #991b1b; }
.block-other .block-content { background: #fef2f2; }
.block-text .block-header { background: #f1f5f9; color: #475569; }
.block-text .block-content { background: #fafafa; }
.json-code { margin: 0; padding: 14px; background: #1e293b; color: #e2e8f0; border-radius: 8px; overflow-x: auto; font-family: 'JetBrains Mono', monospace; font-size: 13px; line-height: 1.6; }
.json-key { color: #7dd3fc; }
.json-string { color: #fca5a5; }
.json-number { color: #86efac; }
.json-bool { color: #c4b5fd; }
.char-badge { background: rgba(0, 0, 0, 0.05); padding: 4px 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; color: #64748b; }
"""


@app.get("/", response_class=HTMLResponse)
def index(filter_correct: str = "all", filter_method: str = "all", filter_status: str = "all"):
    """主页面 - 列表视图"""
    data = load_jsonl_data()
    
    # 统计信息
    total = len(data)
    correct_count = sum(1 for d in data if d.get('correct') is True)
    incorrect_count = sum(1 for d in data if d.get('correct') is False)
    accuracy = (correct_count / total * 100) if total > 0 else 0
    avg_score = sum(d.get('score', 0) for d in data) / total if total > 0 else 0
    
    # 统计方法分布
    methods = {}
    for d in data:
        method = d.get('method', 'unknown')
        methods[method] = methods.get(method, 0) + 1
    
    # 统计轮数
    avg_assistant_turns = sum(d.get('assistant_turns', 0) for d in data) / total if total > 0 else 0
    avg_user_turns = sum(d.get('user_turns', 0) for d in data) / total if total > 0 else 0
    avg_total_turns = sum(d.get('total_turns', 0) for d in data) / total if total > 0 else 0
    max_total_turns = max((d.get('total_turns', 0) for d in data), default=0)
    min_total_turns = min((d.get('total_turns', 0) for d in data), default=0)
    
    # 统计LLM Judge使用情况
    llm_judge_count = sum(1 for d in data if d.get('raw_response') and d.get('raw_response').strip())
    llm_judge_correct = sum(1 for d in data if d.get('raw_response') and d.get('raw_response').strip() and d.get('correct') is True)
    llm_judge_accuracy = (llm_judge_correct / llm_judge_count * 100) if llm_judge_count > 0 else 0
    
    # 统计异常终止情况（没有 <answer> 标签的样本）
    no_answer_count = 0
    has_error_count = 0
    truncated_count = 0
    for d in data:
        output = d.get('output', '')
        # 没有 answer 标签
        if '<answer>' not in output or '</answer>' not in output:
            no_answer_count += 1
        # 包含错误/超时信息
        if any(err in output.lower() for err in ['error', 'timeout', 'failed', 'exception']):
            has_error_count += 1
        # 被截断（通常在末尾没有正常结束）
        if output.strip() and not output.strip().endswith(('</answer>', '</think>', '</tool_call>')):
            truncated_count += 1
    
    no_answer_ratio = (no_answer_count / total * 100) if total > 0 else 0
    has_error_ratio = (has_error_count / total * 100) if total > 0 else 0
    truncated_ratio = (truncated_count / total * 100) if total > 0 else 0
    
    # 统计 token 数量
    has_token_stats = any('total_tokens' in d for d in data)
    if has_token_stats:
        avg_input_tokens = sum(d.get('input_tokens', 0) for d in data) / total if total > 0 else 0
        avg_output_tokens = sum(d.get('output_tokens', 0) for d in data) / total if total > 0 else 0
        avg_total_tokens = sum(d.get('total_tokens', 0) for d in data) / total if total > 0 else 0
        max_total_tokens = max((d.get('total_tokens', 0) for d in data), default=0)
    else:
        avg_input_tokens = avg_output_tokens = avg_total_tokens = max_total_tokens = 0
    
    # 辅助函数：检查样本状态
    def get_sample_status(d):
        output = d.get('output', '')
        if '<answer>' not in output or '</answer>' not in output:
            return 'no_answer'
        if output.strip() and not output.strip().endswith(('</answer>', '</think>', '</tool_call>')):
            return 'truncated'
        return 'normal'
    
    # 应用筛选
    filtered_data = data
    if filter_correct == "correct":
        filtered_data = [d for d in filtered_data if d.get('correct') is True]
    elif filter_correct == "incorrect":
        filtered_data = [d for d in filtered_data if d.get('correct') is False]
    
    if filter_method != "all":
        filtered_data = [d for d in filtered_data if d.get('method', 'unknown') == filter_method]
    
    if filter_status == "no_answer":
        filtered_data = [d for d in filtered_data if get_sample_status(d) == 'no_answer']
    elif filter_status == "truncated":
        filtered_data = [d for d in filtered_data if get_sample_status(d) == 'truncated']
    elif filter_status == "normal":
        filtered_data = [d for d in filtered_data if get_sample_status(d) == 'normal']
    
    # 方法选项
    method_options = "".join(
        f'<option value="{m}" {"selected" if filter_method == m else ""}>{m} ({c})</option>'
        for m, c in sorted(methods.items())
    )
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Rollout Viewer - {JSONL_PATH.name}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>{COMMON_STYLES}</style>
</head>
<body>
    <div class="container">
        <h1>🔍 Rollout Trajectory Viewer</h1>
        <div class="subtitle">{JSONL_PATH}</div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #16a34a;">{correct_count}</div>
                <div class="stat-label">Correct</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #dc2626;">{incorrect_count}</div>
                <div class="stat-label">Incorrect</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{accuracy:.1f}%</div>
                <div class="stat-label">Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #7c3aed;">{avg_score:.2f}</div>
                <div class="stat-label">Avg Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #0891b2;">{avg_total_turns:.1f}</div>
                <div class="stat-label">Avg Total Turns</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #06b6d4;">{avg_assistant_turns:.1f}</div>
                <div class="stat-label">Avg Assistant Turns</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #f59e0b;">{llm_judge_count}</div>
                <div class="stat-label">LLM Judge Used</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #10b981;">{llm_judge_accuracy:.1f}%</div>
                <div class="stat-label">LLM Judge Correct Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #ef4444;">{no_answer_count}</div>
                <div class="stat-label">No Answer Tag ({no_answer_ratio:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #f97316;">{truncated_count}</div>
                <div class="stat-label">Truncated ({truncated_ratio:.1f}%)</div>
            </div>
            {"" if not has_token_stats else f'''
            <div class="stat-card">
                <div class="stat-value" style="color: #8b5cf6;">{avg_total_tokens:.0f}</div>
                <div class="stat-label">Avg Total Tokens</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #a855f7;">{avg_output_tokens:.0f}</div>
                <div class="stat-label">Avg Output Tokens</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="color: #c084fc;">{max_total_tokens}</div>
                <div class="stat-label">Max Total Tokens</div>
            </div>
            '''}
        </div>
        
        <div class="filter-controls">
            <span class="filter-label">结果筛选:</span>
            <select class="filter-select" id="filterCorrect" onchange="applyFilters()">
                <option value="all" {"selected" if filter_correct == "all" else ""}>全部 ({total})</option>
                <option value="correct" {"selected" if filter_correct == "correct" else ""}>正确 ({correct_count})</option>
                <option value="incorrect" {"selected" if filter_correct == "incorrect" else ""}>错误 ({incorrect_count})</option>
            </select>
            
            <span class="filter-label" style="margin-left: 20px;">Method 筛选:</span>
            <select class="filter-select" id="filterMethod" onchange="applyFilters()">
                <option value="all" {"selected" if filter_method == "all" else ""}>全部</option>
                {method_options}
            </select>
            
            <span class="filter-label" style="margin-left: 20px;">状态筛选:</span>
            <select class="filter-select" id="filterStatus" onchange="applyFilters()">
                <option value="all" {"selected" if filter_status == "all" else ""}>全部</option>
                <option value="no_answer" {"selected" if filter_status == "no_answer" else ""}>无Answer ({no_answer_count})</option>
                <option value="truncated" {"selected" if filter_status == "truncated" else ""}>被截断 ({truncated_count})</option>
                <option value="normal" {"selected" if filter_status == "normal" else ""}>正常</option>
            </select>
            
            <span style="margin-left: auto; color: #64748b; font-size: 13px;">显示 {len(filtered_data)} 条</span>
        </div>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th style="width: 60px;">#</th>
                        <th style="width: 80px;">Step</th>
                        <th style="width: 100px;">Score</th>
                        <th style="width: 100px;">Correct</th>
                        <th style="width: 100px;">Method</th>
                        <th style="width: 80px;">Turns</th>
                        <th style="width: 100px;">Tokens</th>
                        <th>Question (from input)</th>
                        <th style="width: 200px;">Pred Answer</th>
                        <th style="width: 200px;">Ground Truth</th>
                        <th style="width: 100px;">Action</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for d in filtered_data:
        idx = d['_index']
        step = d.get('step', '-')
        score = d.get('score', 0)
        correct = d.get('correct')
        method = d.get('method', 'unknown')
        total_turns = d.get('total_turns', '-')
        total_tokens = d.get('total_tokens', '-')
        output_tokens = d.get('output_tokens', '-')
        pred_ans = d.get('pred_ans', '')[:100]
        gts = d.get('gts', {})
        gt_target = gts.get('target', [''])[0] if isinstance(gts.get('target'), list) else str(gts.get('target', ''))
        
        # 从 input 中提取问题（找第一个 user 消息中的 Question:）
        input_str = d.get('input', '')
        question_match = re.search(r'Question:\s*(.+?)(?:\n|$)', input_str)
        question = question_match.group(1)[:100] if question_match else input_str[:100]
        
        # 样式
        if correct is True:
            correct_badge = '<span class="badge badge-correct">✓ Correct</span>'
            score_style = "background: #dcfce7; color: #166534;"
        elif correct is False:
            correct_badge = '<span class="badge badge-incorrect">✗ Wrong</span>'
            score_style = "background: #fee2e2; color: #991b1b;"
        else:
            correct_badge = '<span class="badge badge-warning">Unknown</span>'
            score_style = "background: #fef3c7; color: #92400e;"
        
        # Token 显示格式
        if total_tokens != '-':
            token_display = f"{output_tokens}/{total_tokens}"
        else:
            token_display = "-"
        
        html += f"""
                    <tr>
                        <td>{idx + 1}</td>
                        <td><span class="turn-badge">{step}</span></td>
                        <td><span class="score-badge" style="{score_style}">{score:.2f}</span></td>
                        <td>{correct_badge}</td>
                        <td><span class="badge badge-purple">{method}</span></td>
                        <td><span class="turn-badge" style="background: #e0e7ff; color: #4338ca;">{total_turns}</span></td>
                        <td><span class="turn-badge" style="background: #f3e8ff; color: #7c3aed;" title="Output/Total">{token_display}</span></td>
                        <td class="question-cell" title="{html_module.escape(question)}">{html_module.escape(question)}</td>
                        <td class="answer-cell" title="{html_module.escape(pred_ans)}">{html_module.escape(pred_ans)}</td>
                        <td class="answer-cell" title="{html_module.escape(gt_target)}">{html_module.escape(gt_target)}</td>
                        <td><a class="view-btn" href="/view/{idx}" target="_blank">查看</a></td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </div>
    </div>
    <script>
        function applyFilters() {
            const filterCorrect = document.getElementById('filterCorrect').value;
            const filterMethod = document.getElementById('filterMethod').value;
            const filterStatus = document.getElementById('filterStatus').value;
            window.location.href = `/?filter_correct=${filterCorrect}&filter_method=${filterMethod}&filter_status=${filterStatus}`;
        }
    </script>
</body>
</html>
"""
    return html


@app.get("/view/{idx}", response_class=HTMLResponse)
def view_detail(idx: int):
    """查看单个样本详情"""
    data = load_jsonl_data()
    
    if idx < 0 or idx >= len(data):
        return HTMLResponse(f"<h1>索引超出范围: {idx}</h1>", status_code=404)
    
    d = data[idx]
    
    # 基本信息
    step = d.get('step', '-')
    score = d.get('score', 0)
    correct = d.get('correct')
    method = d.get('method', 'unknown')
    reason = d.get('reason', '')
    pred_ans = d.get('pred_ans', '')
    raw_response = d.get('raw_response', '')
    assistant_turns = d.get('assistant_turns', '-')
    user_turns = d.get('user_turns', '-')
    data_source = d.get('data_source', 'unknown')
    input_tokens = d.get('input_tokens', '-')
    output_tokens = d.get('output_tokens', '-')
    total_tokens = d.get('total_tokens', '-')
    
    # Ground truth
    gts = d.get('gts', {})
    gt_target = gts.get('target', '')
    if isinstance(gt_target, list):
        gt_target = ', '.join(str(t) for t in gt_target)
    
    # 解析完整对话: input + output
    # input 包含 system + user 问题 + "assistant\n<think>\n"
    # output 包含模型生成的完整多轮对话（包括多轮 tool_call → tool_response）
    input_str = d.get('input', '')
    output_str = d.get('output', '')
    
    # 合并 input 和 output 来获取完整对话
    # input 通常以 "assistant\n<think>\n" 结尾，output 从思考内容开始
    full_conversation = input_str + output_str
    messages = parse_input_to_messages(full_conversation)
    
    # 样式
    if correct is True:
        correct_badge = '<span class="badge badge-correct" style="font-size: 14px; padding: 6px 14px;">✓ Correct</span>'
    elif correct is False:
        correct_badge = '<span class="badge badge-incorrect" style="font-size: 14px; padding: 6px 14px;">✗ Wrong</span>'
    else:
        correct_badge = '<span class="badge badge-warning" style="font-size: 14px; padding: 6px 14px;">Unknown</span>'
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Sample #{idx + 1} - Rollout Viewer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        {COMMON_STYLES}
        {DETAIL_STYLES}
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← 返回列表</a>
        
        <div class="info-box">
            <h1>📋 Sample #{idx + 1} 详细信息</h1>
            <div class="info-grid">
                <div class="info-row">
                    <div class="info-label">Step</div>
                    <div class="info-value"><span class="turn-badge" style="font-size: 14px;">{step}</span></div>
                </div>
                <div class="info-row">
                    <div class="info-label">Data Source</div>
                    <div class="info-value"><span class="badge badge-info">{data_source}</span></div>
                </div>
                <div class="info-row">
                    <div class="info-label">Score</div>
                    <div class="info-value"><span style="font-size: 20px; font-weight: 700; color: {'#16a34a' if score > 0.5 else '#dc2626'};">{score:.4f}</span></div>
                </div>
                <div class="info-row">
                    <div class="info-label">Result</div>
                    <div class="info-value">{correct_badge}</div>
                </div>
                <div class="info-row">
                    <div class="info-label">Method</div>
                    <div class="info-value"><span class="badge badge-purple">{method}</span></div>
                </div>
                <div class="info-row">
                    <div class="info-label">Turns</div>
                    <div class="info-value">
                        <span class="turn-badge">Total: {d.get('total_turns', '-')}</span>
                        <span class="turn-badge" style="margin-left: 8px;">Assistant: {assistant_turns}</span>
                        <span class="turn-badge" style="margin-left: 8px;">User/Tool: {user_turns}</span>
                    </div>
                </div>
                <div class="info-row">
                    <div class="info-label">Tokens</div>
                    <div class="info-value">
                        <span class="turn-badge" style="background: #f3e8ff; color: #7c3aed;">Input: {input_tokens}</span>
                        <span class="turn-badge" style="margin-left: 8px; background: #fae8ff; color: #a21caf;">Output: {output_tokens}</span>
                        <span class="turn-badge" style="margin-left: 8px; background: #ede9fe; color: #6d28d9;">Total: {total_tokens}</span>
                    </div>
                </div>
                <div class="info-row">
                    <div class="info-label">Ground Truth</div>
                    <div class="info-value"><pre>{html_module.escape(str(gt_target))}</pre></div>
                </div>
                <div class="info-row">
                    <div class="info-label">Pred Answer</div>
                    <div class="info-value"><pre>{html_module.escape(pred_ans)}</pre></div>
                </div>
                <div class="info-row">
                    <div class="info-label">Judge Reason</div>
                    <div class="info-value"><pre>{html_module.escape(reason)}</pre></div>
                </div>
"""
    
    html += """
            </div>
        </div>
"""
    
    # 统计各类消息数量
    system_count = sum(1 for m in messages if m['role'] == 'system')
    user_count = sum(1 for m in messages if m['role'] == 'user')
    assistant_count = sum(1 for m in messages if m['role'] == 'assistant')
    tool_count = sum(1 for m in messages if m['role'] == 'tool')
    
    html += f"""
        <h2 style="font-size: 18px; font-weight: 700; margin-bottom: 12px; color: #0f172a;">💬 对话历史</h2>
        <div style="margin-bottom: 20px; display: flex; gap: 10px; flex-wrap: wrap;">
            <span class="turn-badge" style="background: #e0e7ff; color: #4338ca;">总轮数: {len(messages)}</span>
            <span class="turn-badge" style="background: #f1f5f9; color: #475569;">System: {system_count}</span>
            <span class="turn-badge" style="background: #fee2e2; color: #991b1b;">User: {user_count}</span>
            <span class="turn-badge" style="background: #dbeafe; color: #1e40af;">Assistant: {assistant_count}</span>
            <span class="turn-badge" style="background: #dcfce7; color: #166534;">Tool: {tool_count}</span>
        </div>
"""
    
    # 渲染每条消息
    for msg_idx, msg in enumerate(messages):
        role = msg['role']
        content = msg['content']
        
        html += f'<div class="message {role}">'
        html += f'<div class="message-header">'
        html += f'<span class="role-badge">{role.upper()}</span>'
        html += f'<span class="char-badge">Turn {msg_idx + 1} · {len(content):,} chars</span>'
        html += f'</div>'
        html += f'<div class="message-body">'
        
        # 解析消息内容
        # 根据 tool_agent_loop.py 的逻辑:
        # - system: 系统提示，纯文本
        # - user: 用户问题，纯文本
        # - assistant: 模型输出，包含 <think>, <tool_call>, <answer> 标签
        # - tool: 工具响应，包含 <tool_response> 标签（从 user role 识别出来的）
        if role == 'system':
            parts = [('text', content)]
        elif role == 'user':
            parts = [('text', content)]
        elif role == 'tool':
            # 工具响应，提取 <tool_response> 内容
            parts = parse_message_parts(content)
        else:  # assistant
            parts = parse_message_parts(content)
        
        for part_idx, (part_type, part_content) in enumerate(parts):
            block_class = f"block-{part_type}"
            type_name = part_type.replace('_', ' ').title()
            block_id = f"block-{msg_idx}-{part_idx}"
            
            html += f'<div class="content-block {block_class}">'
            html += f'<div class="block-header" onclick="toggleBlock(\'{block_id}\')">'
            html += f'<span>{type_name}</span>'
            html += f'<span><span class="char-badge">{len(part_content):,} chars</span> <span class="expand-icon expanded" id="{block_id}-icon">▼</span></span>'
            html += f'</div>'
            html += f'<div class="block-content" id="{block_id}">'
            
            if part_type == 'tool_call':
                html += format_json_html(part_content)
            elif part_type == 'tool_response':
                # Tool response需要特殊处理：先尝试解析JSON，如果失败则按纯文本显示
                html += format_json_html(part_content)
            else:
                # 其他内容（think, answer等）也保留换行
                html += linkify_urls(part_content, preserve_newlines=True)
            
            html += f'</div></div>'
        
        html += f'</div></div>'
    
    # raw_response 是 LLM Judge 的原始响应
    if raw_response:
        html += f"""
        <h2 style="font-size: 18px; font-weight: 700; margin: 30px 0 20px 0; color: #0f172a;">🤖 LLM Judge Raw Response</h2>
        <div class="info-box">
            <pre style="white-space: pre-wrap; word-wrap: break-word; font-size: 13px;">{html_module.escape(raw_response)}</pre>
        </div>
"""
    
    html += """
    </div>
    <script>
        function toggleBlock(blockId) {
            const content = document.getElementById(blockId);
            const icon = document.getElementById(blockId + '-icon');
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                icon.classList.add('expanded');
            } else {
                content.classList.add('collapsed');
                icon.classList.remove('expanded');
            }
        }
    </script>
</body>
</html>
"""
    return html


@app.get("/api/data")
def get_data():
    """获取所有数据的 API"""
    return load_jsonl_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rollout JSONL Viewer')
    parser.add_argument('jsonl_path', help='Path to the JSONL file')
    parser.add_argument('--port', type=int, default=7788, help='Port to run on (default: 7788)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    JSONL_PATH = Path(args.jsonl_path).resolve()
    
    if not JSONL_PATH.exists():
        print(f"❌ Error: File {JSONL_PATH} does not exist")
        exit(1)
    
    if not JSONL_PATH.suffix == '.jsonl':
        print(f"⚠️  Warning: File does not have .jsonl extension: {JSONL_PATH}")
    
    # 预加载数据
    data = load_jsonl_data()
    
    print(f"🚀 启动 Rollout JSONL Viewer...")
    print(f"📂 文件路径: {JSONL_PATH}")
    print(f"📊 加载样本数: {len(data)}")
    print(f"🌐 访问地址: http://localhost:{args.port}")
    
    uvicorn.run(app, host=args.host, port=args.port)

