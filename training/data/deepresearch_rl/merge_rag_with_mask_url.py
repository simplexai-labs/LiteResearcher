#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并多个 parquet 文件并正确处理 mask_url

使用方法:
    from merge_rag_with_mask_url import merge_parquet_files

    merge_parquet_files(
        input_files=["path1.parquet", "path2.parquet", "path3.parquet"],
        output_file="output.parquet"
    )

=======================================================================
输入文件统计 (stage2_all_0210):
--------------------------------------------------------------------------------
No.  Count    File
--------------------------------------------------------------------------------
1    16       rag_mqa_subgraph6_1-7_32k-48k.parquet
2    902      rag_mqa_subgraph6_1-7_lt16k.parquet
3    314      rag_mqa_subgraph6_1-7_16k-32k.parquet
4    442      rag_mqa_1-7_16k-32k.parquet
5    88       rag_mqa_1-7_32k-48k.parquet
6    20       rag_mqa_1-7_48k-64k.parquet
7    20       rag_mqa_1-7_48k-64k.parquet (重复)
8    117      rag_mqa_subgraph7_1-7_16k-32k.parquet
9    6        rag_mqa_subgraph7_1-7_32k-48k.parquet
10   386      rag_mqa_subgraph7_1-7_lt16k.parquet
11   354      rag_mqa_subgraph5_1-7_16k-32k.parquet
12   29       rag_mqa_subgraph5_1-7_32k-48k.parquet
13   3        rag_mqa_subgraph5_1-7_48k-64k.parquet
14   601      rag_mqa_subgraph5_1-7_lt16k.parquet
15   863      rag_direct_1-7_16k-32k.parquet
16   88       rag_direct_1-7_32k-48k.parquet
17   15       rag_direct_1-7_48k-64k.parquet
18   4342     rag_direct_1-7_lt16k.parquet
19   1448     rag_direct_wiki_1-7_16k-32k.parquet
20   114      rag_direct_wiki_1-7_32k-48k.parquet
21   2        rag_direct_wiki_1-7_48k-64k.parquet
22   4229     rag_direct_wiki_1-7_lt16k.parquet
23   409      rag_science_1-7_16k-32k.parquet
24   60       rag_science_1-7_32k-48k.parquet
25   18       rag_science_1-7_48k-64k.parquet
26   1313     rag_science_1-7_lt16k.parquet
--------------------------------------------------------------------------------
Total: 16179 (注: 序号7是重复的)

数据分类:
- rag_mqa_*: MQA (Multi-Query Answering) 相关数据
- rag_direct_*: 直接 RAG 数据
- rag_direct_wiki_*: Wikipedia 来源的直接 RAG 数据
- rag_science_*: 科学相关 RAG 数据
- subgraph5/6/7: 子图分割数据

长度分布:
- lt16k: 小于 16k tokens
- 16k-32k: 16k-32k tokens
- 32k-48k: 32k-48k tokens
- 48k-64k: 48k-64k tokens
=======================================================================
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

# 系统提示词模板（与 stage2_wiki 保持一致）
SYSTEM_PROMPT = """* You are Deep AI Research Assistant

The question I give you is a complex question that requires a *deep research* to answer.

I will provide you with two tools to help you answer the question:
* A web search tool to help you perform google search. 
* A webpage browsing tool to help you get new page content.

You don't have to answer the question now, but you should first think about the research plan or what to search next.

Your output format should be one of the following two formats:

<think>
YOUR THINKING PROCESS
</think>
<answer>
YOUR ANSWER AFTER GETTING ENOUGH INFORMATION
</answer>

or

<think>
YOUR THINKING PROCESS
</think>
<tool_call>
YOUR TOOL CALL WITH CORRECT FORMAT
</tool_call>

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

You should always follow the above two formats strictly.
Only output the final answer (in words, numbers or phrase) inside the <answer></answer> tag, without any explanations or extra information. If this is a yes-or-no question, you should only answer yes or no.

Current date: 2025-12-26"""


def convert_reward_model(gt):
    """转换 reward_model 格式"""
    if isinstance(gt, str):
        return {'ground_truth': {'target': np.array([gt], dtype=object)}, 'style': 'llm'}
    elif isinstance(gt, dict) and 'target' in gt:
        return {'ground_truth': gt, 'style': 'llm'}
    return {'ground_truth': {'target': np.array([str(gt)], dtype=object)}, 'style': 'llm'}


def build_tools_kwargs(mask_url: str) -> Dict[str, Any]:
    """
    从 mask_url 构建完整的 tools_kwargs 结构
    
    Args:
        mask_url: 需要 mask 的 URL，如果为空字符串则 tools_kwargs 中的 url 也为空
    
    Returns:
        完整的 tools_kwargs 字典
    """
    mask_url = mask_url if mask_url else ""
    
    return {
        'search': {
            'create_kwargs': {
                'url': mask_url
            }
        },
        'browse': {
            'create_kwargs': {
                'url': mask_url
            }
        }
    }


def convert_row(row: pd.Series) -> pd.Series:
    """
    转换单行数据，确保包含完整的 tools_kwargs 结构
    
    关键点：
    - 从 extra_info 中提取 mask_url
    - 构建完整的 tools_kwargs 结构，使 mask_url 能够被正确传递到工具
    """
    # 处理 reward_model
    gt = row['reward_model'].get('ground_truth', '') if isinstance(row['reward_model'], dict) else row['reward_model']
    
    # 从原始 extra_info 中提取信息
    original_extra_info = row.get('extra_info', {})
    if isinstance(original_extra_info, str):
        try:
            original_extra_info = json.loads(original_extra_info)
        except:
            original_extra_info = {}
    
    # 提取 mask_url（可能在不同位置）
    mask_url = ""
    if isinstance(original_extra_info, dict):
        # 优先从 mask_url 字段读取
        mask_url = original_extra_info.get('mask_url', '')
        # 如果没有 mask_url，尝试从 url 字段读取
        if not mask_url:
            mask_url = original_extra_info.get('url', '')
    
    # 构建完整的 extra_info，包含 tools_kwargs
    # 注意: index 统一转 str，避免 int/UUID 混合导致 pyarrow 序列化报错
    extra_info = {
        'index': str(original_extra_info.get('index', 0)),
        'mask_url': str(mask_url),
        'question': str(row.get('question', '')),
        'split': str(original_extra_info.get('split', 'train')),
        'need_tools_kwargs': True,
        'tools_kwargs': build_tools_kwargs(mask_url)
    }
    
    # 确保 prompt 格式正确
    prompt = row.get('prompt', [])
    if isinstance(prompt, np.ndarray):
        prompt_list = prompt.tolist()
    elif isinstance(prompt, list):
        prompt_list = prompt
    else:
        prompt_list = [prompt]
    
    # 检查是否已有 system prompt
    has_system = any(
        isinstance(msg, dict) and msg.get('role') == 'system' 
        for msg in prompt_list
    )
    
    if not has_system:
        # 添加 system prompt
        prompt_list = [
            {'content': SYSTEM_PROMPT, 'role': 'system'},
            {'content': f" Question: {row.get('question', '')}", 'role': 'user'}
        ]
    else:
        # 确保 user prompt 格式正确
        if len(prompt_list) >= 2:
            user_msg = prompt_list[1]
            if isinstance(user_msg, dict) and not user_msg.get('content', '').startswith(' Question:'):
                prompt_list[1] = {'content': f" Question: {row.get('question', '')}", 'role': 'user'}
    
    return pd.Series({
        'question': row.get('question', ''),
        'data_source': row.get('data_source', 'rag_direct'),
        'prompt': np.array(prompt_list, dtype=object),
        'ability': row.get('ability', 'search'),
        'reward_model': convert_reward_model(gt),
        'extra_info': extra_info,
        'metadata': row.get('metadata', None)
    })


def extract_data_source_from_path(filepath: Path) -> str:
    """
    从文件路径提取 data_source（使用文件名，去掉扩展名）
    
    Args:
        filepath: 文件路径
    
    Returns:
        data_source 字符串
    """
    # 获取文件名（不含扩展名）
    filename = filepath.stem
    return filename


def process_parquet_file(filepath: Path, data_source: str = None) -> pd.DataFrame:
    """
    处理单个 parquet 文件
    
    Args:
        filepath: parquet 文件路径
        data_source: 数据源名称，如果为 None 则从路径自动提取
    
    Returns:
        处理后的 DataFrame
    """
    if data_source is None:
        data_source = extract_data_source_from_path(filepath)
    
    print(f"  读取: {filepath.name}")
    print(f"    数据源: {data_source}")
    
    if not filepath.exists():
        print(f"  ⚠️  警告: 文件不存在，跳过")
        return None
    
    try:
        df = pd.read_parquet(filepath)
        print(f"    原始数据: {len(df)} 条")
        
        # 如果原数据没有 data_source 或需要覆盖，则设置
        if 'data_source' not in df.columns or data_source:
            df['data_source'] = data_source
        
        # 转换每一行
        df_converted = df.apply(convert_row, axis=1)
        
        print(f"    转换后: {len(df_converted)} 条")
        
        # 统计 mask_url 和 tools_kwargs
        mask_url_count = df_converted['extra_info'].apply(
            lambda x: bool(x.get('mask_url', ''))
        ).sum()
        tools_kwargs_count = df_converted['extra_info'].apply(
            lambda x: 'tools_kwargs' in x and bool(x.get('tools_kwargs', {}))
        ).sum()
        
        print(f"    包含 mask_url: {mask_url_count}/{len(df_converted)} ({mask_url_count/len(df_converted)*100:.1f}%)")
        print(f"    包含 tools_kwargs: {tools_kwargs_count}/{len(df_converted)} ({tools_kwargs_count/len(df_converted)*100:.1f}%)")
        
        return df_converted
        
    except Exception as e:
        print(f"  ❌ 错误: 处理文件时出错 - {e}")
        return None


def save_json_sample(df: pd.DataFrame, output_file: Path, sample_count: int = 10):
    """保存 JSON 样本"""
    def to_serializable(obj):
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, (pd._libs.tslibs.timestamps.Timestamp, pd.Timestamp)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj
    
    print(f"\n保存 {sample_count} 条样本到 JSON...")
    
    # 均匀采样
    if len(df) <= sample_count:
        indices = list(range(len(df)))
    else:
        step = len(df) // sample_count
        indices = [i * step for i in range(sample_count)]
        if indices[-1] != len(df) - 1:
            indices[-1] = len(df) - 1
    
    samples = []
    for i in indices:
        row = df.iloc[i]
        sample = {
            'question': row['question'],
            'data_source': row['data_source'],
            'prompt': to_serializable(row['prompt']),
            'ability': row['ability'],
            'reward_model': to_serializable(row['reward_model']),
            'extra_info': to_serializable(row['extra_info']),
            'metadata': row['metadata']
        }
        samples.append(sample)
    
    json_file = output_file.with_suffix('.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 已保存: {json_file}")


def merge_parquet_files(input_files: List[str], output_file: str, save_sample: bool = True, sample_count: int = 10):
    """
    合并多个 parquet 文件并正确处理 mask_url
    
    Args:
        input_files: 输入文件路径列表，如 ["path1.parquet", "path2.parquet"]
        output_file: 输出文件路径，如 "output.parquet"
        save_sample: 是否保存 JSON 样本（默认: True）
        sample_count: 样本数量（默认: 10）
    
    Returns:
        pd.DataFrame: 合并后的 DataFrame
    """
    print("=" * 80)
    print("合并 Parquet 文件并处理 mask_url")
    print("=" * 80)
    print()
    
    # 转换为 Path 对象
    input_paths = [Path(f) for f in input_files]
    output_path = Path(output_file)
    
    all_dfs = []
    
    # 处理每个输入文件（自动从路径提取 data_source）
    for filepath in input_paths:
        df = process_parquet_file(filepath, data_source=None)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
        print()
    
    if not all_dfs:
        print("❌ 错误: 没有成功处理任何文件！")
        return None
    
    # 合并所有数据
    print("=" * 80)
    print(f"合并 {len(all_dfs)} 个文件...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"✓ 合并完成，总数据量: {len(merged_df)} 条")
    print()
    
    # 最终统计
    mask_url_count = merged_df['extra_info'].apply(
        lambda x: bool(x.get('mask_url', ''))
    ).sum()
    tools_kwargs_count = merged_df['extra_info'].apply(
        lambda x: 'tools_kwargs' in x and bool(x.get('tools_kwargs', {}))
    ).sum()
    tools_kwargs_url_count = merged_df['extra_info'].apply(
        lambda x: x.get('tools_kwargs', {}).get('search', {}).get('create_kwargs', {}).get('url', '') != ''
    ).sum()
    
    print("最终统计:")
    print(f"  总数据量: {len(merged_df)} 条")
    print(f"  包含 mask_url: {mask_url_count} 条 ({mask_url_count/len(merged_df)*100:.1f}%)")
    print(f"  包含 tools_kwargs: {tools_kwargs_count} 条 ({tools_kwargs_count/len(merged_df)*100:.1f}%)")
    print(f"  tools_kwargs 中有 url: {tools_kwargs_url_count} 条 ({tools_kwargs_url_count/len(merged_df)*100:.1f}%)")
    print()
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存 parquet
    print(f"保存到: {output_path}")
    merged_df.to_parquet(output_path, index=False)
    file_size = output_path.stat().st_size / 1024 / 1024  # MB
    print(f"✓ 已保存，文件大小: {file_size:.2f} MB")
    print()
    
    # 保存 JSON 样本
    if save_sample:
        save_json_sample(merged_df, output_path, sample_count)
    
    print()
    print("=" * 80)
    print("✅ 处理完成！")
    print("=" * 80)
    print()
    print("关键点:")
    print("  • 所有数据都包含完整的 tools_kwargs 结构")
    print("  • mask_url 已正确传递到 tools_kwargs['search']['create_kwargs']['url']")
    print("  • mask_url 已正确传递到 tools_kwargs['browse']['create_kwargs']['url']")
    print("  • Google Search Tool 可以正确读取并使用 mask_url")
    print()
    
    return merged_df


if __name__ == "__main__":
    # 示例用法
    merge_parquet_files(
        input_files=[
            "/share/project/wanli/data/full_mixture/stage_3_v5/rag_direct_1-12_correct.parquet",
            "/share/project/wanli/data/full_mixture/stage_3_v5/rag_direct_1-7_correct.parquet",
        ],
        output_file="/share/project/wanli/Search_Agent/verl/data/deepresearch_rl/stage3_v5/stage3_v5.parquet"
    )
    