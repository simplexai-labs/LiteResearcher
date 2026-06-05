#!/usr/bin/env python3
"""
将 GAIA.jsonl 和 Xbench.jsonl 转换为与 stage2_rag_only.parquet 相同格式的 parquet 文件
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

# System prompt - 与 stage2_rag_only 保持一致
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


def load_jsonl(file_path):
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_prompt(question):
    """创建与 stage2_rag_only 相同格式的 prompt"""
    return [
        {
            'content': SYSTEM_PROMPT,
            'role': 'system'
        },
        {
            'content': f' Question: {question}',
            'role': 'user'
        }
    ]


def convert_to_parquet_format(jsonl_data, data_source_name):
    """
    将 JSONL 数据转换为 parquet 格式
    
    Args:
        jsonl_data: JSONL 数据列表
        data_source_name: 数据源名称 (GAIA 或 Xbench)
    """
    converted_data = []
    
    for idx, item in enumerate(jsonl_data):
        question = item['question']
        answer = item['answer']
        
        record = {
            'question': question,
            'data_source': data_source_name,  # 使用传入的数据源名称
            'prompt': create_prompt(question),
            'ability': 'search',
            'reward_model': {
                'ground_truth': {
                    'target': np.array([str(answer)], dtype=object)
                },
                'style': 'llm'
            },
            'extra_info': {
                'index': idx,
                'split': 'test',  # 评估数据标记为 test
                'original_id': item.get('id', idx)
            },
            'metadata': None
        }
        converted_data.append(record)
    
    return pd.DataFrame(converted_data)


def visualize_samples(df, output_path, num_samples=3):
    """可视化前几个样本并保存为 JSON"""
    samples = []
    for i in range(min(num_samples, len(df))):
        sample = {}
        for col in df.columns:
            value = df.iloc[i][col]
            # 转换 numpy array 为 list
            if isinstance(value, np.ndarray):
                value = value.tolist()
            # 转换 dict 中的 numpy array
            elif isinstance(value, dict):
                value = convert_numpy_in_dict(value)
            sample[col] = value
        samples.append(sample)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已保存 {len(samples)} 个样本到: {output_path}")


def convert_numpy_in_dict(d):
    """递归转换字典中的 numpy 类型"""
    if isinstance(d, dict):
        return {k: convert_numpy_in_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_numpy_in_dict(item) for item in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(d)
    elif isinstance(d, (np.float_, np.float16, np.float32, np.float64)):
        return float(d)
    elif isinstance(d, np.bool_):
        return bool(d)
    return d


def main():
    # 输入文件路径
    gaia_jsonl = '/share/project/wanli/Search_Agent/DeepResearch/inference/eval_data/benchmarks/GAIA.jsonl'
    xbench_jsonl = '/share/project/wanli/Search_Agent/DeepResearch/inference/eval_data/benchmarks/Xbench.jsonl'
    
    # 输出文件夹
    output_dir = Path('/share/project/wanli/Search_Agent/verl/data/eval_benchmarks')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("开始转换评估数据集...")
    print("=" * 60)
    
    # 处理 GAIA
    print("\n处理 GAIA.jsonl...")
    gaia_data = load_jsonl(gaia_jsonl)
    gaia_df = convert_to_parquet_format(gaia_data, 'GAIA')
    
    gaia_parquet_path = output_dir / 'GAIA_test.parquet'
    gaia_df.to_parquet(gaia_parquet_path, index=False)
    print(f"✅ GAIA 转换完成: {len(gaia_df)} 条数据")
    print(f"   保存至: {gaia_parquet_path}")
    
    # 可视化 GAIA 样本
    gaia_json_path = output_dir / 'GAIA_test_samples.json'
    visualize_samples(gaia_df, gaia_json_path, num_samples=3)
    
    # 处理 Xbench
    print("\n处理 Xbench.jsonl...")
    xbench_data = load_jsonl(xbench_jsonl)
    xbench_df = convert_to_parquet_format(xbench_data, 'Xbench')
    
    xbench_parquet_path = output_dir / 'Xbench_test.parquet'
    xbench_df.to_parquet(xbench_parquet_path, index=False)
    print(f"✅ Xbench 转换完成: {len(xbench_df)} 条数据")
    print(f"   保存至: {xbench_parquet_path}")
    
    # 可视化 Xbench 样本
    xbench_json_path = output_dir / 'Xbench_test_samples.json'
    visualize_samples(xbench_df, xbench_json_path, num_samples=3)
    
    # 处理 stage2_rag_only.parquet 的可视化
    print("\n处理 stage2_rag_only.parquet 可视化...")
    stage2_path = '/share/project/wanli/Search_Agent/verl/data/deepresearch_rl/stage2_final/stage2_rag_only.parquet'
    stage2_df = pd.read_parquet(stage2_path)
    
    stage2_json_path = '/share/project/wanli/Search_Agent/verl/data/deepresearch_rl/stage2_final/stage2_rag_only_samples.json'
    visualize_samples(stage2_df, stage2_json_path, num_samples=3)
    
    print("\n" + "=" * 60)
    print("所有转换完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  1. {gaia_parquet_path}")
    print(f"  2. {gaia_json_path}")
    print(f"  3. {xbench_parquet_path}")
    print(f"  4. {xbench_json_path}")
    print(f"  5. {stage2_json_path}")


if __name__ == '__main__':
    main()
