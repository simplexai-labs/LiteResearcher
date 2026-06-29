#!/usr/bin/env python3
"""合并 stage2 的 rag_direct 和 rag_mqa 数据"""
import json, os
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/share/project/wanli/data/full_mixture/stage_2")
REF_DATA = "/share/project/wanli/Search_Agent/verl/data/deepresearch_rl/stage2_final/stage2_rag_only.parquet"
OUTPUT = "/share/project/wanli/Search_Agent/verl/data/deepresearch_rl/only_local_rag_stage2/only_local_rag_stage2.parquet"

# 获取 system prompt
ref_df = pd.read_parquet(REF_DATA)
SYSTEM_PROMPT = ref_df.iloc[0]['prompt'][0]['content']

def convert_reward_model(gt):
    if isinstance(gt, str):
        return {'ground_truth': {'target': np.array([gt], dtype=object)}, 'style': 'llm'}
    elif isinstance(gt, dict) and 'target' in gt:
        return {'ground_truth': gt, 'style': 'llm'}
    return {'ground_truth': {'target': np.array([str(gt)], dtype=object)}, 'style': 'llm'}

def convert_row(row):
    gt = row['reward_model'].get('ground_truth', '') if isinstance(row['reward_model'], dict) else row['reward_model']
    return pd.Series({
        'question': row['question'],
        'data_source': row['data_source'],
        'prompt': np.array([
            {'content': SYSTEM_PROMPT, 'role': 'system'},
            {'content': f" Question: {row['question']}", 'role': 'user'}
        ], dtype=object),
        'ability': row.get('ability', 'search'),
        'reward_model': convert_reward_model(gt),
        'extra_info': {'index': 0, 'mask_url': '', 'split': 'train', 'question': row['question']},
        'metadata': None
    })

# 加载所有文件
files = [
    "rag_direct_1-7_lt16k.parquet", "rag_direct_1-7_16k-32k.parquet", 
    "rag_direct_1-7_32k-48k.parquet", "rag_direct_1-7_48k-64k.parquet",
    "rag_mqa_1-7_lt16k.parquet", "rag_mqa_1-7_16k-32k.parquet",
    "rag_mqa_1-7_32k-48k.parquet", "rag_mqa_1-7_48k-64k.parquet"
]

dfs = []
for f in files:
    print(f"Loading {f}")
    df = pd.read_parquet(BASE_DIR / f)
    dfs.append(df.apply(convert_row, axis=1))

df_final = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(df_final)} samples")

df_final.to_parquet(OUTPUT, index=False)
print(f"Saved to {OUTPUT}")

# 验证
df_check = pd.read_parquet(OUTPUT)
print(f"\ndata_source 分布:\n{df_check['data_source'].value_counts()}")

# 保存示例
samples_dir = os.path.dirname(OUTPUT) + "/samples"
os.makedirs(samples_dir, exist_ok=True)

def to_serializable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_serializable(v) for v in obj]
    return obj

for ds in df_check['data_source'].unique():
    sample = to_serializable(df_check[df_check['data_source'] == ds].iloc[0].to_dict())
    with open(f"{samples_dir}/{ds.replace('/', '_')}.json", 'w', encoding='utf-8') as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

print(f"示例已保存到 {samples_dir}/")
