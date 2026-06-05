#!/usr/bin/env python3
"""合并 rag_science 数据为 stage2_science.parquet（排除 rag_science_0）"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/share/project/wanli/data/full_mixture/stage_2")
REF_DATA = "/share/project/wanli/Search_Agent/verl/data/deepresearch_rl/stage2_final/stage2_rag_only.parquet"
OUTPUT = "/share/project/wanli/Search_Agent/verl/data/deepresearch_rl/only_local_rag_stage2/stage2_science.parquet"

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
    """转换单行数据，包含完整的 tools_kwargs 结构（mask_url 为空）"""
    gt = row['reward_model'].get('ground_truth', '') if isinstance(row['reward_model'], dict) else row['reward_model']
    
    # 从原始 extra_info 中提取信息
    original_extra_info = row.get('extra_info', {})
    if isinstance(original_extra_info, str):
        original_extra_info = json.loads(original_extra_info)
    
    # Science 数据的 mask_url 为空
    mask_url = ''
    
    # 构建完整的 extra_info，包含 tools_kwargs
    extra_info = {
        'index': original_extra_info.get('index', 0),
        'mask_url': mask_url,
        'question': row['question'],
        'split': original_extra_info.get('split', 'train'),
        'need_tools_kwargs': True,
        'tools_kwargs': {
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
    }
    
    return pd.Series({
        'question': row['question'],
        'data_source': row['data_source'],
        'prompt': np.array([
            {'content': SYSTEM_PROMPT, 'role': 'system'},
            {'content': f" Question: {row['question']}", 'role': 'user'}
        ], dtype=object),
        'ability': row.get('ability', 'search'),
        'reward_model': convert_reward_model(gt),
        'extra_info': extra_info,
        'metadata': None
    })

# 加载所有 rag_science 文件（排除 rag_science_0）
files = [
    f"rag_science_{i}.parquet" for i in range(1, 8)
]

dfs = []
for f in files:
    file_path = BASE_DIR / f
    if file_path.exists():
        print(f"Loading {f}")
        df = pd.read_parquet(file_path)
        dfs.append(df.apply(convert_row, axis=1))
    else:
        print(f"⚠️  {f} 不存在，跳过")

if not dfs:
    print("❌ 没有找到任何文件！")
    exit(1)

df_final = pd.concat(dfs, ignore_index=True)
print(f"\n✅ Total: {len(df_final)} samples")

df_final.to_parquet(OUTPUT, index=False)
print(f"✅ Saved to {OUTPUT}")

# 验证
df_check = pd.read_parquet(OUTPUT)
print(f"\n📊 data_source 分布:\n{df_check['data_source'].value_counts()}")

# 验证 tools_kwargs
tools_kwargs_count = df_check['extra_info'].apply(lambda x: 'tools_kwargs' in x).sum()
print(f"\n✅ tools_kwargs 存在数量: {tools_kwargs_count}/{len(df_check)}")
