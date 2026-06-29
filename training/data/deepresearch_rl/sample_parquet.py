#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 parquet 文件转换为 JSON（保留所有数据）

使用方法:
    from sample_parquet import parquet_to_json
    
    parquet_to_json(
        input_file="data.parquet",
        output_file="data.json"
    )
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Any


def to_serializable(obj: Any) -> Any:
    """转换为可序列化的对象"""
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


def parquet_to_json(input_file: str, output_file: str):
    """
    将 parquet 文件转换为 JSON（保留所有数据）
    
    Args:
        input_file: 输入 parquet 文件路径
        output_file: 输出 JSON 文件路径
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 读取 parquet
    df = pd.read_parquet(input_path)
    sample_count = len(df)
    
    print(f"读取文件: {input_file}")
    print(f"样本数量: {sample_count} 条")
    
    # 转换为 JSON（保留所有字段）
    samples = []
    for _, row in df.iterrows():
        sample = {k: to_serializable(v) for k, v in row.items()}
        samples.append(sample)
    
    # 保存 JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"保存到: {output_file}")
    print(f"✓ 完成，共 {sample_count} 条数据")


if __name__ == "__main__":
    # 示例用法
    parquet_to_json(
        input_file="data.parquet",
        output_file="data.json"
    )
