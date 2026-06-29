#!/usr/bin/env python3
"""
对 benchmark 数据进行采样和合并
- 超过 200 条的数据集：随机采样 200 条
- 不超过 200 条的数据集：保留全部
- 合并成一个 parquet 文件用于推理
"""

import pandas as pd
from pathlib import Path
import argparse

def sample_and_merge_benchmarks(
    input_dir: str,
    output_file: str,
    max_samples: int = 200,
    random_seed: int = 42
):
    """
    采样和合并 benchmark 数据

    Args:
        input_dir: 输入目录（包含各个 benchmark 的 parquet 文件）
        output_file: 输出文件路径
        max_samples: 每个 benchmark 的最大样本数
        random_seed: 随机种子（保证可复现）
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)

    # 获取所有 benchmark 文件
    files = sorted(input_dir.glob("*_test.parquet"))

    if not files:
        print(f"❌ 未找到任何 parquet 文件在: {input_dir}")
        return

    print("=" * 70)
    print(f"采样和合并 Benchmark 数据")
    print("=" * 70)
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_file}")
    print(f"最大样本数: {max_samples}")
    print(f"随机种子: {random_seed}")
    print("=" * 70)

    all_dfs = []
    total_original = 0
    total_sampled = 0

    for file in files:
        benchmark_name = file.stem.replace("_test", "")

        # 读取数据
        df = pd.read_parquet(file)
        num_original = len(df)
        total_original += num_original

        # 采样或保留全部
        if num_original > max_samples:
            df_sampled = df.sample(n=max_samples, random_state=random_seed)
            num_sampled = max_samples
            action = f"采样 {max_samples}/{num_original}"
        else:
            df_sampled = df
            num_sampled = num_original
            action = "保留全部"

        total_sampled += num_sampled

        print(f"{benchmark_name:20s}: {num_original:4d} 条 → {num_sampled:3d} 条 ({action})")

        all_dfs.append(df_sampled)

    # 合并所有数据
    print("=" * 70)
    print("合并数据中...")
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # 保存到文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_file, index=False)

    print(f"✅ 合并完成!")
    print("=" * 70)
    print(f"原始总样本数: {total_original}")
    print(f"采样后总数:   {total_sampled}")
    print(f"输出文件:     {output_file}")
    print(f"文件大小:     {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 70)

    # 显示数据源分布
    if "data_source" in merged_df.columns:
        print("\n数据源分布:")
        print(merged_df["data_source"].value_counts().sort_index())

    return merged_df


def main():
    parser = argparse.ArgumentParser(description="采样和合并 benchmark 数据")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/share/project/wanli/Search_Agent/verl/data/benchmarks_processed/individual",
        help="输入目录（默认: benchmarks_processed/individual）"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/share/project/wanli/Search_Agent/verl/data/benchmarks_processed/sampled_benchmarks_test.parquet",
        help="输出文件路径"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="每个 benchmark 的最大样本数（默认: 200）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )

    args = parser.parse_args()

    sample_and_merge_benchmarks(
        input_dir=args.input_dir,
        output_file=args.output_file,
        max_samples=args.max_samples,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
