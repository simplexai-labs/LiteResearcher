#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Merge all benchmark parquet files into a single file and create sample JSON files.
Also reorganizes the directory structure.
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_samples_to_json(df, benchmark_name, output_dir, n_samples=5):
    """
    Save first n samples from dataframe to JSON file for visualization.

    Args:
        df: DataFrame to extract samples from
        benchmark_name: Name of the benchmark
        output_dir: Directory to save JSON file
        n_samples: Number of samples to save (default: 5)
    """
    samples = []
    n_to_save = min(n_samples, len(df))

    for idx in range(n_to_save):
        row = df.iloc[idx]
        sample = {
            "data_source": row['data_source'],
            "prompt": row['prompt'],
            "ability": row['ability'],
            "agent_name": row['agent_name'],
            "reward_model": {
                "ground_truth": {
                    "target": row['reward_model']['ground_truth']['target'].tolist()
                    if isinstance(row['reward_model']['ground_truth']['target'], np.ndarray)
                    else row['reward_model']['ground_truth']['target']
                },
                "style": row['reward_model'].get('style', 'rule')
            },
            "extra_info": row['extra_info'],
            "metadata": row['metadata']
        }
        samples.append(sample)

    json_path = os.path.join(output_dir, f"{benchmark_name}_first_{n_to_save}_samples.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    logger.info(f"Saved {n_to_save} samples to {json_path}")
    return json_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge benchmark parquet files and create sample JSONs"
    )
    parser.add_argument(
        "--input_dir",
        default="/share/project/wanli/Search_Agent/verl/data/benchmarks_processed",
        help="Input directory containing individual benchmark parquet files"
    )
    parser.add_argument(
        "--output_dir",
        default="/share/project/wanli/Search_Agent/verl/data/benchmarks_processed",
        help="Output directory for merged file and subdirectories"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples to save as JSON for each benchmark"
    )

    args = parser.parse_args()

    # Define benchmarks
    benchmarks = ["GAIA", "GPQA", "HLE", "WebWalkerQA", "Browsecomp", "Xbench"]

    # Create subdirectory for individual benchmarks
    individual_dir = os.path.join(args.output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)

    # Create samples directory
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    all_dfs = []

    logger.info("="*60)
    logger.info("Processing individual benchmarks...")
    logger.info("="*60)

    for benchmark in benchmarks:
        parquet_file = os.path.join(args.input_dir, f"{benchmark}_test.parquet")

        if not os.path.exists(parquet_file):
            logger.warning(f"File not found: {parquet_file}, skipping...")
            continue

        # Read the parquet file
        df = pd.read_parquet(parquet_file)
        logger.info(f"\n{benchmark}: {len(df)} samples")

        # Save samples to JSON
        save_samples_to_json(df, benchmark, samples_dir, args.n_samples)

        # Move parquet to individual directory
        dest_path = os.path.join(individual_dir, f"{benchmark}_test.parquet")
        if parquet_file != dest_path:  # Only copy if not already in subdirectory
            shutil.copy2(parquet_file, dest_path)
            logger.info(f"Copied {benchmark}_test.parquet to individual/")

        # Add to list for merging
        all_dfs.append(df)

    if not all_dfs:
        logger.error("No benchmark files found!")
        return

    # Merge all benchmarks
    logger.info("\n" + "="*60)
    logger.info("Merging all benchmarks...")
    logger.info("="*60)

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Shuffle the merged dataframe for better distribution
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"\nTotal samples in merged dataset: {len(merged_df)}")
    logger.info("\nData source distribution:")
    print(merged_df['data_source'].value_counts())

    # Save merged parquet
    merged_path = os.path.join(args.output_dir, "all_benchmarks_test.parquet")
    merged_df.to_parquet(merged_path, index=False)
    logger.info(f"\nSaved merged dataset to: {merged_path}")

    # Save samples from merged dataset
    save_samples_to_json(merged_df, "all_benchmarks", samples_dir, args.n_samples)

    # Clean up original files in root directory (if they're not in individual/)
    logger.info("\n" + "="*60)
    logger.info("Cleaning up original files...")
    logger.info("="*60)

    for benchmark in benchmarks:
        original_file = os.path.join(args.input_dir, f"{benchmark}_test.parquet")
        individual_file = os.path.join(individual_dir, f"{benchmark}_test.parquet")

        if os.path.exists(original_file) and os.path.exists(individual_file) and original_file != individual_file:
            os.remove(original_file)
            logger.info(f"Removed original {benchmark}_test.parquet (now in individual/)")

    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"\nDirectory structure:")
    logger.info(f"  {args.output_dir}/")
    logger.info(f"    ├── all_benchmarks_test.parquet  ({len(merged_df)} samples)")
    logger.info(f"    ├── individual/")
    for benchmark in benchmarks:
        individual_file = os.path.join(individual_dir, f"{benchmark}_test.parquet")
        if os.path.exists(individual_file):
            df_size = len(pd.read_parquet(individual_file))
            logger.info(f"    │   ├── {benchmark}_test.parquet  ({df_size} samples)")
    logger.info(f"    └── samples/")
    logger.info(f"        ├── all_benchmarks_first_{args.n_samples}_samples.json")
    for benchmark in benchmarks:
        sample_file = os.path.join(samples_dir, f"{benchmark}_first_{args.n_samples}_samples.json")
        if os.path.exists(sample_file):
            logger.info(f"        ├── {benchmark}_first_{args.n_samples}_samples.json")

    logger.info("\n" + "="*60)
    logger.info("Done!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
