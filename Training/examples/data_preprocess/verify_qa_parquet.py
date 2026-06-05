#!/usr/bin/env python3
"""
Verification script for the generated QA parquet files.
"""

import pandas as pd
import argparse


def verify_parquet_file(parquet_path):
    """Verify the structure and content of the parquet file."""
    print(f"\n{'='*80}")
    print(f"Verifying: {parquet_path}")
    print('='*80)

    # Load the parquet file
    df = pd.read_parquet(parquet_path)

    print(f"\n✓ Total samples: {len(df)}")
    print(f"✓ Columns: {df.columns.tolist()}")

    # Check required fields
    required_fields = ['data_source', 'prompt', 'agent_name', 'reward_model', 'extra_info']
    for field in required_fields:
        assert field in df.columns, f"Missing required field: {field}"
    print(f"✓ All required fields present")

    # Verify prompt structure
    sample_prompt = df.iloc[0]['prompt']
    # Parquet may convert list to numpy array
    import numpy as np
    assert isinstance(sample_prompt, (list, np.ndarray)), "Prompt should be a list or array"
    assert len(sample_prompt) == 2, "Prompt should have 2 messages (system + user)"
    assert sample_prompt[0]['role'] == 'system', "First message should be system"
    assert sample_prompt[1]['role'] == 'user', "Second message should be user"
    print(f"✓ Prompt structure verified")

    # Verify extra_info structure
    sample_extra_info = df.iloc[0]['extra_info']
    required_extra_info_keys = ['index', 'need_tools_kwargs', 'question', 'split', 'tools_kwargs']
    for key in required_extra_info_keys:
        assert key in sample_extra_info, f"Missing key in extra_info: {key}"
    print(f"✓ extra_info structure verified")

    # Verify tools_kwargs
    tools_kwargs = sample_extra_info['tools_kwargs']
    assert 'search' in tools_kwargs, "Missing 'search' in tools_kwargs"
    assert 'browse' in tools_kwargs, "Missing 'browse' in tools_kwargs"
    print(f"✓ tools_kwargs structure verified")

    # Verify reward_model
    sample_reward = df.iloc[0]['reward_model']
    assert 'ground_truth' in sample_reward, "Missing ground_truth in reward_model"
    print(f"✓ reward_model structure verified")

    # Print sample data
    print(f"\n{'='*80}")
    print("Sample QA Pair:")
    print('='*80)
    sample = df.iloc[0]
    print(f"\nQuestion: {sample['extra_info']['question'][:200]}...")
    print(f"\nGround Truth: {sample['reward_model']['ground_truth']}")
    print(f"\nData Source: {sample['data_source']}")
    print(f"Agent Name: {sample['agent_name']}")
    print(f"Ability: {sample.get('ability', 'N/A')}")

    # Statistics
    print(f"\n{'='*80}")
    print("Statistics:")
    print('='*80)
    print(f"Data sources: {df['data_source'].unique().tolist()}")
    print(f"Agent names: {df['agent_name'].unique().tolist()}")

    # Check for any null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"\n⚠ Warning: Found null values:")
        print(null_counts[null_counts > 0])
    else:
        print(f"\n✓ No null values found")

    print(f"\n{'='*80}")
    print(f"✓ Verification complete for {parquet_path}")
    print('='*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Verify QA parquet files")
    parser.add_argument(
        "--train_file",
        default="data/deepresearch_qa_processed/train_with_tools.parquet",
        help="Path to training parquet file"
    )
    parser.add_argument(
        "--test_file",
        default="data/deepresearch_qa_processed/test_with_tools.parquet",
        help="Path to test parquet file"
    )

    args = parser.parse_args()

    # Verify training file
    verify_parquet_file(args.train_file)

    # Verify test file
    verify_parquet_file(args.test_file)

    print("\n✅ All verification checks passed!")


if __name__ == "__main__":
    main()
