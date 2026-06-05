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
Process QA JSON files from deepResearchRepo into parquet format for VERL training.
Extracts question and answer fields from JSON files and formats them according to SearchR1 conventions.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# System prompt for search task
DEFAULT_SYSTEM_CONTENT = """
* You are a meticulous and effective AI Research Assistant.

The question I provide requires in-depth research to answer. Your task is to use the available tools to find the necessary information and then provide a conclusive answer.

You have two tools at your disposal:
* **search**: To find relevant webpages using a search query. Required parameter: query (string)
* **browse**: To visit a specific webpage link and extract key information based on an instruction. Required parameters: link (string) and what_to_find (string)

##CRITICAL RULES FOR TOOL USAGE##
1. For the **search** tool: Always provide a clear search query
2. For the **browse** tool:
   - The link argument MUST be an exact link copied from a previous <tool_response> from the search tool
   - The what_to_find argument MUST specify exactly what information you need from that webpage
   - NEVER invent, guess, or modify a link
   - ALWAYS include both link AND what_to_find parameters

Your thinking process should explicitly state which search result you are choosing and why, and what specific information you need to extract.

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
{"name": "tool_name", "arguments": {"parameter1": "value1", "parameter2": "value2"}}
</tool_call>

CRITICAL: Tool calls MUST use the exact format above with "arguments" wrapping the parameters.
Examples:
- Correct: {"name": "search", "arguments": {"query": "your search query"}}
- Correct: {"name": "browse", "arguments": {"link": "https://...", "what_to_find": "what you want to find"}}
- WRONG: {"name": "search", "query": "your search query"}
- WRONG: {"name": "browse", "link": "https://...", "what_to_find": "..."

You should always follow the above two formats strictly.
Only output the final answer (in words, numbers or phrase) inside the <answer></answer> tag, without any explanations or extra information.For example, <answer> Beijing </answer>.

"""

DEFAULT_USER_CONTENT_PREFIX = " Question: "


def load_json_files(input_dir):
    """
    Load all JSON files from the input directory.

    Args:
        input_dir: Path to directory containing JSON files

    Returns:
        List of dictionaries containing question, answer, and metadata
    """
    input_path = Path(input_dir)
    json_files = sorted(input_path.glob("*.json"))

    logger.info(f"Found {len(json_files)} JSON files in {input_dir}")

    data_list = []
    failed_files = []

    for json_file in tqdm(json_files, desc="Loading JSON files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract question and answer
            question = data.get("question", "")
            answer = data.get("answer", "")

            if not question or not answer:
                logger.warning(f"Missing question or answer in {json_file.name}")
                continue

            data_list.append({
                "question": question,
                "answer": answer,
                "file_name": json_file.name,
                "target_entity": data.get("target_entity_for_answer", ""),
                "metadata": {
                    "source_file": json_file.name,
                    "has_relationship_constrains": "relationship_constrains" in data,
                    "has_entity_constrains": "entity_constrains" in data,
                }
            })

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {json_file.name}: {e}")
            failed_files.append(json_file.name)
        except Exception as e:
            logger.error(f"Error processing {json_file.name}: {e}")
            failed_files.append(json_file.name)

    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files: {failed_files[:10]}")

    logger.info(f"Successfully loaded {len(data_list)} QA pairs")
    return data_list


def process_qa_data(data_list, system_content, user_content_prefix, data_source_name="deepResearch_qa"):
    """
    Process QA data into the format required for VERL training.

    Args:
        data_list: List of dictionaries containing question and answer
        system_content: System prompt content
        user_content_prefix: User content prefix
        data_source_name: Name to tag the data source

    Returns:
        DataFrame with processed data
    """
    processed_rows = []

    for idx, item in enumerate(tqdm(data_list, desc="Processing QA pairs")):
        question = item["question"]
        answer = item["answer"]

        # Build prompt structure
        user_content = user_content_prefix.rstrip("\n") + question
        prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        # Prepare ground truth (answer can be a list or string)
        if isinstance(answer, str):
            ground_truth = [answer]
        else:
            ground_truth = answer

        # Build reward model info (must match SearchR1 format with "target" key)
        reward_model_data = {
            "ground_truth": {
                "target": ground_truth
            },
            "style": "rule"
        }

        # Build shared create_kwargs for tools (ground_truth must have "target" key)
        shared_create_kwargs = {
            "ground_truth": {
                "target": ground_truth
            },
            "question": question,
            "data_source": data_source_name
        }

        # Build tools kwargs structure for both search and browse tools
        tools_kwargs = {
            "search": {"create_kwargs": shared_create_kwargs},
            "browse": {"create_kwargs": shared_create_kwargs}
        }

        # Build extra_info structure
        extra_info = {
            "index": idx,
            "need_tools_kwargs": True,
            "question": question,
            "split": "train",  # Can be overridden during train/test split
            "tools_kwargs": tools_kwargs,
        }

        processed_row = {
            "data_source": data_source_name,
            "prompt": prompt,
            "ability": "question_answering",  # Default ability
            "agent_name": "tool_agent",
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": item.get("metadata", {}),
        }

        processed_rows.append(processed_row)

    df = pd.DataFrame(processed_rows)

    # Shuffle the data before returning
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Created and shuffled DataFrame with {len(df)} rows")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Process QA JSON files into parquet format for VERL training"
    )
    parser.add_argument(
        "--input_dir",
        default="/share/project/wanli/Search_Agent/deepResearchRepo/deepResearchRepo/qaDatasets/searchr1_ans/set1/tmp",
        help="Directory containing JSON files with question and answer fields"
    )
    parser.add_argument(
        "--output_dir",
        default="data/deepresearch_qa_processed",
        help="Directory to save the processed parquet file"
    )
    parser.add_argument(
        "--output_name",
        default="train_with_tools.parquet",
        help="Name of the output parquet file"
    )
    parser.add_argument(
        "--data_source_name",
        default="deepResearch_qa",
        help="Name to tag the data source"
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=None,
        help="If specified, split data into train/test with this ratio (e.g., 0.9 for 90%% train)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON files
    data_list = load_json_files(args.input_dir)

    if not data_list:
        logger.error("No data loaded. Exiting.")
        return

    # Process data
    df_processed = process_qa_data(
        data_list,
        system_content=DEFAULT_SYSTEM_CONTENT,
        user_content_prefix=DEFAULT_USER_CONTENT_PREFIX,
        data_source_name=args.data_source_name
    )

    # Handle train/test split if requested
    if args.train_test_split is not None:
        if not (0 < args.train_test_split < 1):
            logger.error("train_test_split must be between 0 and 1")
            return

        # Shuffle and split
        df_shuffled = df_processed.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df_shuffled) * args.train_test_split)

        df_train = df_shuffled.iloc[:split_idx].copy()
        df_test = df_shuffled.iloc[split_idx:].copy()

        # Update split field in extra_info
        for idx in df_train.index:
            df_train.at[idx, 'extra_info']['split'] = 'train'
        for idx in df_test.index:
            df_test.at[idx, 'extra_info']['split'] = 'test'

        # Save train and test files
        train_path = os.path.join(output_dir, "train_with_tools.parquet")
        test_path = os.path.join(output_dir, "test_with_tools.parquet")

        df_train.to_parquet(train_path, index=False)
        df_test.to_parquet(test_path, index=False)

        logger.info(f"Saved {len(df_train)} training samples to {train_path}")
        logger.info(f"Saved {len(df_test)} test samples to {test_path}")
    else:
        # Save single file
        output_path = os.path.join(output_dir, args.output_name)
        df_processed.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df_processed)} samples to {output_path}")

    logger.info("Processing complete!")

    # Print sample data
    logger.info("\n" + "="*80)
    logger.info("Sample data (first row):")
    logger.info("="*80)
    sample = df_processed.iloc[0]
    logger.info(f"Data source: {sample['data_source']}")
    logger.info(f"Question: {sample['extra_info']['question'][:200]}...")
    logger.info(f"Ground truth: {sample['reward_model']['ground_truth']}")
    logger.info(f"Prompt structure: {[msg['role'] for msg in sample['prompt']]}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
