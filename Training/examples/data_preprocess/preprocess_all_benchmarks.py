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
Preprocess all benchmarks from deepResearchRepo to verl-compatible parquet format.
This script processes GAIA, GPQA, HLE, WebWalkerQA, Browsecomp, and Xbench.
Matches the exact format of searchR1_processed_with_dual_tools.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# System prompt for dual-tool (search + browse) setup
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


def load_json_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_standard_entry(question, answers, data_source, idx, ability, metadata_dict=None):
    """
    Create a standard data entry matching searchR1 format exactly.

    Args:
        question: The question string
        answers: List of answer strings (will be converted to numpy array)
        data_source: Source identifier (e.g., "GAIA", "GPQA")
        idx: Index of the entry
        ability: Ability category string
        metadata_dict: Optional metadata dict (will be set to None if not provided)
    """
    # Ensure answers is a list
    if isinstance(answers, str):
        answers = [answers]

    # Convert to numpy array like searchR1
    answers_array = np.array(answers, dtype=object)

    # Build prompt
    user_content = DEFAULT_USER_CONTENT_PREFIX.rstrip("\n") + question
    prompt = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": user_content}
    ]

    # Build ground_truth structure exactly like searchR1
    ground_truth = {"target": answers_array}

    # Build shared create_kwargs for both tools
    shared_create_kwargs = {
        "data_source": data_source,
        "ground_truth": ground_truth,
        "question": question
    }

    # Build tools kwargs (search and browse)
    tools_kwargs = {
        "browse": {"create_kwargs": shared_create_kwargs},
        "search": {"create_kwargs": shared_create_kwargs}
    }

    # Build extra_info (no additional fields beyond the standard ones)
    extra_info = {
        "index": idx,
        "need_tools_kwargs": True,
        "question": question,
        "split": "test",
        "tools_kwargs": tools_kwargs
    }

    # Build reward_model exactly like searchR1
    reward_model = {
        "ground_truth": ground_truth,
        "style": "rule"
    }

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": ability,
        "agent_name": "tool_agent",
        "reward_model": reward_model,
        "extra_info": extra_info,
        "metadata": metadata_dict  # None or dict with metadata
    }


def process_gaia(benchmark_dir, output_dir):
    """Process GAIA benchmark."""
    logger.info("Processing GAIA benchmark...")
    file_path = os.path.join(benchmark_dir, "GAIA", "dev.json")

    if not os.path.exists(file_path):
        logger.warning(f"GAIA file not found: {file_path}")
        return None

    data = load_json_data(file_path)
    processed_data = []

    for idx, item in enumerate(data):
        question = item.get("Question", "")
        answer = item.get("answer", "")

        entry = create_standard_entry(
            question=question,
            answers=[answer],
            data_source="GAIA",
            idx=idx,
            ability="fact-reasoning",
            metadata_dict=None  # Match searchR1 format
        )
        processed_data.append(entry)

    df = pd.DataFrame(processed_data)
    output_path = os.path.join(output_dir, "GAIA_test.parquet")
    df.to_parquet(output_path, index=False)
    logger.info(f"GAIA: Saved {len(df)} samples to {output_path}")
    return output_path


def process_gpqa(benchmark_dir, output_dir):
    """Process GPQA benchmark."""
    logger.info("Processing GPQA benchmark...")
    file_path = os.path.join(benchmark_dir, "GPQA", "diamond.json")

    if not os.path.exists(file_path):
        logger.warning(f"GPQA file not found: {file_path}")
        return None

    data = load_json_data(file_path)
    processed_data = []

    for idx, item in enumerate(data):
        question = item.get("Question", "")
        answer = item.get("Correct Answer", "")

        entry = create_standard_entry(
            question=question,
            answers=[answer],
            data_source="GPQA",
            idx=idx,
            ability="fact-reasoning",
            metadata_dict=None
        )
        processed_data.append(entry)

    df = pd.DataFrame(processed_data)
    output_path = os.path.join(output_dir, "GPQA_test.parquet")
    df.to_parquet(output_path, index=False)
    logger.info(f"GPQA: Saved {len(df)} samples to {output_path}")
    return output_path


def process_hle(benchmark_dir, output_dir):
    """Process HLE benchmark."""
    logger.info("Processing HLE benchmark...")
    file_path = os.path.join(benchmark_dir, "HLE", "test.json")

    if not os.path.exists(file_path):
        logger.warning(f"HLE file not found: {file_path}")
        return None

    data = load_json_data(file_path)
    processed_data = []

    for idx, item in enumerate(data):
        question = item.get("Question", "")  # HLE uses capital Q
        answer = item.get("answer", "")

        entry = create_standard_entry(
            question=question,
            answers=[answer],
            data_source="HLE",
            idx=idx,
            ability="fact-reasoning",
            metadata_dict=None
        )
        processed_data.append(entry)

    df = pd.DataFrame(processed_data)
    output_path = os.path.join(output_dir, "HLE_test.parquet")
    df.to_parquet(output_path, index=False)
    logger.info(f"HLE: Saved {len(df)} samples to {output_path}")
    return output_path


def process_webwalkerqa(benchmark_dir, output_dir):
    """Process WebWalkerQA benchmark."""
    logger.info("Processing WebWalkerQA benchmark...")
    file_path = os.path.join(benchmark_dir, "WebWalkerQA", "test.json")

    if not os.path.exists(file_path):
        logger.warning(f"WebWalkerQA file not found: {file_path}")
        return None

    data = load_json_data(file_path)
    processed_data = []

    for idx, item in enumerate(data):
        question = item.get("Question", "")
        answer = item.get("answer", "")

        entry = create_standard_entry(
            question=question,
            answers=[answer],
            data_source="WebWalkerQA",
            idx=idx,
            ability="fact-reasoning",
            metadata_dict=None
        )
        processed_data.append(entry)

    df = pd.DataFrame(processed_data)
    output_path = os.path.join(output_dir, "WebWalkerQA_test.parquet")
    df.to_parquet(output_path, index=False)
    logger.info(f"WebWalkerQA: Saved {len(df)} samples to {output_path}")
    return output_path


def process_browsecomp(benchmark_dir, output_dir):
    """Process Browsecomp benchmark."""
    logger.info("Processing Browsecomp benchmark...")
    file_path = os.path.join(benchmark_dir, "Browsecomp", "browsecomp.json")

    if not os.path.exists(file_path):
        logger.warning(f"Browsecomp file not found: {file_path}")
        return None

    data = load_json_data(file_path)
    processed_data = []

    for idx, item in enumerate(data):
        question = item.get("problem", "")  # Browsecomp uses 'problem' not 'question'
        answer = item.get("answer", "")

        entry = create_standard_entry(
            question=question,
            answers=[answer],
            data_source="Browsecomp",
            idx=idx,
            ability="fact-reasoning",
            metadata_dict=None
        )
        processed_data.append(entry)

    df = pd.DataFrame(processed_data)
    output_path = os.path.join(output_dir, "Browsecomp_test.parquet")
    df.to_parquet(output_path, index=False)
    logger.info(f"Browsecomp: Saved {len(df)} samples to {output_path}")
    return output_path


def process_xbench(benchmark_dir, output_dir):
    """Process Xbench benchmark."""
    logger.info("Processing Xbench benchmark...")
    file_path = os.path.join(benchmark_dir, "Xbench", "qa_pairs.json")

    if not os.path.exists(file_path):
        logger.warning(f"Xbench file not found: {file_path}")
        return None

    data = load_json_data(file_path)
    processed_data = []

    for idx, item in enumerate(data):
        question = item.get("question", "")  # Xbench uses lowercase
        answer = item.get("answer", "")  # Xbench uses lowercase

        entry = create_standard_entry(
            question=question,
            answers=[answer],
            data_source="Xbench",
            idx=idx,
            ability="fact-reasoning",
            metadata_dict=None
        )
        processed_data.append(entry)

    df = pd.DataFrame(processed_data)
    output_path = os.path.join(output_dir, "Xbench_test.parquet")
    df.to_parquet(output_path, index=False)
    logger.info(f"Xbench: Saved {len(df)} samples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess all benchmarks to verl-compatible format")
    parser.add_argument(
        "--benchmark_dir",
        default="/share/project/wanli/Search_Agent/deepResearchRepo/deepResearchRepo/benchmark",
        help="Directory containing all benchmark subdirectories"
    )
    parser.add_argument(
        "--output_dir",
        default="/share/project/wanli/Search_Agent/verl/data/benchmarks_processed",
        help="Output directory for processed parquet files"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["GAIA", "GPQA", "HLE", "WebWalkerQA", "Browsecomp", "Xbench"],
        choices=["GAIA", "GPQA", "HLE", "WebWalkerQA", "Browsecomp", "Xbench"],
        help="Which benchmarks to process (default: all)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Process each benchmark
    processors = {
        "GAIA": process_gaia,
        "GPQA": process_gpqa,
        "HLE": process_hle,
        "WebWalkerQA": process_webwalkerqa,
        "Browsecomp": process_browsecomp,
        "Xbench": process_xbench
    }

    processed_files = []
    for benchmark in args.benchmarks:
        if benchmark in processors:
            try:
                output_path = processors[benchmark](args.benchmark_dir, args.output_dir)
                if output_path:
                    processed_files.append(output_path)
            except Exception as e:
                logger.error(f"Error processing {benchmark}: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning(f"Unknown benchmark: {benchmark}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"Total files processed: {len(processed_files)}")
    for file_path in processed_files:
        df = pd.read_parquet(file_path)
        logger.info(f"  - {os.path.basename(file_path)}: {len(df)} samples")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
