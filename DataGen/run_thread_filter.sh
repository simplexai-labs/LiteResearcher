#!/bin/bash

# run_thread_filter.sh - Run QA pair quality filtering with configurable workers
# Usage: ./run_thread_filter.sh [workers]

# Default number of workers
WORKERS=${1:-128}

# Input and output directories
INPUT_DIR="qa_outputs_bbc"
OUTPUT_DIR="output_filtered_bbc"
CHECKPOINT="filter_ckpt_bbc.txt"

echo "Starting QA pair filtering..."
echo "Workers: $WORKERS"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo ""

# Run the filter script
python3 scripts/filtering/filter_qa_pairs.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    --ckpt "$CHECKPOINT"

echo ""
echo "Filtering completed!"
