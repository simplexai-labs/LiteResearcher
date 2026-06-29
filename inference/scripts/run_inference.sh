#!/bin/bash
# Run inference only (assumes servers are already running)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$RELEASE_DIR"

# Load .env if exists
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Default values
MODEL="${MODEL:-}"
DATASET="${DATASET:-data/example.jsonl}"
OUTPUT="${OUTPUT:-./output}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.1}"
MAX_WORKERS="${MAX_WORKERS:-20}"
ROLL_OUT_COUNT="${ROLL_OUT_COUNT:-1}"

if [ -z "$MODEL" ]; then
    echo "Error: MODEL not set. Set MODEL env var or add to .env"
    echo "  Example: MODEL=/path/to/model or MODEL=your-model-name"
    exit 1
fi

echo "=========================================="
echo "  Running Inference"
echo "=========================================="
echo "  Model:       $MODEL"
echo "  Dataset:     $DATASET"
echo "  Output:      $OUTPUT"
echo "  Temperature: $TEMPERATURE"
echo "  Workers:     $MAX_WORKERS"
echo "  Rollouts:    $ROLL_OUT_COUNT"
echo "=========================================="

python -m src.run_inference \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --presence_penalty "$PRESENCE_PENALTY" \
    --max_workers "$MAX_WORKERS" \
    --roll_out_count "$ROLL_OUT_COUNT" \
    "$@"
