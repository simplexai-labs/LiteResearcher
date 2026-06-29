#!/bin/bash
# Start SGLang inference server (example)
# Adjust --tp, --dp, --model-path to your setup
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$RELEASE_DIR"

# Load .env if exists
if [ -f .env ]; then
    set -a; source .env; set +a
    echo "✓ Loaded .env"
fi

MODEL_PATH="${SGLANG_MODEL_PATH:-/path/to/your/model}"
PORT="${PLANNING_PORTS:-6001}"
TP="${SGLANG_TP:-1}"
DP="${SGLANG_DP:-1}"
MAX_MODEL_LEN="${MAIN_MAX_MODEL_LEN:-90000}"

echo "=========================================="
echo "  Starting SGLang Server"
echo "=========================================="
echo "  Model:     $MODEL_PATH"
echo "  Port:      $PORT"
echo "  TP:        $TP"
echo "  DP:        $DP"
echo "  Max len:   $MAX_MODEL_LEN"
echo "=========================================="

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --tp "$TP" \
    --dp "$DP" \
    --max-total-tokens "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --enable-thinking \
    --thinking-budget 10000
