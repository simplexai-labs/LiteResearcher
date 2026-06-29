#!/usr/bin/env bash
set -euo pipefail

# Disable proxy to avoid routing through unreachable local proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

PYTHON_BIN="${PYTHON_BIN:-python3}"
LLM_WORKERS=128
QUEUE_SIZE=512
FETCH_DELAY=0.1
LLM_HOST="127.0.0.1"
LLM_PORT=8000
LLM_API_KEY="${LLM_API_KEY:-}"
LLM_MODEL="${LLM_MODEL:-}"
HTML_HOST="${HTML_HOST:-127.0.0.1}"
HTML_PORT=8990
START_INDEX=0
END_INDEX=667648
OUTPUT_DIR="qa_outputs"
RESUME_DIR=""  # Empty by default; do not resume automatically
WIKI_MODE=""
WIKI_DIR="wiki_en"
WIKI_MAX_FILES=""
WIKI_HISTORY_MODE=""
WIKI_REVISIONS_DIR="wiki_revisions"
BBC_MODE=""
BBC_DIR="${BBC_DIR:-bbc_news_alltime}"
BBC_LIMIT=""
RANDOM_INDEX=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./run_threaded_qa.sh [options]
  --llm-workers N        Number of LLM processing workers (default: 128)
  --queue-size N         Max size of HTML task queue (default: 512)
  --fetch-delay SECS     Delay between HTML fetches in seconds (default: 0.1)
  --llm-host HOST        Host for the local LLM API server (default: 127.0.0.1)
  --llm-port PORT        Port for the local LLM API server (default: 8000)
  --llm-api-key KEY      API key/password configured on the LLM server (required)
  --llm-model NAME       Model name registered with the LLM server
  --html-host HOST       Public HTML service host (default: 47.111.147.142)
  --html-port PORT       Public HTML service port (default: 8990)
  --start-index N        Starting row index (default: 0)
  --end-index N          Ending row index inclusive (default: 667648)
  --output-dir PATH      Directory for QA output JSON files
  --resume-dir PATH      Resume from this output directory
  --wiki                 Use local wiki arrow files instead of HTTP service
  --wiki-dir PATH        Directory containing wiki arrow files (default: wiki_en)
  --wiki-max-files N     Max number of arrow files to load (default: all)
  --wiki-history         Process historical Wikipedia revisions
  --wiki-revisions-dir PATH  Directory containing revision JSON files (default: wiki_revisions)
  --bbc                  Use BBC news parquet files
  --bbc-dir PATH         Directory containing BBC news parquet files
  --bbc-limit N          Limit number of BBC articles to process
  --random-index         Randomly shuffle processing order
  --extra ARG            Additional arguments passed through to the Python script (repeatable)
  -h, --help             Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llm-workers)
      LLM_WORKERS="$2"; shift 2 ;;
    --queue-size)
      QUEUE_SIZE="$2"; shift 2 ;;
    --fetch-delay)
      FETCH_DELAY="$2"; shift 2 ;;
    --llm-host)
      LLM_HOST="$2"; shift 2 ;;
    --llm-port)
      LLM_PORT="$2"; shift 2 ;;
    --llm-api-key)
      LLM_API_KEY="$2"; shift 2 ;;
    --llm-model)
      LLM_MODEL="$2"; shift 2 ;;
    --html-host)
      HTML_HOST="$2"; shift 2 ;;
    --html-port)
      HTML_PORT="$2"; shift 2 ;;
    --start-index)
      START_INDEX="$2"; shift 2 ;;
    --end-index)
      END_INDEX="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --resume-dir)
      RESUME_DIR="$2"; shift 2 ;;
    --wiki)
      WIKI_MODE="--wiki"; shift ;;
    --wiki-dir)
      WIKI_DIR="$2"; shift 2 ;;
    --wiki-max-files)
      WIKI_MAX_FILES="$2"; shift 2 ;;
    --wiki-history)
      WIKI_HISTORY_MODE="--wiki-history"; shift ;;
    --wiki-revisions-dir)
      WIKI_REVISIONS_DIR="$2"; shift 2 ;;
    --bbc)
      BBC_MODE="--bbc"; shift ;;
    --bbc-dir)
      BBC_DIR="$2"; shift 2 ;;
    --bbc-limit)
      BBC_LIMIT="$2"; shift 2 ;;
    --random-index)
      RANDOM_INDEX="--random-index"; shift ;;
    --extra)
      EXTRA_ARGS+=("$2"); shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ -z "$LLM_API_KEY" ]]; then
  echo "Error: --llm-api-key is required, or set LLM_API_KEY" >&2
  exit 1
fi

if [[ -z "$LLM_MODEL" ]]; then
  echo "Error: --llm-model is required, or set LLM_MODEL" >&2
  exit 1
fi

if [[ -n "$WIKI_HISTORY_MODE" && -z "${SCRAPEDO_API_KEY:-}" ]]; then
  echo "Error: SCRAPEDO_API_KEY is required for --wiki-history" >&2
  exit 1
fi

# Build command arguments
CMD_ARGS=(
  --start-index "$START_INDEX"
  --end-index "$END_INDEX"
  --llm-workers "$LLM_WORKERS"
  --queue-size "$QUEUE_SIZE"
  --fetch-delay "$FETCH_DELAY"
  --llm-host "$LLM_HOST"
  --llm-port "$LLM_PORT"
  --llm-api-key "$LLM_API_KEY"
  --llm-model-name "$LLM_MODEL"
  --output-dir "$OUTPUT_DIR"
)

# Add mode-specific arguments
if [[ -n "$BBC_MODE" ]]; then
  CMD_ARGS+=("$BBC_MODE")
  CMD_ARGS+=(--bbc-dir "$BBC_DIR")
  if [[ -n "$BBC_LIMIT" ]]; then
    CMD_ARGS+=(--bbc-limit "$BBC_LIMIT")
  fi
elif [[ -n "$WIKI_MODE" ]]; then
  CMD_ARGS+=("$WIKI_MODE")
  CMD_ARGS+=(--wiki-dir "$WIKI_DIR")
  if [[ -n "$WIKI_MAX_FILES" ]]; then
    CMD_ARGS+=(--wiki-max-files "$WIKI_MAX_FILES")
  fi
else
  # Add HTML service arguments only in HTTP mode
  CMD_ARGS+=(
    --html-host "$HTML_HOST"
    --html-port "$HTML_PORT"
    --html-timeout 90
    --html-retries 3
  )
fi

# Add resume if specified
if [[ -n "$RESUME_DIR" ]]; then
  CMD_ARGS+=(--resume "$RESUME_DIR")
fi

# Add random index if specified
if [[ -n "$RANDOM_INDEX" ]]; then
  CMD_ARGS+=("$RANDOM_INDEX")
fi

# Add extra arguments
CMD_ARGS+=("${EXTRA_ARGS[@]}")

# Choose script based on mode
if [[ -n "$WIKI_HISTORY_MODE" ]]; then
  # Wiki History mode - use different script
  "$PYTHON_BIN" scripts/wiki/threaded_wiki_history_qa_generator.py \
    --wiki-revisions-dir "$WIKI_REVISIONS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --llm-host "$LLM_HOST" \
    --llm-port "$LLM_PORT" \
    --llm-api-key "$LLM_API_KEY" \
    --llm-model-name "$LLM_MODEL" \
    --llm-workers "$LLM_WORKERS" \
    --start-index "$START_INDEX" \
    --end-index "$END_INDEX" \
    --scrapedo-api-key "$SCRAPEDO_API_KEY"
else
  # Normal mode
  "$PYTHON_BIN" scripts/generation/threaded_qa_batch_generator.py "${CMD_ARGS[@]}"
fi
