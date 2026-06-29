#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

./run_threaded_qa.sh --wiki \
  --start-index "${START_INDEX:-0}" \
  --end-index "${END_INDEX:-6407814}" \
  --wiki-dir "${WIKI_DIR:-wiki_en}" \
  --output-dir "${OUTPUT_DIR:-qa_outputs_wiki}" \
  --resume-dir "${RESUME_DIR:-qa_outputs_wiki}" \
  --random-index \
  --llm-workers "${LLM_WORKERS:-128}" \
  --queue-size "${QUEUE_SIZE:-512}"
