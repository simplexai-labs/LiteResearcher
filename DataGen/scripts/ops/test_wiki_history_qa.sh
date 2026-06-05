#!/usr/bin/env bash
set -euo pipefail

TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-qa_outputs_wiki_history_test}"
WIKI_REVISIONS_DIR="${WIKI_REVISIONS_DIR:-wiki_revisions}"

if [[ -z "${LLM_API_KEY:-}" ]]; then
  echo "Error: set LLM_API_KEY or pass --llm-api-key through run_threaded_qa.sh" >&2
  exit 1
fi

if [[ -z "${LLM_MODEL:-}" ]]; then
  echo "Error: set LLM_MODEL or pass --llm-model through run_threaded_qa.sh" >&2
  exit 1
fi

if [[ -z "${SCRAPEDO_API_KEY:-}" ]]; then
  echo "Error: set SCRAPEDO_API_KEY for wiki-history scraping" >&2
  exit 1
fi

mkdir -p "$TEST_OUTPUT_DIR"

echo "======================================================================"
echo "Wiki History QA smoke test"
echo "======================================================================"
echo "Output: $TEST_OUTPUT_DIR"
echo "Revision dir: $WIKI_REVISIONS_DIR"
echo "Range: first 5 revisions"
echo ""

./run_threaded_qa.sh \
  --wiki-history \
  --wiki-revisions-dir "$WIKI_REVISIONS_DIR" \
  --output-dir "$TEST_OUTPUT_DIR" \
  --llm-workers 2 \
  --llm-host "${LLM_HOST:-127.0.0.1}" \
  --llm-port "${LLM_PORT:-8000}" \
  --llm-api-key "$LLM_API_KEY" \
  --llm-model "$LLM_MODEL" \
  --start-index 0 \
  --end-index 5

echo ""
echo "Smoke test complete."
echo "Inspect results with:"
echo "  ls -lh $TEST_OUTPUT_DIR"
echo "  cat $TEST_OUTPUT_DIR/wiki_history_summary.json"
