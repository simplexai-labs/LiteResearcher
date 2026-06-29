#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-qa_outputs_wiki}"
LOG_FILE="${LOG_FILE:-wiki_qa_processing.log}"

echo "=== Wiki QA Processing Monitor ==="
echo ""

if pgrep -af "threaded_qa_batch_generator.py|generate_qa_pairs_wiki.py" >/dev/null; then
  echo "Process is running"
  pgrep -af "threaded_qa_batch_generator.py|generate_qa_pairs_wiki.py"
else
  echo "Process is not running"
fi

echo ""
echo "=== Output Statistics ==="
if [[ -d "$OUTPUT_DIR" ]]; then
  num_files=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.json" -type f | wc -l)
  echo "JSON files generated: $num_files"
  latest_file=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.json" -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  if [[ -n "$latest_file" ]]; then
    echo "Latest file: $(basename "$latest_file")"
    echo "Last modified: $(stat -c '%y' "$latest_file" | cut -d'.' -f1)"
  fi
else
  echo "Output directory not found: $OUTPUT_DIR"
fi

echo ""
echo "=== Recent Log Tail ==="
if [[ -f "$LOG_FILE" ]]; then
  tail -30 "$LOG_FILE"
else
  echo "Log file not found: $LOG_FILE"
fi
