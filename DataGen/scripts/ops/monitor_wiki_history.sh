#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-qa_outputs_wiki_history}"
INTERVAL="${INTERVAL:-5}"

while true; do
  clear
  echo "======================================================================"
  echo "Wikipedia History QA Monitor"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Output: $OUTPUT_DIR"
  echo "======================================================================"
  echo ""

  if [[ -d "$OUTPUT_DIR" ]]; then
    total_files=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.json" -type f | wc -l)
    success_files=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.json" -type f \
      ! -name "wiki_history_summary.json" ! -name "failed_revisions.json" | wc -l)
    recent_1min=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.json" -type f -mmin -1 \
      ! -name "wiki_history_summary.json" ! -name "failed_revisions.json" | wc -l)

    echo "JSON files: $total_files"
    echo "Generated QA files: $success_files"
    echo "Generated in the last minute: $recent_1min"

    latest_file=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.json" -type f \
      ! -name "wiki_history_summary.json" ! -name "failed_revisions.json" \
      -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    if [[ -n "$latest_file" ]]; then
      echo "Latest file: $(basename "$latest_file")"
      echo "Latest modified: $(stat -c '%y' "$latest_file" | cut -d'.' -f1)"
    fi
  else
    echo "Output directory not found: $OUTPUT_DIR"
  fi

  echo ""
  echo "Press Ctrl+C to exit."
  sleep "$INTERVAL"
done
