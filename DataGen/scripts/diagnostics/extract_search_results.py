#!/usr/bin/env python3
"""Extract normalized web search results from per-request JSON files."""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any


RESULT_PATTERN = re.compile(
    r"(\d+)\.\s+\[([^\]]+)\]\(([^)]+)\)"
    r"(?:\nDate published:\s+([^\n]+))?\n\n"
    r"(.+?)(?=\n\n\d+\.\s+\[|\n*$)",
    re.DOTALL,
)


def parse_search_result(search_result_text: str, timestamp: str) -> list[dict[str, Any]]:
    """Parse markdown-like search results into structured rows."""
    results: list[dict[str, Any]] = []
    for match in RESULT_PATTERN.finditer(search_result_text):
        result: dict[str, Any] = {
            "index": int(match.group(1)),
            "title": match.group(2).strip(),
            "link": match.group(3).strip(),
            "snippet": match.group(5).strip(),
            "timestamp": timestamp,
        }
        date_published = match.group(4)
        if date_published:
            result["date_published"] = date_published.strip()
        results.append(result)
    return results


def extract_from_json_file(json_path: Path) -> list[dict[str, Any]]:
    """Extract search-result rows from one JSON process log."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        search_result = data.get("search_result", "")
        if not search_result or "Web Results" not in search_result:
            return []

        web_results_match = re.search(r"## Web Results\n(.+)", search_result, re.DOTALL)
        if not web_results_match:
            return []

        rows = parse_search_result(web_results_match.group(1), data.get("timestamp", ""))
        for row in rows:
            row["request_id"] = data.get("request_id", "")
        return rows
    except Exception as exc:
        print(f"Failed to process {json_path}: {exc}")
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract search results from a search_process directory into JSONL."
    )
    parser.add_argument(
        "--search-process-dir",
        default="serper_filter_wiki_history/search_process",
        help="Directory containing per-request JSON files.",
    )
    parser.add_argument(
        "--output-file",
        default="wiki_history_search_results.jsonl",
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_process_dir = Path(args.search_process_dir)
    output_file = Path(args.output_file)

    json_files = sorted(search_process_dir.glob("*.json"))
    total_files = len(json_files)
    print(f"Found {total_files} JSON files in {search_process_dir}")

    start_time = time.time()
    all_results: list[dict[str, Any]] = []
    processed_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_from_json_file, json_file) for json_file in json_files]
        for future in as_completed(futures):
            all_results.extend(future.result())
            processed_count += 1
            if processed_count % 1000 == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                speed = processed_count / elapsed
                print(f"Processed {processed_count}/{total_files} files ({speed:.1f} files/sec)")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for result in all_results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"Extracted {len(all_results)} results in {elapsed:.2f}s")
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
