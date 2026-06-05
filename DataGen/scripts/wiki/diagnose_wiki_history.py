#!/usr/bin/env python3
"""Inspect output from the Wikipedia history Q&A generator."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze_failures(output_dir: Path) -> None:
    """Print a compact summary of failed revision records."""
    failed_file = output_dir / "failed_revisions.json"
    if not failed_file.exists():
        print("No failed_revisions.json file found.")
        return

    data = json.loads(failed_file.read_text(encoding="utf-8"))
    failed_revisions = data.get("failed_revisions", [])
    if not failed_revisions:
        print("No failed revisions recorded.")
        return

    print("\n" + "=" * 80)
    print("Failure Summary")
    print("=" * 80)
    print(f"Total failed revisions: {len(failed_revisions)}")

    error_types: Counter[str] = Counter()
    for record in failed_revisions:
        error = record.get("error", "Unknown")
        if "Failed to fetch HTML" in error:
            error_types["Failed to fetch HTML"] += 1
        elif "Markdown too short" in error:
            error_types["Markdown too short"] += 1
        elif "LLM failed" in error:
            error_types["LLM failed"] += 1
        else:
            error_types[error[:80]] += 1

    print("\nError types:")
    for error, count in error_types.most_common():
        percentage = count / len(failed_revisions) * 100
        print(f"  {error}: {count} ({percentage:.1f}%)")

    print("\nFirst 5 failures:")
    for index, record in enumerate(failed_revisions[:5], 1):
        print(f"\n{index}. oldid: {record.get('oldid', 'N/A')}")
        print(f"   timestamp: {record.get('timestamp', 'N/A')}")
        print(f"   url: {record.get('revision_url', 'N/A')[:100]}")
        print(f"   error: {record.get('error', 'N/A')}")


def check_output_stats(output_dir: Path) -> None:
    """Print output directory statistics and one sample record."""
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}")
        return

    json_files = list(output_dir.glob("*.json"))
    success_files = [
        path
        for path in json_files
        if path.name not in {"wiki_history_summary.json", "failed_revisions.json"}
    ]

    print("\n" + "=" * 80)
    print("Output Directory")
    print("=" * 80)
    print(f"Directory: {output_dir}")
    print(f"JSON files: {len(json_files)}")
    print(f"Generated QA files: {len(success_files)}")

    summary_file = output_dir / "wiki_history_summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        stats = summary.get("statistics", {})
        print("\nSummary file:")
        print(f"  success: {stats.get('success', 0)}")
        print(f"  failed: {stats.get('failed', 0)}")
        print(f"  total QA pairs: {stats.get('total_qa_pairs', 0)}")
        print(f"  avg QA pairs/revision: {summary.get('avg_qa_pairs', 0):.1f}")
        if "total_processed" in summary:
            print(f"  total processed: {summary.get('total_processed', 0)}")
            print(f"  success rate: {summary.get('success_rate', 0):.1f}%")

    if success_files:
        sample_file = success_files[0]
        sample = json.loads(sample_file.read_text(encoding="utf-8"))
        print("\nSample generated file:")
        print(f"  file: {sample_file.name}")
        print(f"  oldid: {sample.get('oldid', 'N/A')}")
        print(f"  timestamp constraint: {sample.get('timestamp_constraint', 'N/A')}")
        print(f"  QA pairs: {sample.get('total_pairs', 0)}")
        if sample.get("qa_pairs"):
            question = sample["qa_pairs"][0].get("question", "N/A")
            print(f"  sample question: {question[:120]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Wikipedia history QA output.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="qa_outputs_wiki_history",
        help="Generator output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    print("Diagnosing Wikipedia history QA output")
    check_output_stats(output_dir)
    analyze_failures(output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
