#!/usr/bin/env python3
"""Smoke-test helper functions used by the Wikipedia history generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.wiki.threaded_wiki_history_qa_generator import (
    fix_revision_url,
    format_date_for_question,
    parse_wikipedia_date,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test wiki history utility functions.")
    parser.add_argument(
        "--wiki-revisions-dir",
        default="wiki_revisions",
        help="Optional revision directory to inspect.",
    )
    parser.add_argument("--sample-files", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("Testing Wikipedia history utility functions")
    print("=" * 80)

    print("\nDate parsing and formatting:")
    for date_str in [
        "04:17, 27 July 2005",
        "19:38, 22 February 2026",
        "01:44, 7 July 2025",
    ]:
        print(f"  original: {date_str}")
        print(f"  parsed: {parse_wikipedia_date(date_str)}")
        print(f"  question date: {format_date_for_question(date_str)}")

    print("\nRevision URL normalization:")
    for url, oldid in [
        ("https://en.wikipedia.org/w/index.php?title=Tseri&diff=prev&oldid=1299200568", "1299200568"),
        (
            "https://en.wikipedia.org/w/index.php?title=William_Keeling&diff=1321171736&oldid=19691422",
            "19691422",
        ),
    ]:
        print(f"  original: {url}")
        print(f"  fixed: {fix_revision_url(url, oldid)}")

    revisions_dir = Path(args.wiki_revisions_dir)
    print(f"\nInspecting revision files in: {revisions_dir}")
    if not revisions_dir.exists():
        print("  directory does not exist, skipping file inspection")
        return

    json_files = list(revisions_dir.glob("*.json"))
    print(f"  found {len(json_files)} JSON files")
    for json_file in json_files[: args.sample_files]:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  failed to read {json_file.name}: {exc}")
            continue

        revisions = data.get("revisions") or []
        if not data.get("success") or not revisions:
            continue

        middle_revision = revisions[len(revisions) // 2]
        fixed_url = fix_revision_url(
            middle_revision.get("revision_url", ""),
            middle_revision.get("oldid", ""),
        )
        print(f"\n  file: {json_file.name}")
        print(f"    revisions: {len(revisions)}")
        print(f"    middle oldid: {middle_revision.get('oldid')}")
        print(f"    middle timestamp: {middle_revision.get('timestamp_text')}")
        print(f"    fixed URL: {fixed_url[:120]}")


if __name__ == "__main__":
    main()
