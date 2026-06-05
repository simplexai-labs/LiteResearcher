#!/usr/bin/env python3
"""Small smoke test for Wikipedia history revision extraction."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.wiki.extract_wiki_revisions import process_single_url


DEFAULT_URLS = [
    "https://en.wikipedia.org/w/index.php?title=Greenland&action=history&offset=&limit=500",
    "https://en.wikipedia.org/w/index.php?title=Python_(programming_language)&action=history&offset=&limit=500",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny revision-extraction smoke test.")
    parser.add_argument("--output-dir", default="wiki_revisions_test")
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        help="History URL to test. May be repeated.",
    )
    parser.add_argument(
        "--proxy",
        default=os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or "",
        help="Optional proxy URL used for both http and https requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else {}
    urls = args.urls or DEFAULT_URLS

    print("Testing Wikipedia revision extraction")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    for index, url in enumerate(urls, 1):
        print(f"\n[{index}/{len(urls)}] {url}")
        result = process_single_url(url, output_dir, proxies)
        if result["success"]:
            print(f"  success: {result['revision_count']} revisions")
            print(f"  file: {result['output_file']}")
        else:
            print(f"  failed: {result['error']}")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
