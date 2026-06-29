#!/usr/bin/env python3
"""Sample Wikipedia article titles from Arrow files and create history URLs."""

from __future__ import annotations

import argparse
import os
import random
import urllib.parse
from glob import glob
from pathlib import Path
from urllib.parse import quote

import requests
from datasets import load_dataset
from tqdm import tqdm


def load_arrow_files(wiki_dir: str, max_files: int | None = None) -> list[str]:
    wiki_path = Path(wiki_dir)
    arrow_files = sorted(glob(str(wiki_path / "wikipedia-train-*.arrow")))
    if max_files:
        arrow_files = arrow_files[:max_files]

    all_titles: list[str] = []
    print(f"Reading titles from {len(arrow_files)} Arrow files...")
    for arrow_file in tqdm(arrow_files):
        try:
            ds = load_dataset("arrow", data_files=[arrow_file], split="train")
            if "title" in ds.column_names:
                all_titles.extend(ds["title"])
            else:
                print(f"Warning: {arrow_file} has no 'title' column; columns={ds.column_names}")
        except Exception as exc:
            print(f"Failed to read {arrow_file}: {exc}")
    return all_titles


def create_history_url(title: str) -> str:
    encoded_title = quote(title.replace(" ", "_"))
    return f"https://en.wikipedia.org/w/index.php?title={encoded_title}&action=history&offset=&limit=2000"


def test_urls(urls: list[str], num_test: int = 5, proxy: str = "", scrapedo_api_key: str = "") -> bool:
    print(f"\nTesting {num_test} sampled URLs...")
    test_sample = random.sample(urls, min(num_test, len(urls)))
    proxies = {"http": proxy, "https": proxy} if proxy else {}

    success_count = 0
    for url in test_sample:
        try:
            if scrapedo_api_key:
                api_url = f"http://api.scrape.do/?token={scrapedo_api_key}&url={urllib.parse.quote(url)}"
                response = requests.get(api_url, timeout=15, proxies=proxies)
            else:
                response = requests.get(url, timeout=15, proxies=proxies)
            status = "OK" if response.status_code == 200 else f"HTTP {response.status_code}"
            print(f"{status}: {url}")
            if response.status_code == 200:
                success_count += 1
        except Exception as exc:
            print(f"ERROR: {url} - {type(exc).__name__}: {exc}")

    print(f"\nSuccess: {success_count}/{len(test_sample)}")
    return success_count > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-dir", default="wiki_en", help="Directory containing Wikipedia Arrow files.")
    parser.add_argument("--output-file", default="sampled_wiki_history_urls.txt")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true", help="Do not test sampled URLs.")
    parser.add_argument("--proxy", default=os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or "")
    parser.add_argument("--scrapedo-api-key", default=os.getenv("SCRAPEDO_API_KEY", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    print("Step 1: reading Wikipedia titles...")
    all_titles = load_arrow_files(args.wiki_dir, max_files=args.max_files)
    print(f"Found {len(all_titles)} titles")
    if not all_titles:
        raise SystemExit("No titles found.")

    sample_size = min(args.sample_size, len(all_titles))
    print(f"\nStep 2: sampling {sample_size} titles...")
    sampled_titles = random.sample(all_titles, sample_size)

    print("\nStep 3: creating history URLs...")
    urls = [create_history_url(title) for title in tqdm(sampled_titles)]

    if not args.skip_test:
        print("\nStep 4: testing URL accessibility...")
        test_urls(urls, num_test=5, proxy=args.proxy, scrapedo_api_key=args.scrapedo_api_key)

    print(f"\nStep 5: saving to {args.output_file}...")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(urls) + "\n", encoding="utf-8")

    print(f"\nDone. Saved {len(urls)} URLs to {output_path}")
    print("\nExample URLs:")
    for url in urls[:5]:
        print(f"  {url}")


if __name__ == "__main__":
    main()
