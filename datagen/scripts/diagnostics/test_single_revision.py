#!/usr/bin/env python3
"""Smoke-test one Wikipedia revision and an OpenAI-compatible LLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
from pathlib import Path

import requests


def test_scrape_api(url: str, scrapedo_api_key: str, proxy: str = "") -> bool:
    print(f"\nTesting scrape.do API with URL: {url}")
    api_url = f"http://api.scrape.do/?token={scrapedo_api_key}&url={urllib.parse.quote(url)}"
    proxies = {"http": proxy, "https": proxy} if proxy else {}

    try:
        response = requests.get(api_url, timeout=60, proxies=proxies)
        print(f"Status: {response.status_code}")
        print(f"Content length: {len(response.content)} bytes")
        if response.status_code != 200:
            return False

        html = response.text
        print(f"HTML length: {len(html)} characters")
        print(f"First 500 chars: {html[:500]}")
        is_valid = "Wikipedia" in html and "mw-parser-output" in html
        print("Valid Wikipedia page detected" if is_valid else "Not a valid Wikipedia page")
        return is_valid
    except Exception as exc:
        print(f"Scrape exception: {exc}")
        return False


def test_llm_api(host: str, port: int, api_key: str, model_name: str) -> bool:
    print(f"\nTesting LLM API at http://{host}:{port}")
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello, please say 'test success'."}],
        "max_tokens": 50,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)[:500]}")
            return True
        print(f"Response: {response.text[:500]}")
        return False
    except Exception as exc:
        print(f"LLM exception: {exc}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-dir", default="wiki_revisions")
    parser.add_argument("--scrapedo-api-key", default=os.getenv("SCRAPEDO_API_KEY", ""))
    parser.add_argument("--proxy", default=os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or "")
    parser.add_argument("--llm-host", default=os.getenv("LLM_HOST", "127.0.0.1"))
    parser.add_argument("--llm-port", type=int, default=int(os.getenv("LLM_PORT", "8000")))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--llm-model-name", default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()
    if not args.scrapedo_api_key:
        parser.error("--scrapedo-api-key is required, or set SCRAPEDO_API_KEY.")
    if not args.skip_llm and (not args.llm_api_key or not args.llm_model_name):
        parser.error("--llm-api-key and --llm-model-name are required unless --skip-llm is set.")
    return args


def main() -> None:
    args = parse_args()
    revision_dir = Path(args.revision_dir)
    sample_files = sorted(revision_dir.glob("*.json"))[:5]
    if not sample_files:
        print(f"No revision files found in {revision_dir}")
        return

    print(f"Found {len(sample_files)} sample revision files")
    for json_file in sample_files:
        print(f"\n{'=' * 80}")
        print(f"Testing with file: {json_file.name}")
        print(f"{'=' * 80}")

        data = json.loads(json_file.read_text(encoding="utf-8"))
        revisions = data.get("revisions", [])
        if not revisions:
            print("No revisions in file")
            continue

        mid_idx = len(revisions) // 2
        revision = revisions[mid_idx]
        url = revision.get("revision_url")
        print(f"oldid: {revision.get('oldid')}")
        print(f"timestamp: {revision.get('timestamp')}")
        print(f"revision_url: {url}")
        if not url:
            continue

        scrape_ok = test_scrape_api(url, args.scrapedo_api_key, args.proxy)
        if scrape_ok and not args.skip_llm:
            llm_ok = test_llm_api(args.llm_host, args.llm_port, args.llm_api_key, args.llm_model_name)
            if llm_ok:
                print("\nAll tests passed for this revision.")
                return
        elif scrape_ok:
            print("\nScrape test passed.")
            return

    print("\nNo sample revision passed all requested checks.")


if __name__ == "__main__":
    main()
