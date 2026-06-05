#!/usr/bin/env python3
"""Extract Wikipedia revision metadata from history-page URLs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


SCRAPEDO_API_KEY = os.getenv("SCRAPEDO_API_KEY", "")


def get_url_hash(url: str) -> str:
    """Generate an MD5 hash for a URL."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def fetch_html_via_scrape_api(url: str, proxies: dict[str, str], max_retries: int = 3) -> str | None:
    """Fetch HTML through scrape.do with light retry/backoff."""
    if not SCRAPEDO_API_KEY:
        raise RuntimeError("SCRAPEDO_API_KEY is required.")

    api_url = f"http://api.scrape.do/?token={SCRAPEDO_API_KEY}&url={urllib.parse.quote(url)}"
    for attempt in range(max_retries):
        try:
            response = requests.get(api_url, timeout=30, proxies=proxies)
            if response.status_code == 200:
                return response.text
            if response.status_code == 429:
                time.sleep(2**attempt)
                continue
            return None
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    return None


def normalize_revision_href(href: str) -> str:
    """Convert a relative Wikipedia href to an absolute URL."""
    if href.startswith("/"):
        return "https://en.wikipedia.org" + href
    return href


def clean_revision_url(href: str) -> tuple[str | None, str | None]:
    """Return a clean revision URL and oldid from a history-page href."""
    href = normalize_revision_href(href)
    parsed = urllib.parse.urlparse(href)
    params = urllib.parse.parse_qs(parsed.query)
    oldid = (params.get("oldid") or [None])[0]
    if not oldid:
        return None, None

    title = (params.get("title") or [None])[0]
    if title:
        revision_url = f"https://en.wikipedia.org/w/index.php?title={title}&oldid={oldid}"
    else:
        revision_url = f"https://en.wikipedia.org/w/index.php?oldid={oldid}"
    return revision_url, oldid


def extract_revision_from_li(li: Any, history_url: str) -> dict[str, Any] | None:
    """Extract one revision record from a Wikipedia history list item."""
    link = li.find("a", {"class": "mw-changeslist-date"})
    if not link:
        link = li.find("a", href=lambda value: value and "oldid=" in value)
    if not link:
        return None

    revision_url, oldid = clean_revision_url(link.get("href", ""))
    if not revision_url or not oldid:
        return None

    revision: dict[str, Any] = {
        "revision_url": revision_url,
        "oldid": oldid,
        "source_history_url": history_url,
    }

    time_text = link.get_text(strip=True)
    if time_text:
        revision["timestamp_text"] = time_text
        revision["timestamp"] = time_text

    user_link = li.find("a", {"class": "mw-userlink"})
    if user_link:
        revision["editor"] = user_link.get_text(strip=True)
    else:
        bdi_tag = li.find("bdi")
        if bdi_tag:
            revision["editor"] = bdi_tag.get_text(strip=True)

    comment_span = li.find("span", {"class": "comment"})
    if comment_span:
        revision["comment"] = comment_span.get_text(strip=True)

    size_span = li.find("span", {"class": "history-size"})
    if size_span:
        revision["size"] = size_span.get_text(strip=True)

    byte_change = li.find("span", {"class": lambda value: value and "mw-plusminus" in value})
    if byte_change:
        revision["byte_change"] = byte_change.get_text(strip=True)

    tags = li.find_all("a", {"class": "mw-tag-markers"})
    if tags:
        revision["tags"] = [tag.get_text(strip=True) for tag in tags]

    return revision


def extract_revisions_from_html(html: str, history_url: str) -> list[dict[str, Any]]:
    """Extract revision metadata rows from a Wikipedia history page."""
    soup = BeautifulSoup(html, "html.parser")
    history_lists = []
    history_lists.extend(soup.find_all("ul", {"class": "mw-contributions-list"}))

    pagehistory = soup.find("ul", {"id": "pagehistory"})
    if pagehistory:
        history_lists.append(pagehistory)

    history_lists.extend(soup.find_all("ul", {"class": "special"}))
    if not history_lists:
        history_lists = soup.find_all(
            "li",
            {"class": lambda value: value and any(token in str(value) for token in ["mw-tag", "mw-contribution"])},
        )

    revisions: list[dict[str, Any]] = []
    seen_oldids: set[str] = set()
    iterable = history_lists if not history_lists or history_lists[0].name == "li" else None

    list_items = iterable or [
        li for history_list in history_lists for li in history_list.find_all("li", recursive=False)
    ]
    for li in list_items:
        revision = extract_revision_from_li(li, history_url)
        if not revision:
            continue
        oldid = revision.get("oldid")
        if oldid in seen_oldids:
            continue
        revisions.append(revision)
        if oldid:
            seen_oldids.add(oldid)

    return revisions


def process_single_url(url: str, output_dir: Path, proxies: dict[str, str]) -> dict[str, Any]:
    """Fetch one history URL, extract revisions, and write one JSON result."""
    result: dict[str, Any] = {
        "url": url,
        "url_hash": get_url_hash(url),
        "success": False,
        "error": None,
        "revision_count": 0,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        html = fetch_html_via_scrape_api(url, proxies)
        if not html:
            result["error"] = "Failed to fetch HTML"
            return result

        result["html_size"] = len(html)
        revisions = extract_revisions_from_html(html, url)

        if not revisions:
            result["error"] = "No revisions found in HTML"
            result["html_preview"] = html[:1000]
        else:
            result["success"] = True
            result["revision_count"] = len(revisions)
            result["revisions"] = revisions

        output_file = output_dir / f"{get_url_hash(url)}.json"
        output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output_file"] = str(output_file)
    except Exception as exc:
        result["error"] = str(exc)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", default="sampled_wiki_history_urls.txt")
    parser.add_argument("--output-dir", default="wiki_revisions")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--scrapedo-api-key", default=os.getenv("SCRAPEDO_API_KEY", ""))
    parser.add_argument("--proxy", default=os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or "")
    args = parser.parse_args()
    if not args.scrapedo_api_key:
        parser.error("--scrapedo-api-key is required, or set SCRAPEDO_API_KEY.")
    return args


def main() -> None:
    global SCRAPEDO_API_KEY

    args = parse_args()
    SCRAPEDO_API_KEY = args.scrapedo_api_key
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else {}

    output_dir.mkdir(parents=True, exist_ok=True)
    urls = [line.strip() for line in input_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    print(f"Read {len(urls)} URLs from {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")

    stats = {"total": len(urls), "success": 0, "failed": 0, "total_revisions": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_url, url, output_dir, proxies): url for url in urls}
        with tqdm(total=len(urls), desc="Processing URLs") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result["success"]:
                        stats["success"] += 1
                        stats["total_revisions"] += result["revision_count"]
                    else:
                        stats["failed"] += 1
                except Exception:
                    stats["failed"] += 1
                pbar.set_postfix(
                    success=stats["success"],
                    failed=stats["failed"],
                    revisions=stats["total_revisions"],
                )
                pbar.update(1)

    summary_file = output_dir / "extraction_summary.json"
    summary_file.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "statistics": stats,
                "avg_revisions_per_page": stats["total_revisions"] / max(stats["success"], 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("Extraction complete")
    print(f"Success: {stats['success']}/{stats['total']}")
    print(f"Failed: {stats['failed']}/{stats['total']}")
    print(f"Total revisions: {stats['total_revisions']}")
    print(f"Summary file: {summary_file}")


if __name__ == "__main__":
    main()
