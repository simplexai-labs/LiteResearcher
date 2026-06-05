#!/usr/bin/env python3
"""
Filter positive.jsonl by checking if the answer exists in the CURRENT version
of the Wikipedia page. If the answer is found in the current wiki page, the QA
pair is "bad" (answer hasn't changed, so it's not time-sensitive) and goes to
negative. If not found, the QA pair is "good" (answer has changed) and stays
positive.

Steps for each QA pair:
1. Match question to a revision_url via qa_outputs_wiki_history/
2. Convert revision_url to current wiki URL
3. Fetch current wiki page (direct first, then scrape.do fallback)
4. Convert HTML to markdown
5. Use string-matching logic (from serper_filter.py) to check if answer is in page

Usage:
    python filter_current_wiki.py \
        --input serper_filter_wiki_history/positive.jsonl \
        --qa-dir qa_outputs_wiki_history \
        --output-dir output_filtered_current_wiki \
        --workers 8 \
        --resume
"""

import os
import re
import json
import html
import time
import argparse
import logging
import urllib.parse
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

# ============================================================================
# Import answer matching logic from serper_filter
# ============================================================================
from scripts.filtering.serper_filter import (
    answer_in_search_result,
    extract_numbers,
    chinese_number_to_int,
    normalize_text,
)

# ============================================================================
# Configuration
# ============================================================================
SCRAPEDO_API_KEY = os.getenv("SCRAPEDO_API_KEY", "")
PROXY = {}  # Add proxy if needed, e.g. {"http": "...", "https": "..."}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# HTML to Markdown Converter
# ============================================================================
def _normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def _join_markdown_chunks(chunks) -> str:
    filtered = [chunk for chunk in chunks if chunk]
    return "".join(filtered)


def _html_to_markdown_with_bs4(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "canvas"]):
        tag.decompose()

    def render_node(node) -> str:
        from bs4 import NavigableString, Tag

        if isinstance(node, NavigableString):
            return _normalize_whitespace(html.unescape(str(node))) + " "

        if not isinstance(node, Tag):
            return ""

        name = node.name.lower()

        if name == "br":
            return "\n"

        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            content = _normalize_whitespace(
                _join_markdown_chunks(render_node(child) for child in node.children)
            )
            return f"{'#' * level} {content}\n\n" if content else ""

        if name == "p":
            content = _normalize_whitespace(
                _join_markdown_chunks(render_node(child) for child in node.children)
            )
            return f"{content}\n\n" if content else ""

        if name in {"ul", "ol"}:
            items = []
            for child in node.find_all("li", recursive=False):
                items.append(render_node(child))
            return "".join(items) + "\n"

        if name == "li":
            content = _normalize_whitespace(
                _join_markdown_chunks(render_node(child) for child in node.children)
            )
            return f"- {content}\n" if content else ""

        if name == "a":
            content = _normalize_whitespace(
                _join_markdown_chunks(render_node(child) for child in node.children)
            )
            href = node.get("href", "").strip()
            if content and href:
                return f"[{content}]({href})"
            return content

        if name in {"strong", "b"}:
            content = _normalize_whitespace(
                _join_markdown_chunks(render_node(child) for child in node.children)
            )
            return f"**{content}**" if content else ""

        if name in {"em", "i"}:
            content = _normalize_whitespace(
                _join_markdown_chunks(render_node(child) for child in node.children)
            )
            return f"*{content}*" if content else ""

        return _join_markdown_chunks(render_node(child) for child in node.children)

    body = soup.body or soup
    markdown = _join_markdown_chunks(render_node(child) for child in body.children)
    cleaned = re.sub(r"\n{3,}", "\n\n", markdown)
    return cleaned.strip()


def _html_to_markdown_basic(raw_html: str) -> str:
    text = re.sub(
        r"<\s*(script|style|noscript|iframe)[^>]*>.*?<\s*/\s*\1\s*>",
        " ",
        raw_html,
        flags=re.S | re.I,
    )
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p>", "\n\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _normalize_whitespace(text)


def html_to_markdown(raw_html: str) -> str:
    if not raw_html:
        return ""
    try:
        return _html_to_markdown_with_bs4(raw_html)
    except Exception:
        pass
    return _html_to_markdown_basic(raw_html)


# ============================================================================
# Build question -> revision_url index
# ============================================================================
def build_question_index(qa_dir: str) -> dict:
    """
    Scan all JSON files in qa_dir and build a mapping:
        question_text -> revision_url
    """
    index = {}
    files = [f for f in os.listdir(qa_dir) if f.endswith(".json")]
    print(f"Building question index from {len(files)} files in {qa_dir} ...")

    for fname in tqdm(files, desc="Indexing", unit="file"):
        fpath = os.path.join(qa_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        revision_url = data.get("revision_url", "")
        if not revision_url:
            continue

        qa_pairs = data.get("qa_pairs", [])
        for pair in qa_pairs:
            q = pair.get("question", "").strip()
            if q:
                index[q] = revision_url

    print(f"Index built: {len(index)} questions mapped.")
    return index


# ============================================================================
# URL conversion: revision URL -> current wiki URL
# ============================================================================
def revision_url_to_current(revision_url: str) -> Optional[str]:
    """
    Convert a Wikipedia revision URL to the current article URL.

    Example:
      https://en.wikipedia.org/w/index.php?title=Paezan_languages&oldid=35758321
      -> https://en.wikipedia.org/wiki/Paezan_languages
    """
    parsed = urllib.parse.urlparse(revision_url)
    params = urllib.parse.parse_qs(parsed.query)
    title = params.get("title", [None])[0]
    if not title:
        return None
    # Reconstruct current URL
    return f"{parsed.scheme}://{parsed.netloc}/wiki/{title}"


# ============================================================================
# Fetch current wiki page
# ============================================================================
def fetch_page(url: str) -> Optional[str]:
    """
    Fetch a URL via scrape.do API (3 attempts, last with render=true).
    Returns raw HTML or None on failure.
    """
    for attempt in range(3):
        try:
            if (attempt + 1) % 3 == 0:
                # 3rd attempt: with render
                api_url = (
                    f"https://api.scrape.do/?token={SCRAPEDO_API_KEY}"
                    f"&url={urllib.parse.quote(url)}"
                    f"&customWait=2000&render=true"
                )
            else:
                # 1st, 2nd attempt: without render
                api_url = (
                    f"https://api.scrape.do/?token={SCRAPEDO_API_KEY}"
                    f"&url={urllib.parse.quote(url)}"
                )
            resp = requests.get(api_url, timeout=120, proxies=PROXY)
            if resp.status_code == 200 and len(resp.text) > 500:
                return resp.text
            # Log non-200 for debugging
            logger.warning(
                f"scrape.do attempt {attempt+1}/3 status={resp.status_code} "
                f"len={len(resp.text)} url={url}"
            )
        except Exception as e:
            logger.warning(f"scrape.do attempt {attempt+1}/3 error: {e} url={url}")

    return None


# ============================================================================
# URL-level cache to avoid refetching the same wiki page
# ============================================================================
_url_cache = {}
_url_cache_lock = Lock()


def fetch_and_convert(url: str) -> Optional[str]:
    """Fetch page and convert to markdown, with caching."""
    with _url_cache_lock:
        if url in _url_cache:
            return _url_cache[url]

    raw_html = fetch_page(url)
    if raw_html is None:
        md = None
    else:
        md = html_to_markdown(raw_html)

    with _url_cache_lock:
        _url_cache[url] = md

    return md


# ============================================================================
# Process a single QA pair
# ============================================================================
def process_item(item: dict, question_index: dict, item_idx: int = 0) -> dict:
    """
    Process one QA pair. Returns a dict with:
        result: "positive" | "negative" | "skipped"
        item: original QA pair dict
        reason: explanation string
        process: detailed process record (for logging)
    """
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()
    start_time = time.time()

    # Base process record
    process = {
        "item_index": item_idx,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "revision_url": None,
        "current_url": None,
        "fetch_success": False,
        "markdown_length": 0,
        "markdown_content": "",
        "answer_normalized": normalize_text(answer) if answer else "",
        "answer_numbers": extract_numbers(answer) if answer else [],
        "question_numbers": extract_numbers(question) if question else [],
        "answer_chinese_numbers": chinese_number_to_int(answer) if answer else [],
        "wiki_numbers": [],
        "wiki_chinese_numbers": [],
        "answer_found_in_wiki": False,
        "is_positive": True,
        "reason": "",
        "elapsed_seconds": 0.0,
    }

    if not question or not answer:
        process["reason"] = "empty question/answer"
        process["elapsed_seconds"] = round(time.time() - start_time, 6)
        return {"result": "skipped", "item": item, "reason": process["reason"], "process": process}

    # Step 1: find revision_url
    revision_url = question_index.get(question)
    process["revision_url"] = revision_url
    if not revision_url:
        process["reason"] = "no revision_url found"
        process["elapsed_seconds"] = round(time.time() - start_time, 6)
        return {"result": "positive", "item": item, "reason": process["reason"], "process": process}

    # Step 2: convert to current URL
    current_url = revision_url_to_current(revision_url)
    process["current_url"] = current_url
    if not current_url:
        process["reason"] = "cannot parse revision_url"
        process["elapsed_seconds"] = round(time.time() - start_time, 6)
        return {"result": "positive", "item": item, "reason": process["reason"], "process": process}

    # Step 3: fetch and convert
    md_content = fetch_and_convert(current_url)
    if not md_content:
        process["reason"] = "fetch failed"
        process["elapsed_seconds"] = round(time.time() - start_time, 6)
        return {"result": "positive", "item": item, "reason": process["reason"], "process": process}

    process["fetch_success"] = True
    process["markdown_length"] = len(md_content)
    process["markdown_content"] = md_content
    process["wiki_numbers"] = extract_numbers(md_content)
    process["wiki_chinese_numbers"] = chinese_number_to_int(md_content)

    # Step 4: check if answer is in current wiki page
    answer_found = answer_in_search_result(answer, md_content, question)
    process["answer_found_in_wiki"] = answer_found
    process["is_positive"] = not answer_found
    process["elapsed_seconds"] = round(time.time() - start_time, 6)

    if answer_found:
        process["reason"] = "answer found in current wiki"
        return {"result": "negative", "item": item, "reason": process["reason"], "process": process}
    else:
        process["reason"] = "answer not in current wiki"
        return {"result": "positive", "item": item, "reason": process["reason"], "process": process}


# ============================================================================
# Main
# ============================================================================
def main():
    global SCRAPEDO_API_KEY, PROXY

    parser = argparse.ArgumentParser(
        description="Filter QA pairs by checking if answer exists in current Wikipedia"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="serper_filter_wiki_history/positive.jsonl",
        help="Input positive JSONL file",
    )
    parser.add_argument(
        "--qa-dir",
        type=str,
        default="qa_outputs_wiki_history",
        help="Directory with QA output JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_filtered_current_wiki",
        help="Output directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Max items to process (for debugging)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output (skip already processed questions)",
    )
    parser.add_argument(
        "--scrapedo-api-key",
        type=str,
        default=os.getenv("SCRAPEDO_API_KEY", ""),
        help="scrape.do API key. Defaults to SCRAPEDO_API_KEY.",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or "",
        help="Optional proxy URL used for both http and https requests.",
    )
    args = parser.parse_args()
    if not args.scrapedo_api_key:
        parser.error("--scrapedo-api-key is required, or set SCRAPEDO_API_KEY.")

    SCRAPEDO_API_KEY = args.scrapedo_api_key
    if args.proxy:
        PROXY = {"http": args.proxy, "https": args.proxy}

    os.makedirs(args.output_dir, exist_ok=True)
    process_dir = os.path.join(args.output_dir, "search_process")
    os.makedirs(process_dir, exist_ok=True)
    positive_file = os.path.join(args.output_dir, "positive.jsonl")
    negative_file = os.path.join(args.output_dir, "negative.jsonl")

    # Build question -> revision_url index
    question_index = build_question_index(args.qa_dir)

    # Load already processed questions for resume
    processed_questions = set()
    if args.resume:
        for fpath in [positive_file, negative_file]:
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line.strip())
                            processed_questions.add(obj.get("question", ""))
                        except Exception:
                            pass
        print(f"Resume mode: {len(processed_questions)} already processed, will skip them.")

    # Load input items
    items = []
    skipped = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            q = obj.get("question", "").strip()
            if args.resume and q in processed_questions:
                skipped += 1
                continue
            items.append(obj)
            if args.max_items and len(items) >= args.max_items:
                break

    if skipped:
        print(f"Skipped {skipped} already processed items.")
    print(f"Processing {len(items)} items with {args.workers} workers ...")

    # Thread-safe file writers
    write_lock = Lock()
    counters = {"positive": 0, "negative": 0, "skipped": 0}

    pbar = tqdm(total=len(items), desc="Filtering", unit="item")

    def worker_fn(item_with_idx):
        idx, item = item_with_idx
        return process_item(item, question_index, item_idx=idx)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(worker_fn, (i, item)): item
            for i, item in enumerate(items)
        }

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"Worker exception: {e}")
                result = {
                    "result": "positive",
                    "item": futures[future],
                    "reason": f"exception: {e}",
                }

            label = result["result"]
            qa_pair = result["item"]
            process_record = result.get("process")

            # Save detailed process log
            if process_record:
                idx = process_record.get("item_index", 0)
                proc_file = os.path.join(process_dir, f"{idx:06d}.json")
                try:
                    with open(proc_file, "w", encoding="utf-8") as pf:
                        json.dump(process_record, pf, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.error(f"Failed to write process file: {e}")

            with write_lock:
                if label == "positive":
                    with open(positive_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
                    counters["positive"] += 1
                elif label == "negative":
                    with open(negative_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
                    counters["negative"] += 1
                else:
                    counters["skipped"] += 1

                pbar.set_postfix(
                    pos=counters["positive"],
                    neg=counters["negative"],
                    skip=counters["skipped"],
                )
                pbar.update(1)

    pbar.close()

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Positive (answer changed / not found): {counters['positive']}")
    print(f"  Negative (answer same in current wiki): {counters['negative']}")
    print(f"  Skipped: {counters['skipped']}")
    print(f"  Output: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
