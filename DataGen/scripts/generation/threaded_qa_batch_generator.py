#!/usr/bin/env python3
"""Threaded batch generator that pulls HTML markdown from a remote service and
queries a local OpenAI-compatible LLM API to build QA pairs."""

from __future__ import annotations

import argparse
import html
import hashlib
import json
import logging
import os
import queue
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set
from glob import glob

# Disable proxy environment variables before importing requests
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"):
    os.environ.pop(key, None)

import requests
from tqdm import tqdm

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.ipc  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pa = None  # type: ignore

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pq = None  # type: ignore

# Thread-local HTTP sessions keep connections reused per worker without locking.
_thread_local = threading.local()


def _get_http_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update({"User-Agent": "DirectQADataGen/1.0"})
        
        # Simple session without complex pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=0
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        _thread_local.session = session
    return _thread_local.session  # type: ignore[return-value]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def _join_markdown_chunks(chunks: Iterable[str]) -> str:
    filtered = [chunk for chunk in chunks if chunk]
    return "".join(filtered)


def _html_to_markdown_with_bs4(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")  # type: ignore[arg-type]
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "canvas"]):
        tag.decompose()

    def render_node(node) -> str:
        from bs4 import NavigableString, Tag  # type: ignore

        if isinstance(node, NavigableString):
            return _normalize_whitespace(html.unescape(str(node))) + " "

        if not isinstance(node, Tag):
            return ""

        name = node.name.lower()

        if name == "br":
            return "\n"

        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"{'#' * level} {content}\n\n" if content else ""

        if name == "p":
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"{content}\n\n" if content else ""

        if name in {"ul", "ol"}:
            items = []
            for child in node.find_all("li", recursive=False):
                items.append(render_node(child))
            return "".join(items) + "\n"

        if name == "li":
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"- {content}\n" if content else ""

        if name == "a":
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            href = node.get("href", "").strip()
            if content and href:
                return f"[{content}]({href})"
            return content

        if name in {"strong", "b"}:
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"**{content}**" if content else ""

        if name in {"em", "i"}:
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"*{content}*" if content else ""

        return _join_markdown_chunks(render_node(child) for child in node.children)

    body = soup.body or soup  # type: ignore[assignment]
    markdown = _join_markdown_chunks(render_node(child) for child in body.children)
    cleaned = re.sub(r"\n{3,}", "\n\n", markdown)
    return cleaned.strip()


def _html_to_markdown_basic(raw_html: str) -> str:
    text = re.sub(r"<\s*(script|style|noscript|iframe)[^>]*>.*?<\s*/\s*\1\s*>", " ", raw_html, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p>", "\n\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _normalize_whitespace(text)


def html_to_markdown(raw_html: str) -> str:
    if not raw_html:
        return ""
    if BeautifulSoup is not None:
        try:
            return _html_to_markdown_with_bs4(raw_html)
        except Exception:
            pass
    return _html_to_markdown_basic(raw_html)


def build_prompt(markdown_content: str) -> str:
    return (
        "You are a data extraction and Q&A generation expert. Your task is to create "
        "high-quality, standalone question-answer pairs from webpage content for training "
        "a research model that will never see the original source.\n\n"
        
        "CRITICAL REQUIREMENTS:\n\n"
        
        "1. COMPLETE INDEPENDENCE:\n"
        "   - Questions MUST be answerable without ANY reference to the source\n"
        "   - FORBIDDEN phrases: 'according to the article', 'in this report', 'the webpage states', "
        "'according to...', 'as mentioned', 'this document', '根据文章', '该报告', '本文', '据悉' etc.\n"
        "   - Every piece of context must be embedded directly into the question\n"
        "   - Treat each Q&A as if it will be used in isolation, with no source available\n\n"
        
        "2. LANGUAGE SELECTION:\n"
        "   - Chinese content (about China, Chinese companies, Chinese topics) → Chinese Q&A\n"
        "   - English/International content → English Q&A\n"
        "   - Match the language to the content's primary context and audience\n\n"
        
        "3. MEANINGFUL DATA ONLY:\n"
        "   - Extract VALUABLE information: numbers, dates, percentages, amounts, names, statistics, "
        "quantities, measurements, rankings, rates, ratios\n"
        "   - FORBIDDEN topics: website URLs, email addresses, social media handles, contact information, "
        "navigation links, menu items, footer information\n"
        "   - Focus on substantive facts that have research or informational value\n\n"
        
        "4. MAXIMUM SPECIFICITY IN QUESTIONS:\n"
        "   - Include MULTIPLE identifying details to eliminate ambiguity:\n"
        "     * Full official names (not abbreviations alone)\n"
        "     * Specific time periods (exact quarters, months, years)\n"
        "     * Geographic locations when relevant\n"
        "     * Product/project/report/event full names\n"
        "     * Role/title/position for people\n"
        "   - BAD: \"What was Tesla's revenue?\"\n"
        "   - GOOD: \"What was Tesla Inc.'s total automotive revenue in the third quarter of 2024?\"\n"
        "   - BAD: \"比亚迪的销量是多少?\"\n"
        "   - GOOD: \"比亚迪股份有限公司2024年11月的新能源乘用车销量是多少辆?\"\n\n"
        
        "5. HANDLE AMBIGUITY:\n"
        "   - If multiple entities share the same name, specify distinguishing details:\n"
        "     * Location (Apple Inc. in Cupertino vs Apple Records in London)\n"
        "     * Industry (Tesla the car company vs Tesla the scientist)\n"
        "     * Time period to establish context\n"
        "   - Use full legal/official names when available\n"
        "   - Add descriptors: \"CEO\", \"headquartered in\", \"founded in\", etc.\n\n"
        
        "6. ANSWER PRECISION:\n"
        "   - Exact values with proper units\n"
        "   - Examples: \"$46.2 billion USD\", \"15.3%\", \"December 15, 2025\", \"500,000 units\"\n"
        "   - Include currency, percentages, measurement units\n"
        "   - For Chinese: \"45.2亿元人民币\", \"15.3%\", \"2024年12月15日\", \"50万辆\"\n\n"
        
        "OUTPUT FORMAT:\n"
        "{\n"
        "    \"qa_pairs\": [\n"
        "        {\"question\": \"...\", \"answer\": \"...\"}\n"
        "    ],\n"
        "    \"total_pairs\": <number>\n"
        "}\n\n"
        
        "QUALITY EXAMPLES:\n\n"
        
        "✓ CORRECT English:\n"
        "Q: \"What was Apple Inc.'s total revenue for fiscal Q4 2024 ending September 28, 2024?\"\n"
        "A: \"$94.93 billion USD\"\n\n"
        
        "✓ CORRECT Chinese:\n"
        "Q: \"宁德时代新能源科技股份有限公司在2024年第三季度的动力电池装机量是多少GWh?\"\n"
        "A: \"97.7 GWh\"\n\n"
        
        "✗ WRONG (lacks context):\n"
        "Q: \"What was the revenue?\"\n\n"
        
        "✗ WRONG (references source):\n"
        "Q: \"According to the report, what was Tesla's profit?\"\n\n"
        
        "✗ WRONG (useless data):\n"
        "Q: \"What is the company's website URL?\"\n\n"
        
        "✗ WRONG (ambiguous):\n"
        "Q: \"What was Apple's revenue in 2024?\"\n"
        "(Which Apple? Which quarter/period in 2024? Which revenue category?)\n\n"
        
        "Now extract data from this content:\n\n"
        "---CONTENT START---\n"
        f"{markdown_content}\n"
        "---CONTENT END---"
    )


def parse_llm_json(content: str) -> Dict[str, Any]:
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    json_match = re.search(r"\{[^{}]*\"qa_pairs\"[^{}]*\[.*?\][^{}]*\}", content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    return json.loads(content)


def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def load_processed_urls(resume_dir: Path) -> Set[str]:
    """Load all processed URLs from resume directory for deduplication."""
    if not resume_dir.exists() or not resume_dir.is_dir():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
    
    processed_urls: Set[str] = set()
    for file_path in resume_dir.glob("*.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            url = data.get("url")
            if url:
                processed_urls.add(url)
        except Exception:
            continue
    
    return processed_urls


class BatchProcessor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.html_base_url = args.html_base_url.rstrip("/") if args.html_base_url else None
        self.llm_api_url = args.llm_api_url.rstrip("/")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("BatchProcessor")
        
        # Wiki mode: prepare arrow files list and calculate cumulative counts
        self.wiki_arrow_files = None
        self.wiki_cumulative_counts = []  # Cumulative article counts for each file
        if args.wiki:
            self._prepare_wiki_files()
        
        # BBC mode: load and shuffle all news articles
        self.bbc_records = []
        if args.bbc:
            self._prepare_bbc_data()
        
        # Producer-Consumer queues
        self.html_queue = queue.Queue(maxsize=args.queue_size)
        self.stats = {"success": 0, "error": 0, "skipped": 0, "missing": 0, "url_duplicate": 0}
        self.stats_lock = threading.Lock()
        
        # Load processed URLs for deduplication
        self.processed_urls: Set[str] = set()
        if args.resume:
            try:
                self.processed_urls = load_processed_urls(Path(args.resume))
                self.logger.info("Loaded %s processed URLs for deduplication", len(self.processed_urls))
            except Exception as e:
                self.logger.warning("Failed to load processed URLs: %s", e)
    
    def _prepare_wiki_files(self) -> None:
        """Prepare list of wiki arrow files and calculate sizes."""
        if pa is None:
            raise RuntimeError("pyarrow package not installed, required for --wiki mode")
        
        wiki_dir = Path(self.args.wiki_dir)
        self.wiki_arrow_files = sorted(glob(str(wiki_dir / "*.arrow")))
        
        if not self.wiki_arrow_files:
            raise RuntimeError(f"No arrow files found in {wiki_dir}")
        
        num_files = self.args.wiki_max_files if self.args.wiki_max_files else len(self.wiki_arrow_files)
        self.wiki_arrow_files = self.wiki_arrow_files[:num_files]
        
        # Calculate cumulative counts by quickly scanning file headers
        self.logger.info("Scanning %s wiki arrow files...", len(self.wiki_arrow_files))
        cumulative = 0
        for arrow_file in self.wiki_arrow_files:
            try:
                # Quick scan: just get the count without loading data
                mmap = pa.memory_map(arrow_file)
                with pa.ipc.open_stream(mmap) as reader:
                    # Read schema only, count total rows
                    count = 0
                    for batch in reader:
                        count += len(batch)
                cumulative += count
                self.wiki_cumulative_counts.append(cumulative)
                self.logger.info("  %s: %s articles (cumulative: %s)", 
                               Path(arrow_file).name, count, cumulative)
            except Exception as e:
                self.logger.error("  Failed to scan %s: %s", Path(arrow_file).name, e)
                raise
        
        self.logger.info("Total wiki articles available: %s", cumulative)

    def _prepare_bbc_data(self) -> None:
        """Load all BBC news parquet files, extract fields, shuffle."""
        if pq is None:
            raise RuntimeError("pyarrow.parquet not installed, required for --bbc mode")
        
        bbc_dir = Path(self.args.bbc_dir)
        parquet_files = sorted(glob(str(bbc_dir / "*/*.parquet")))
        
        if not parquet_files:
            raise RuntimeError(f"No parquet files found in {bbc_dir}")
        
        self.logger.info("Loading %s BBC parquet files...", len(parquet_files))
        
        all_records = []
        for pf in parquet_files:
            try:
                table = pq.read_table(pf, columns=["title", "description", "content", "link", "published_date"])
                for i in range(table.num_rows):
                    record = {
                        "title": table["title"][i].as_py() or "",
                        "description": table["description"][i].as_py() or "",
                        "content": table["content"][i].as_py() or "",
                        "link": table["link"][i].as_py() or "",
                        "published_date": table["published_date"][i].as_py() or "",
                    }
                    all_records.append(record)
                self.logger.info("  %s: %s articles", Path(pf).parent.name, table.num_rows)
            except Exception as e:
                self.logger.error("  Failed to load %s: %s", pf, e)
                raise
        
        self.logger.info("Loaded %s BBC articles total", len(all_records))
        
        # Shuffle all records
        random.shuffle(all_records)
        self.logger.info("Shuffled all BBC articles")
        
        # Apply limit if specified
        if self.args.bbc_limit and self.args.bbc_limit < len(all_records):
            all_records = all_records[:self.args.bbc_limit]
            self.logger.info("Limited to %s BBC articles", len(all_records))
        
        self.bbc_records = all_records

    def _get_bbc_record(self, index: int) -> Optional[Dict[str, Any]]:
        """Get BBC article by index from shuffled in-memory records."""
        if index >= len(self.bbc_records):
            return None
        
        record = self.bbc_records[index]
        title = record.get("title", "")
        published_date = record.get("published_date", "")
        description = record.get("description", "")
        content = record.get("content", "")
        
        # Construct markdown from BBC fields
        markdown = f"# {title}\n\nPublished: {published_date}\n\n{description}\n\n{content}"
        
        url = record.get("link", "") or f"bbc:{index}"
        
        return {
            "row_index": index,
            "url": url,
            "title": title,
            "published_date": published_date,
            "markdown": markdown,
            "text": markdown,
        }

    def fetch_html_worker(self, indices: Iterable[int], pbar: tqdm) -> None:
        """Single-threaded producer: fetch HTML records sequentially."""
        for index in indices:
            try:
                # Adaptive rate limiting based on queue size
                while self.html_queue.qsize() > self.args.queue_size * 0.8:
                    time.sleep(0.2)  # Slow down if queue is filling up
                
                # Choose data source based on mode
                if self.args.bbc:
                    record = self._get_bbc_record(index)
                elif self.args.wiki:
                    record = self._get_wiki_record(index)
                else:
                    record = self._fetch_record(index)
                
                if not record:
                    with self.stats_lock:
                        self.stats["missing"] += 1
                    pbar.update(1)
                    continue
                
                row_index = _safe_int(record.get("row_index")) or index
                markdown = record.get("markdown") or record.get("text") or ""
                
                if not markdown and record.get("html"):
                    markdown = html_to_markdown(record.get("html", ""))
                
                markdown = markdown.strip()
                
                if not markdown or len(markdown) < self.args.min_markdown_chars:
                    with self.stats_lock:
                        self.stats["skipped"] += 1
                    pbar.update(1)
                    continue
                
                if len(markdown) > self.args.max_markdown_chars:
                    markdown = markdown[: self.args.max_markdown_chars] + "\n\n[Content truncated...]"
                
                # Validate URL - skip if remote data is corrupted
                url = record.get("url") or f"record:{row_index}"
                if not url.startswith(("http://", "https://", "record:", "bbc:")):
                    self.logger.warning("Invalid URL for index %s, skipping", index)
                    with self.stats_lock:
                        self.stats["skipped"] += 1
                    pbar.update(1)
                    continue
                
                # Check if URL already processed (for resume deduplication)
                if url in self.processed_urls:
                    with self.stats_lock:
                        self.stats["url_duplicate"] += 1
                    pbar.update(1)
                    continue
                
                url_hash = get_url_hash(url)
                output_file = self.output_dir / f"{row_index:06d}_{url_hash}.json"
                
                if self.args.skip_existing and output_file.exists():
                    with self.stats_lock:
                        self.stats["skipped"] += 1
                    pbar.update(1)
                    continue
                
                # Put task into queue for LLM workers
                task = {
                    "row_index": row_index,
                    "url": url,
                    "url_hash": url_hash,
                    "markdown": markdown,
                    "html_length": record.get("html_length") or len(record.get("html", "")),
                    "storage_uid": record.get("storage_uid", ""),
                    "html_timestamp": record.get("timestamp", ""),
                    "output_file": output_file,
                    "article_id": record.get("id"),
                    "title": record.get("title", ""),
                    "published_date": record.get("published_date", ""),
                }
                self.html_queue.put(task)
                
                # Small delay between requests (only for HTTP mode)
                if not self.args.wiki and not self.args.bbc:
                    time.sleep(self.args.fetch_delay)
                
            except Exception as exc:
                self.logger.debug("Fetch index %s failed: %s", index, exc)
                with self.stats_lock:
                    self.stats["error"] += 1
                pbar.update(1)
        
        # Signal completion to LLM workers
        for _ in range(self.args.llm_workers):
            self.html_queue.put(None)
    
    def _get_wiki_record(self, index: int) -> Optional[Dict[str, Any]]:
        """Get wiki article by index, loading only the needed file."""
        if not self.wiki_arrow_files or not self.wiki_cumulative_counts:
            return None
        
        # Find which file contains this index using cumulative counts
        file_idx = 0
        local_index = index
        
        for i, cumulative in enumerate(self.wiki_cumulative_counts):
            if index < cumulative:
                file_idx = i
                if i > 0:
                    local_index = index - self.wiki_cumulative_counts[i - 1]
                else:
                    local_index = index
                break
        else:
            # Index beyond all files
            return None
        
        arrow_file = self.wiki_arrow_files[file_idx]
        
        try:
            # Load and find the specific record
            mmap = pa.memory_map(arrow_file)
            with pa.ipc.open_stream(mmap) as reader:
                current_row = 0
                for batch in reader:
                    batch_size = len(batch)
                    if local_index < current_row + batch_size:
                        # Found the batch containing our record
                        batch_local_idx = local_index - current_row
                        record = {
                            col: batch[col][batch_local_idx].as_py()
                            for col in batch.schema.names
                        }
                        return {
                            "row_index": index,
                            "id": record.get('id', index),
                            "url": record.get('url', f"wiki:{index}"),
                            "title": record.get('title', ''),
                            "text": record.get('text', ''),
                            "markdown": record.get('text', ''),
                        }
                    current_row += batch_size
        except Exception as e:
            self.logger.error("Failed to read record %s from %s: %s", 
                           local_index, Path(arrow_file).name, e)
            return None
        
        return None

    def llm_worker(self, pbar: tqdm) -> None:
        """Consumer: process tasks from queue with LLM."""
        while True:
            task = self.html_queue.get()
            
            if task is None:  # Poison pill
                self.html_queue.task_done()
                break
            
            try:
                qa_result = self._call_llm(task["markdown"])
                parsed = qa_result.get("parsed", {})
                
                # BBC mode: prepend date prefix to each question
                if self.args.bbc and task.get("published_date"):
                    pub_date = task["published_date"]
                    for qa in parsed.get("qa_pairs", []):
                        q = qa.get("question", "")
                        if q:
                            # Lowercase the first char of original question
                            q_body = q[0].lower() + q[1:] if q and q[0].isupper() else q
                            qa["question"] = f"{q_body}"
                
                output_data = {
                    "row_index": task["row_index"],
                    "url": task["url"],
                    "url_hash": task["url_hash"],
                    "markdown_length": len(task["markdown"]),
                    "html_length": task["html_length"],
                    "storage_uid": task["storage_uid"],
                    "qa_pairs": parsed.get("qa_pairs", []),
                    "total_pairs": parsed.get("total_pairs", 0),
                    "model_name": self.args.llm_model_name,
                    "model_raw_output": qa_result.get("raw"),
                    "html_timestamp": task["html_timestamp"],
                    "generation_timestamp": datetime.now().isoformat(),
                }
                
                # Add wiki-specific fields if in wiki mode
                if self.args.wiki:
                    output_data["article_id"] = task.get("article_id")
                    output_data["title"] = task.get("title")
                
                # Add BBC-specific fields if in BBC mode
                if self.args.bbc:
                    output_data["published_date"] = task.get("published_date")
                    output_data["title"] = task.get("title")
                
                with task["output_file"].open("w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                with self.stats_lock:
                    self.stats["success"] += 1
                
            except Exception as exc:
                self.logger.debug("LLM processing failed: %s", exc)
                with self.stats_lock:
                    self.stats["error"] += 1
            
            finally:
                self.html_queue.task_done()
                pbar.update(1)

    def _fetch_record(self, index: int) -> Optional[Dict[str, Any]]:
        url = f"{self.html_base_url}/records/{index}"
        last_error: Optional[Exception] = None
        for attempt in range(1, self.args.html_retries + 1):
            try:
                response = _get_http_session().get(
                    url, 
                    timeout=(3, self.args.html_timeout),
                    stream=False
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout as exc:
                last_error = exc
                if attempt < self.args.html_retries:
                    time.sleep(1)
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt < self.args.html_retries:
                    time.sleep(min(2 ** attempt, 3))
            except Exception as exc:
                last_error = exc
                if attempt < self.args.html_retries:
                    time.sleep(1)
        raise RuntimeError(f"Failed to fetch record {index}: {last_error}")

    def _call_llm(self, markdown: str) -> Dict[str, Any]:
        payload = {
            "model": self.args.llm_model_name,
            "messages": [{"role": "user", "content": build_prompt(markdown)}],
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.args.llm_api_key}", "Content-Type": "application/json"}
        last_error: Optional[Exception] = None
        for attempt in range(1, self.args.llm_retries + 1):
            try:
                response = _get_http_session().post(
                    self.llm_api_url, headers=headers, json=payload, timeout=self.args.llm_timeout
                )
                response.raise_for_status()
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                parsed = parse_llm_json(message)
                return {"parsed": parsed, "raw": message}
            except Exception as exc:  # pylint: disable=broad-except
                last_error = exc
                time.sleep(min(2 ** attempt, 5))
        raise RuntimeError(f"LLM call failed: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threaded QA batch generator with producer-consumer pattern")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=667648)
    parser.add_argument("--llm-workers", type=int, default=8, help="Number of LLM processing workers")
    parser.add_argument("--queue-size", type=int, default=50, help="Max size of HTML queue")
    parser.add_argument("--fetch-delay", type=float, default=0.2, help="Delay between HTML fetches (seconds)")

    # Wiki mode arguments
    parser.add_argument("--wiki", action="store_true", help="Use local wiki arrow files instead of HTTP service")
    parser.add_argument("--wiki-dir", type=str, default="wiki_en", 
                       help="Directory containing wiki arrow files")
    parser.add_argument("--wiki-max-files", type=int, default=None, 
                       help="Max number of arrow files to load (default: all)")

    # BBC mode arguments
    parser.add_argument("--bbc", action="store_true", help="Use BBC news parquet files")
    parser.add_argument("--bbc-dir", type=str,
                       default="/home/bince/data/datasets--RealTimeData--bbc_news_alltime",
                       help="Directory containing BBC news parquet files")
    parser.add_argument("--bbc-limit", type=int, default=None,
                       help="Limit number of BBC articles to process (from shuffled data)")

    parser.add_argument("--html-host", type=str, default="127.0.0.1")
    parser.add_argument("--html-port", type=int, default=8522)
    parser.add_argument("--html-base-url", type=str, help="Full base URL, overrides host/port")
    parser.add_argument("--html-timeout", type=int, default=30)
    parser.add_argument("--html-retries", type=int, default=3)

    parser.add_argument("--llm-host", type=str, default="127.0.0.1")
    parser.add_argument("--llm-port", type=int, default=8000)
    parser.add_argument("--llm-api-url", type=str, help="Full OpenAI-compatible endpoint, overrides host/port")
    parser.add_argument("--llm-api-key", type=str, required=True)
    parser.add_argument("--llm-model-name", type=str, required=True)
    parser.add_argument("--llm-timeout", type=int, default=600)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)

    parser.add_argument("--output-dir", type=str, default="qa_outputs")
    parser.add_argument("--resume", type=str, help="Resume from existing output directory (URL deduplication)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--random-index", action="store_true", help="Randomly shuffle processing order")
    parser.add_argument("--min-markdown-chars", type=int, default=200)
    parser.add_argument("--max-markdown-chars", type=int, default=30000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Initialize logging and mode-specific setup
    if args.bbc:
        logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG, format="%(asctime)s - %(message)s")
        logging.info("BBC mode enabled, loading news articles...")
    elif args.wiki:
        logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG, format="%(asctime)s - %(message)s")
        logging.info("Wiki mode enabled, loading articles first...")
    else:
        if not args.html_base_url:
            args.html_base_url = f"http://{args.html_host}:{args.html_port}"  # type: ignore[attr-defined]
    
    if not args.llm_api_url:
        args.llm_api_url = f"http://{args.llm_host}:{args.llm_port}/v1/chat/completions"  # type: ignore[attr-defined]

    if not args.wiki and not args.bbc:
        logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG, format="%(asctime)s - %(message)s")
    
    if args.bbc:
        mode_name = "BBC (local parquet files)"
    elif args.wiki:
        mode_name = "Wiki (local arrow files)"
    else:
        mode_name = "HTTP (remote service)"
    logging.info("Using producer-consumer pattern:")
    logging.info("  - Mode: %s", mode_name)
    logging.info("  - 1 fetcher (sequential)")
    logging.info("  - %s LLM workers (parallel)", args.llm_workers)
    logging.info("  - Queue size: %s", args.queue_size)
    if not args.wiki and not args.bbc:
        logging.info("  - Fetch delay: %s seconds", args.fetch_delay)

    if args.resume:
        resume_dir = Path(args.resume)
        try:
            processed_count = len(load_processed_urls(resume_dir))
            logging.info("Resume mode: will skip %s already processed URLs", processed_count)
        except FileNotFoundError as exc:
            logging.error(str(exc))
            return

    processor = BatchProcessor(args)
    
    # In BBC mode, adjust indices based on loaded records
    if args.bbc:
        total_bbc = len(processor.bbc_records)
        logging.info("BBC mode: %s articles to process", total_bbc)
        args.start_index = 0
        args.end_index = total_bbc - 1
    # In wiki mode, we can't determine total count without loading all files
    # So just use the end_index as specified
    elif args.wiki:
        logging.info("Wiki mode: processing indices %s to %s", args.start_index, args.end_index)

    start = min(args.start_index, args.end_index)
    end = max(args.start_index, args.end_index)
    total = end - start + 1
    indices = list(range(start, end + 1))
    
    # Shuffle indices if random mode is enabled
    if args.random_index:
        logging.info("Random mode enabled: shuffling %s indices", total)
        random.shuffle(indices)
        logging.info("Indices shuffled")

    with tqdm(total=total, desc="QA generation", unit="record") as pbar:
        # Start LLM worker threads
        llm_threads = []
        for i in range(args.llm_workers):
            t = threading.Thread(target=processor.llm_worker, args=(pbar,), name=f"LLM-Worker-{i}")
            t.start()
            llm_threads.append(t)
        
        # Start HTML fetcher (single thread, sequential)
        fetch_thread = threading.Thread(target=processor.fetch_html_worker, args=(indices, pbar), name="Fetcher")
        fetch_thread.start()
        
        # Wait for fetcher to complete
        fetch_thread.join()
        
        # Wait for all LLM workers to complete
        for t in llm_threads:
            t.join()

    logging.info(
        "Finished. Success=%s skipped=%s missing=%s errors=%s url_duplicates=%s",
        processor.stats["success"],
        processor.stats["skipped"],
        processor.stats["missing"],
        processor.stats["error"],
        processor.stats["url_duplicate"]
    )


if __name__ == "__main__":
    main()
