#!/usr/bin/env python3
"""Generate direct information-seeking Q&A pairs from cached HTML."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore

from directqa.html_storage import HTMLStorage


if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def _join_markdown_chunks(chunks: Iterable[str]) -> str:
    return "".join(chunk for chunk in chunks if chunk)


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
            return "".join(render_node(child) for child in node.find_all("li", recursive=False)) + "\n"
        if name == "li":
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"- {content}\n" if content else ""
        if name == "a":
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            href = node.get("href", "").strip()
            return f"[{content}]({href})" if content and href else content
        if name in {"strong", "b"}:
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"**{content}**" if content else ""
        if name in {"em", "i"}:
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"*{content}*" if content else ""
        return _join_markdown_chunks(render_node(child) for child in node.children)

    body = soup.body or soup
    markdown = _join_markdown_chunks(render_node(child) for child in body.children)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


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
    return f"""You are a data extraction and Q&A generation expert. Analyze the following webpage content in markdown format and extract specific, factual data points.

For each concrete data point you find, create a high-quality question-answer pair.

Requirements:
1. Questions must be specific and self-contained.
2. Answers must be concise and factual, such as numbers, names, dates, amounts, percentages, or short named entities.
3. Focus on extractable, verifiable data points, not opinions or general statements.
4. Do not ask about website navigation, URLs, email addresses, footer text, or contact information.

Return only a JSON object with this structure:
{{
  "qa_pairs": [
    {{"question": "...", "answer": "..."}}
  ],
  "total_pairs": 1
}}

Content:
{markdown_content}
"""


def parse_llm_json(content: str) -> Dict[str, Any]:
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    json_match = re.search(r"\{[^{}]*\"qa_pairs\"[^{}]*\[.*?\][^{}]*\}", content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    return json.loads(content)


def generate_qa_pairs(
    markdown_content: str,
    api_key: str,
    api_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Direct Information Seeking Data Generation",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": build_prompt(markdown_content)}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = parse_llm_json(content)
        return {
            "success": True,
            "parsed_qa": parsed,
            "raw_response": content,
            "full_response": result,
            "model_name": model_name,
        }
    except json.JSONDecodeError as exc:
        return {
            "success": False,
            "error": f"JSON parsing error: {exc}",
            "raw_response": content if "content" in locals() else None,
            "full_response": result if "result" in locals() else None,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "raw_response": None, "full_response": None}


def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def extract_urls(checkpoint_data: Any) -> List[str]:
    urls: List[str] = []
    if isinstance(checkpoint_data, list):
        urls = [item.get("url") or item.get("link") for item in checkpoint_data if isinstance(item, dict)]
    elif isinstance(checkpoint_data, dict):
        for key in ("urls", "links", "data", "results"):
            data = checkpoint_data.get(key)
            if isinstance(data, list):
                urls = [item.get("url") or item.get("link") if isinstance(item, dict) else item for item in data]
                break
    return [url for url in urls if isinstance(url, str) and url.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-json", required=True, help="JSON file containing URLs.")
    parser.add_argument("--storage-dir", required=True, help="HTMLStorage directory, such as storage_1234.")
    parser.add_argument("--output-dir", default="qa_outputs", help="Directory for output JSON files.")
    parser.add_argument("--num-urls", type=int, default=10, help="Maximum URLs to process. Use 0 for all URLs.")
    parser.add_argument("--min-markdown-chars", type=int, default=100)
    parser.add_argument("--max-markdown-chars", type=int, default=30000)
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"), help="Defaults to OPENROUTER_API_KEY.")
    parser.add_argument(
        "--api-url",
        default=os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions"),
        help="OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
        help="Model name sent to the chat API.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key is required, or set OPENROUTER_API_KEY.")
    if not Path(args.checkpoint_json).exists():
        parser.error(f"--checkpoint-json does not exist: {args.checkpoint_json}")
    if not Path(args.storage_dir).exists():
        parser.error(f"--storage-dir does not exist: {args.storage_dir}")
    return args


def main() -> None:
    args = parse_args()
    checkpoint_json = Path(args.checkpoint_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Reading checkpoint JSON from: {checkpoint_json}", flush=True)
    try:
        checkpoint_data = json.loads(checkpoint_json.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[error] Failed to read checkpoint JSON: {exc}", flush=True)
        return

    urls = extract_urls(checkpoint_data)
    if args.num_urls > 0:
        urls = urls[: args.num_urls]
    if not urls:
        print("[warn] No URLs found in checkpoint JSON. Please check the file structure.", flush=True)
        if isinstance(checkpoint_data, dict):
            print(f"Keys: {list(checkpoint_data.keys())[:10]}", flush=True)
        return

    print(f"[ok] Found {len(urls)} URLs to process", flush=True)
    print(f"[info] Initializing HTML Storage from: {args.storage_dir}", flush=True)
    storage = HTMLStorage(data_dirs=[args.storage_dir])

    successful = 0
    failed = 0
    try:
        for idx, url in enumerate(urls, 1):
            print(f"\n{'=' * 80}", flush=True)
            print(f"Processing [{idx}/{len(urls)}]: {url}", flush=True)
            print(f"{'=' * 80}", flush=True)

            result = storage.get(url)
            if not result:
                print("[warn] URL not found in storage, skipping.", flush=True)
                failed += 1
                continue

            html_blob = result.get("html", "")
            if not html_blob:
                print("[warn] Empty HTML content, skipping.", flush=True)
                failed += 1
                continue

            print(f"[ok] Retrieved HTML ({len(html_blob)} bytes)", flush=True)
            markdown = html_to_markdown(html_blob)
            print(f"[ok] Converted to Markdown ({len(markdown)} characters)", flush=True)

            if len(markdown) < args.min_markdown_chars:
                print("[warn] Markdown too short, skipping.", flush=True)
                failed += 1
                continue

            if len(markdown) > args.max_markdown_chars:
                markdown = markdown[: args.max_markdown_chars] + "\n\n[Content truncated due to length...]"
                print(f"[warn] Markdown truncated to {args.max_markdown_chars} characters", flush=True)

            print("[info] Calling chat API to generate Q&A pairs.", flush=True)
            qa_result = generate_qa_pairs(
                markdown,
                api_key=args.api_key,
                api_url=args.api_url,
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            if not qa_result.get("success"):
                print(f"[error] Failed to generate Q&A pairs: {qa_result.get('error')}", flush=True)
                failed += 1
                continue

            parsed = qa_result.get("parsed_qa", {})
            output_data = {
                "url": url,
                "url_hash": get_url_hash(url),
                "storage_uid": result.get("storage_uid", ""),
                "markdown_length": len(markdown),
                "html_length": len(html_blob),
                "qa_pairs": parsed.get("qa_pairs", []),
                "total_pairs": parsed.get("total_pairs", 0),
                "model_name": qa_result.get("model_name", "unknown"),
                "model_raw_output": qa_result.get("raw_response", ""),
                "html_timestamp": result.get("timestamp", ""),
                "generation_timestamp": datetime.now().isoformat(),
            }

            output_file = output_dir / f"{output_data['url_hash']}.json"
            output_file.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] Saved to: {output_file}", flush=True)
            print(f"[info] Generated {output_data['total_pairs']} Q&A pairs", flush=True)

            if output_data["qa_pairs"]:
                print("\nQ&A preview:", flush=True)
                for i, qa in enumerate(output_data["qa_pairs"][:3], 1):
                    print(f"\n  {i}. Q: {qa.get('question', 'N/A')}", flush=True)
                    print(f"     A: {qa.get('answer', 'N/A')}", flush=True)
                if len(output_data["qa_pairs"]) > 3:
                    print(f"\n  ... and {len(output_data['qa_pairs']) - 3} more pairs", flush=True)

            successful += 1
    finally:
        storage.close()

    print(f"\n{'=' * 80}", flush=True)
    print("Processing complete.", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"Successful: {successful}", flush=True)
    print(f"Failed: {failed}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
