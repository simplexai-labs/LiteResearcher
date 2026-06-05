#!/usr/bin/env python3
"""Generate Q&A pairs from local Wikipedia Arrow files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset


def parse_json_response(content: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    direct = re.search(r'\{.*"qa_pairs".*\}', content, re.DOTALL)
    if direct:
        return json.loads(direct.group(0))

    return json.loads(content)


def generate_qa_pairs_with_local_llm(
    text_content: str,
    api_key: str,
    llm_host: str = "127.0.0.1",
    llm_port: int = 8000,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible local LLM API to generate Q&A pairs."""
    url = f"http://{llm_host}:{llm_port}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = f"""You are a data extraction and Q&A generation expert.

Create direct, self-contained question-answer pairs from the text below.

Rules:
1. Focus on concrete facts: numbers, dates, names, amounts, percentages, places, and named events.
2. Questions must include enough context to stand alone.
3. Questions must not say "according to the text", "in this article", or similar source references.
4. Answers should be short and factual.
5. Return only valid JSON with this shape:
{{
  "qa_pairs": [
    {{"question": "...", "answer": "..."}}
  ],
  "total_pairs": 1
}}

Text:
{text_content}
"""

    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4000,
    }
    if model_name:
        payload["model"] = model_name

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = parse_json_response(content)
        return {
            "success": True,
            "parsed_qa": parsed,
            "raw_response": content,
            "full_response": result,
            "model_name": model_name or "local-llm",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "raw_response": locals().get("content"),
            "full_response": locals().get("result"),
        }


def get_text_hash(text: str) -> str:
    """Generate a stable hash for output filenames."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_articles(wiki_dir: Path, max_files: int | None = None) -> list[dict[str, Any]]:
    """Load article rows from Arrow files."""
    arrow_files = sorted(glob(str(wiki_dir / "*.arrow")))
    if max_files is not None:
        arrow_files = arrow_files[:max_files]

    print(f"Reading Arrow files from: {wiki_dir}")
    print(f"Found {len(arrow_files)} Arrow files")

    articles: list[dict[str, Any]] = []
    for arrow_file in arrow_files:
        try:
            print(f"  loading {Path(arrow_file).name}")
            dataset = load_dataset("arrow", data_files=[arrow_file], split="train")
            articles.extend(dataset[i] for i in range(len(dataset)))
        except Exception as exc:
            print(f"  failed to load {Path(arrow_file).name}: {exc}")
    return articles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-dir", default="wiki_en")
    parser.add_argument("--output-dir", default="qa_outputs_wiki")
    parser.add_argument("--llm-host", default=os.getenv("LLM_HOST", "127.0.0.1"))
    parser.add_argument("--llm-port", type=int, default=int(os.getenv("LLM_PORT", "8000")))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--llm-model-name", default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--num-articles", type=int, default=0, help="Maximum articles to process. 0 means all.")
    parser.add_argument("--max-text-length", type=int, default=30000)
    parser.add_argument("--num-arrow-files", type=int, default=0, help="Maximum Arrow files to load. 0 means all.")
    args = parser.parse_args()
    if not args.llm_api_key:
        parser.error("--llm-api-key is required, or set LLM_API_KEY.")
    if not args.llm_model_name:
        parser.error("--llm-model-name is required, or set LLM_MODEL.")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    articles = load_articles(
        Path(args.wiki_dir),
        max_files=args.num_arrow_files or None,
    )
    if not articles:
        print("No articles loaded.")
        return

    num_to_process = len(articles) if args.num_articles == 0 else min(args.num_articles, len(articles))
    successful = 0
    failed = 0
    print(f"Processing {num_to_process} articles")

    for idx, article in enumerate(articles[:num_to_process]):
        article_id = article.get("id", idx)
        url = article.get("url", "")
        title = article.get("title", "")
        text = article.get("text", "")

        print("\n" + "=" * 80)
        print(f"Processing [{idx + 1}/{num_to_process}]")
        print(f"ID: {article_id}")
        print(f"Title: {title}")
        print(f"URL: {url}")

        if not text or len(text) < 100:
            print(f"Text too short ({len(text)} chars), skipping.")
            failed += 1
            continue

        original_length = len(text)
        if len(text) > args.max_text_length:
            text = text[: args.max_text_length] + "\n\n[Content truncated due to length.]"
            print(f"Text truncated from {original_length} to {args.max_text_length} characters")

        qa_result = generate_qa_pairs_with_local_llm(
            text,
            args.llm_api_key,
            args.llm_host,
            args.llm_port,
            args.llm_model_name,
        )
        if not qa_result.get("success"):
            print(f"Failed to generate Q&A pairs: {qa_result.get('error')}")
            failed += 1
            continue

        output_data = {
            "id": article_id,
            "url": url,
            "title": title,
            "text_hash": get_text_hash(text),
            "original_text_length": original_length,
            "processed_text_length": len(text),
            "qa_pairs": qa_result.get("parsed_qa", {}).get("qa_pairs", []),
            "total_pairs": qa_result.get("parsed_qa", {}).get("total_pairs", 0),
            "model_name": qa_result.get("model_name", "unknown"),
            "model_raw_output": qa_result.get("raw_response", ""),
            "generation_timestamp": datetime.now().isoformat(),
        }

        output_file = output_dir / f"{idx:06d}_{get_text_hash(text)[:8]}.json"
        output_file.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {output_file}")
        print(f"Generated {output_data['total_pairs']} Q&A pairs")
        successful += 1

    print("\n" + "=" * 80)
    print("Processing complete")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
