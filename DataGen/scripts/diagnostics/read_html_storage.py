#!/usr/bin/env python3
"""Read one URL from an HTMLStorage directory and optionally export markdown."""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from typing import Iterable

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-dir", required=True, help="HTMLStorage directory.")
    parser.add_argument("--url", required=True, help="URL/key to read from storage.")
    parser.add_argument("--output", default="read_html.md", help="Markdown output path.")
    parser.add_argument("--preview-chars", type=int, default=2000, help="Characters to print from markdown.")
    args = parser.parse_args()
    if not Path(args.storage_dir).exists():
        parser.error(f"--storage-dir does not exist: {args.storage_dir}")
    return args


def main() -> None:
    args = parse_args()
    storage = HTMLStorage(data_dirs=[args.storage_dir])

    try:
        print(f"URL: {args.url}")
        result = storage.get(args.url)
        if not result:
            print("No cached content found for this URL.")
            return

        html_blob = result.get("html", "")
        markdown = html_to_markdown(html_blob)
        output_path = Path(args.output)
        output_path.write_text(markdown, encoding="utf-8")

        print("Read succeeded.")
        print(f"HTML length: {len(html_blob)} bytes")
        print(f"Markdown length: {len(markdown)} characters")
        print(f"Markdown saved to: {output_path}")
        print(f"Storage UID: {result.get('storage_uid', '')}")
        if args.preview_chars > 0:
            print("\n================ Markdown Preview ================\n")
            print(markdown[: args.preview_chars])
            print("\n==================================================\n")
    finally:
        storage.close()


if __name__ == "__main__":
    main()
