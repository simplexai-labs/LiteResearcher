#!/usr/bin/env python3
"""
serper / Google 搜索结果 JSONL  →  LiteResearcher 导入格式

原始 serper 每行（扁平）：
    {"query": "...", "position": 1, "title": "...", "link": "https://...",
     "snippet": "...", "content": "正文..."}

导入端 data.py / rag_core 期望（Dolma 风格，嵌套）：
    {"id": "<url>", "text": "正文", "metadata": {"title": "...", "url": "https://..."}}

字段映射：
    link    → metadata.url   （同时作为 id 主键）
    title   → metadata.title
    content → text           （为空则回退到 snippet）

用法：
    python tools/convert_serper.py INPUT.jsonl OUTPUT.jsonl
    python tools/convert_serper.py INPUT.jsonl OUTPUT.jsonl --dedup
    python tools/convert_serper.py INPUT.jsonl OUTPUT.jsonl --text-field content --min-chars 50
"""

import argparse
import json
import sys


def convert(in_path, out_path, text_field="content", fallback_snippet=True,
            dedup=False, min_chars=1):
    seen = set()
    total = written = skipped_empty = skipped_dup = bad = 0

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue

            url = obj.get("link") or obj.get("url") or ""
            title = obj.get("title", "") or ""
            text = obj.get(text_field, "") or ""
            if not text.strip() and fallback_snippet:
                text = obj.get("snippet", "") or ""

            # 正文为空或过短则丢弃（导入端也会丢弃空 text）
            if len(text.strip()) < min_chars:
                skipped_empty += 1
                continue

            if dedup:
                key = url or text[:200]
                if key in seen:
                    skipped_dup += 1
                    continue
                seen.add(key)

            record = {
                "id": url or f"doc_{total}",
                "text": text,
                "metadata": {"title": title, "url": url},
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if total % 100000 == 0:
                print(f"  ...已处理 {total:,} 行，写出 {written:,}", file=sys.stderr)

    print("=" * 50)
    print(f"✅ 转换完成: {out_path}")
    print(f"   读取行数:   {total:,}")
    print(f"   写出记录:   {written:,}")
    print(f"   跳过(空/短): {skipped_empty:,}")
    if dedup:
        print(f"   跳过(重复): {skipped_dup:,}")
    if bad:
        print(f"   解析失败:   {bad:,}")
    print("=" * 50)


def main():
    p = argparse.ArgumentParser(description="serper JSONL → LiteResearcher 导入格式")
    p.add_argument("input", help="原始 serper jsonl 路径")
    p.add_argument("output", help="输出 jsonl 路径")
    p.add_argument("--text-field", default="content",
                   help="作为正文的字段名（默认 content）")
    p.add_argument("--no-fallback-snippet", action="store_true",
                   help="正文为空时不回退到 snippet")
    p.add_argument("--dedup", action="store_true",
                   help="按 URL 去重（serper 数据常有重复 link）")
    p.add_argument("--min-chars", type=int, default=1,
                   help="正文最小字符数，低于则丢弃（默认 1）")
    args = p.parse_args()

    convert(
        args.input, args.output,
        text_field=args.text_field,
        fallback_snippet=not args.no_fallback_snippet,
        dedup=args.dedup,
        min_chars=args.min_chars,
    )


if __name__ == "__main__":
    main()
