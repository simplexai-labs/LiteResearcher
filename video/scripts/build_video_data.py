"""
Extract per-turn streaming data for video.

For each (benchmark, id), produce:
{
  benchmark, id, question, reference_answer, final_answer, judge_correct, stats,
  turns: [
    {
      think: "...",           # ≤220 chars, will type-stream in video
      tool: {                  # may be null on the final turn
        name: "search" | "visit",
        args: {...},           # short pretty form
        result: "..."          # short excerpt of tool_response
      }
    },
    ...
  ]
}
"""

import json, os, re
from urllib.parse import urlparse

PICKS = [("GAIA", 56), ("Xbench", None), ("WebwalkerQA", None)]
CASES_DIR = "/Users/wanli/Downloads/Literesearcher_release/docs/cases/cases"
OUT = "/Users/wanli/Downloads/Literesearcher_release/video/src/data/cases.json"

MAX_TURNS = 4  # 4 think→tool turns fits comfortably in ~40s


def host(u: str) -> str:
    try:
        h = urlparse(u).netloc
        return h.replace("www.", "") or u
    except Exception:
        return u


def shorten(s: str, n: int) -> str:
    if not s:
        return ""
    s = s.strip()
    # collapse repeated whitespace
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= n else s[: n - 1].rstrip(" ,.，。、") + "…"


def first_para(s: str) -> str:
    if not s:
        return ""
    # take first non-trivial sentence/paragraph
    s = s.strip()
    parts = re.split(r"\n\n|。\s*|\.\s+", s)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 15]
    return parts[0] if parts else s


def tool_summary(tc: dict):
    name = tc.get("name")
    if name == "search":
        q = (tc.get("queries") or [""])[0]
        return {
            "name": "search",
            "args": {"query": shorten(q, 80)},
            "args_str": f'query="{shorten(q, 80)}"',
        }
    if name == "visit":
        urls = tc.get("urls") or [""]
        u = urls[0]
        g = tc.get("goal") or ""
        return {
            "name": "visit",
            "args": {"url": u, "domain": host(u), "goal": shorten(g, 120)},
            "args_str": f'url="{shorten(u, 80)}"',
        }
    return {"name": name or "tool", "args": {}, "args_str": ""}


def extract_result_excerpt(content: str, max_chars: int = 220) -> str:
    if not content:
        return ""
    # strip tool_response wrapper and code fences
    c = content
    c = re.sub(r"</?tool_response[^>]*>", "", c).strip()
    # remove markdown link syntax (keep visible text)
    c = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", c)
    # collapse whitespace
    c = re.sub(r"\s+", " ", c).strip()
    return shorten(c, max_chars)


def build_turns(steps):
    turns = []
    i = 0
    while i < len(steps):
        s = steps[i]
        if s.get("type") != "assistant":
            i += 1
            continue
        think = first_para(s.get("think") or "")
        tcs = s.get("tool_calls") or []
        tool = None
        result = ""
        if tcs:
            tool = tool_summary(tcs[0])
            # find next tool_response
            j = i + 1
            while j < len(steps) and steps[j].get("type") != "tool_response":
                j += 1
            if j < len(steps):
                result = extract_result_excerpt(steps[j].get("content") or "")
        turns.append({
            "think": shorten(think, 220),
            "tool": ({**tool, "result": result} if tool else None),
        })
        i += 1
    return turns


def pick_turns(turns, n):
    """Pick n diverse turns: keep first 2 (initial reasoning + first lookup),
    then a couple later visits, then the very last think."""
    if len(turns) <= n:
        return turns
    head = turns[:2]
    mid = [t for t in turns[2:-1] if t["tool"] and t["tool"]["name"] == "visit"][:1]
    tail = turns[-1:]
    out, seen = [], set()
    for t in head + mid + tail:
        key = (t["think"], json.dumps(t["tool"], ensure_ascii=False) if t["tool"] else "")
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out[:n]


def main():
    out_cases = []
    for bench, want_id in PICKS:
        path = os.path.join(CASES_DIR, f"{bench}.json")
        d = json.load(open(path))
        cases = d["cases"]
        c = next((x for x in cases if x["id"] == want_id), cases[0])
        turns = pick_turns(build_turns(c.get("steps") or []), MAX_TURNS)
        out_cases.append({
            "benchmark": bench,
            "id": c["id"],
            "question": shorten(c.get("question", ""), 240),
            "reference_answer": (c.get("reference_answer") or "").strip(),
            "final_answer": (c.get("final_answer") or "").strip(),
            "judge_correct": c.get("judge_correct", True),
            "stats": c.get("stats") or {},
            "turns": turns,
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"cases": out_cases}, f, ensure_ascii=False, indent=2)
    print(f"wrote {OUT}")
    for c in out_cases:
        ts = c["turns"]
        print(f"  {c['benchmark']}#{c['id']}: {len(ts)} turns, final_answer {len(c['final_answer'])} chars")


if __name__ == "__main__":
    main()
