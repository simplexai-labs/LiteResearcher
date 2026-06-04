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

MAX_TURNS = 999  # keep every turn


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
    s = s.strip()
    parts = re.split(r"\n\n+", s)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 15]
    return parts[0] if parts else s


def think_excerpt(s: str) -> str:
    """Keep the entire think text verbatim."""
    return (s or "").strip()


def tool_summary(tc: dict):
    name = tc.get("name")
    if name == "search":
        queries = list(tc.get("queries") or [""])
        return {
            "name": "search",
            "args": {"queries": queries},
            "args_str": "",
            "queries": queries,
        }
    if name == "visit":
        urls = list(tc.get("urls") or [""])
        g = tc.get("goal") or ""
        return {
            "name": "visit",
            "args": {"urls": urls, "goal": shorten(g, 200)},
            "args_str": "",
            "urls": urls,
            "goal": shorten(g, 200),
        }
    return {"name": name or "tool", "args": {}, "args_str": ""}


def md_truncate(s: str, n: int) -> str:
    """Truncate markdown text without flattening paragraph breaks."""
    if not s:
        return ""
    s = s.strip()
    # collapse runs of >2 newlines, runs of >1 inline whitespace, but KEEP \n\n
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    if len(s) <= n:
        return s
    # cut at a newline if possible
    snippet = s[:n]
    cut = snippet.rfind("\n")
    if cut > n - 200:
        snippet = snippet[:cut]
    return snippet.rstrip(" ,.，。、\n") + "…"


def extract_result_excerpt(content: str, max_chars: int = 1400) -> str:
    if not content:
        return ""
    c = content
    c = re.sub(r"</?tool_response[^>]*>", "", c).strip()
    return md_truncate(c, max_chars)


def build_turns(steps):
    turns = []
    i = 0
    while i < len(steps):
        s = steps[i]
        if s.get("type") != "assistant":
            i += 1
            continue
        think = think_excerpt(s.get("think") or "")
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
            "think": think,
            "tool": ({**tool, "result": result} if tool else None),
        })
        i += 1
    return turns


def pick_turns(turns, n):
    """Keep a contiguous sequence covering the actual research arc.
    If too many, prefer first N-1 turns (initial reasoning + lookups) plus the last
    turn (final convergence) so the viewer sees the open-then-close shape.
    """
    if len(turns) <= n:
        return turns
    head = turns[: n - 1]
    tail = turns[-1:]
    return head + tail


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
            "question": shorten(c.get("question", ""), 320),
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
