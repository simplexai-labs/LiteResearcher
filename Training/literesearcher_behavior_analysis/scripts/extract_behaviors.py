#!/usr/bin/env python3
"""Extract per-trajectory behavioral metrics from rollout JSONL or bench JSON files.

Produces aggregate metrics per step for plotting behavior evolution.
"""
import json
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

# ---------- tool call extraction from raw output ----------
TC_RE = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>")
THINK_RE = re.compile(r"<think>([\s\S]*?)</think>")
ANS_RE = re.compile(r"<answer>([\s\S]*?)</answer>")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
QUOTED_RE = re.compile(r'"[^"]+"')
HEDGE_TOKENS = [
    "wait", "actually", "hmm", "let me reconsider", "let me re-check",
    "let me think again", "i was wrong", "but actually", "重新", "等等",
    "嗯", "让我再", "let me try", "let me think", "on second thought",
]


def parse_output_tools(output: str):
    tcs = []
    for m in TC_RE.finditer(output):
        raw = m.group(1)
        try:
            tc = json.loads(raw)
        except Exception:
            # try fuzzy: many partial JSONs - just record raw
            tcs.append({"name": "__unparsable__", "arguments": {}})
            continue
        tcs.append(tc)
    return tcs


def extract_metrics_from_traj(output: str, score: float, correct: bool,
                              output_tokens: int, total_turns: int,
                              pred_ans=None, method=None):
    """Compute per-trajectory metrics from a single trajectory."""
    m = {}
    m["correct"] = bool(correct)
    m["score"] = float(score) if score is not None else 0.0
    m["output_tokens"] = int(output_tokens) if output_tokens else len(output)
    m["total_turns"] = int(total_turns) if total_turns else 0
    m["truncated"] = m["output_tokens"] >= 31500  # 32k cap
    m["has_answer_tag"] = bool(ANS_RE.search(output))
    m["no_extraction"] = (method == "no_extraction") if method else (not m["has_answer_tag"])

    tcs = parse_output_tools(output)
    m["num_tool_calls"] = len(tcs)
    counts = Counter()
    queries = []
    for tc in tcs:
        name = tc.get("name", "unknown")
        counts[name] += 1
        args = tc.get("arguments", {}) or {}
        q = args.get("query", None)
        if q is None:
            continue
        if isinstance(q, str):
            queries.append(q)
        elif isinstance(q, list):
            queries.extend([str(x) for x in q])

    m["num_search"] = counts.get("search", 0)
    m["num_visit"] = counts.get("visit", 0) + counts.get("visit_url", 0) + counts.get("browse", 0)
    m["num_python"] = counts.get("python", 0) + counts.get("execute_python", 0)
    m["num_other_tools"] = sum(v for k, v in counts.items()
                                if k not in {"search", "visit", "visit_url", "browse",
                                             "python", "execute_python", "__unparsable__"})

    # NEW DERIVED METRICS
    sv = m["num_search"] + m["num_visit"]
    m["browse_ratio"] = m["num_visit"] / sv if sv > 0 else 0.0  # visit / (search+visit)
    m["any_python"] = 1.0 if m["num_python"] > 0 else 0.0
    m["any_visit"]  = 1.0 if m["num_visit"]  > 0 else 0.0

    m["num_queries"] = len(queries)
    m["num_unique_queries"] = len(set(queries))
    if queries:
        m["avg_query_len"] = sum(len(q) for q in queries) / len(queries)
        m["queries_per_search_call"] = m["num_queries"] / max(m["num_search"], 1)
        m["quoted_query_frac"] = sum(1 for q in queries if QUOTED_RE.search(q)) / len(queries)
    else:
        m["avg_query_len"] = 0.0
        m["queries_per_search_call"] = 0.0
        m["quoted_query_frac"] = 0.0

    # think analysis
    thinks = THINK_RE.findall(output)
    m["num_think"] = len(thinks)
    think_text = "\n".join(thinks)
    m["think_chars"] = len(think_text)
    m["think_density"] = len(think_text) / max(len(output), 1)  # think fraction of output
    m["think_chinese_frac"] = (
        sum(1 for c in think_text if CHINESE_RE.match(c)) / max(len(think_text), 1)
    )
    m["full_chinese_frac"] = (
        sum(1 for c in output if CHINESE_RE.match(c)) / max(len(output), 1)
    )
    low = output.lower()
    m["hedge_count"] = sum(low.count(tok) for tok in HEDGE_TOKENS)
    # extremely-rough repetition: count of largest substring repeating
    # use 80-char window unique fraction
    if len(output) > 5000:
        windows = [output[i:i+80] for i in range(0, len(output)-80, 80)]
        m["repetition_score"] = 1.0 - len(set(windows)) / max(len(windows), 1)
    else:
        m["repetition_score"] = 0.0

    # gibberish heuristic: tail of output looks random
    tail = output[-1500:] if len(output) > 1500 else output
    # ratio of non-ascii printable chars in tail
    if tail:
        weird = sum(1 for c in tail if not (32 <= ord(c) < 127 or c in "\n\r\t"))
        m["tail_weird_frac"] = weird / len(tail)
    else:
        m["tail_weird_frac"] = 0.0

    return m


def aggregate_metrics(per_traj):
    """Aggregate list of per-trajectory metric dicts into mean/frac scalars."""
    if not per_traj:
        return {}
    keys = per_traj[0].keys()
    agg = {"n_traj": len(per_traj)}
    for k in keys:
        vals = [m[k] for m in per_traj if m.get(k) is not None]
        if not vals:
            continue
        if isinstance(vals[0], bool):
            agg[k + "_frac"] = sum(vals) / len(vals)
        elif isinstance(vals[0], (int, float)):
            agg[k + "_mean"] = sum(vals) / len(vals)
    return agg


def load_rollout_step(jsonl_path, max_lines=None):
    """Read rollout JSONL; return list of per-traj metrics."""
    per_traj = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            per_traj.append(extract_metrics_from_traj(
                output=d.get("output", ""),
                score=d.get("score"),
                correct=d.get("correct", False),
                output_tokens=d.get("output_tokens"),
                total_turns=d.get("total_turns"),
                pred_ans=d.get("pred_ans"),
                method=d.get("method"),
            ))
    return per_traj


def load_bench_step(ckpt_dir, max_files=None):
    """Read bench result JSONs from a GAIA_pass@k/ckpt/ directory."""
    per_traj = []
    files = sorted(Path(ckpt_dir).glob("result_*.json"))
    if max_files:
        files = files[:max_files]
    for fp in files:
        try:
            with open(fp) as f:
                d = json.load(f)
        except Exception:
            continue
        rec = d.get("record") or d
        msgs = rec.get("messages", [])
        # Reconstruct full assistant output by concatenation
        assistant_parts = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
        output = "\n".join(assistant_parts)
        # use tool_interactions for tool counts (more reliable than regex)
        ti = rec.get("tool_interactions", [])
        # synthesize a fake output enriched with tool_calls so parser sees them
        synth = output
        if not TC_RE.search(synth) and ti:
            for t in ti:
                tc = t.get("tool_call")
                if tc:
                    synth += "\n<tool_call>" + json.dumps(tc) + "</tool_call>"
        tok = rec.get("token_stats", {}) or {}
        judge = rec.get("judge", {}) or {}
        m = extract_metrics_from_traj(
            output=synth,
            score=1.0 if judge.get("correct") else 0.0,
            correct=judge.get("correct", False),
            output_tokens=tok.get("assistant_tokens") or tok.get("total_tokens") or len(output) // 4,
            total_turns=len(rec.get("turn_times", [])),
            pred_ans=rec.get("final_answer"),
            method=rec.get("final_answer_source"),
        )
        per_traj.append(m)
    return per_traj


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: extract_behaviors.py rollout|bench PATH [max_lines]", file=sys.stderr)
        sys.exit(1)
    kind, path = sys.argv[1], sys.argv[2]
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else None
    if kind == "rollout":
        per = load_rollout_step(path, max_lines=mx)
    else:
        per = load_bench_step(path, max_files=mx)
    print(json.dumps(aggregate_metrics(per), indent=2))
