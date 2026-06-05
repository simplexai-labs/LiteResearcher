#!/usr/bin/env python3
"""Mine representative trajectories at key checkpoints for paper-grade examples."""
import json, re, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from extract_behaviors import (
    parse_output_tools, TC_RE, THINK_RE, ANS_RE,
)

ROLLOUT_ROOT = Path("/share/project/wanli/Search_Agent/verl/rollout_trajectory")
S1_ROLL = ROLLOUT_ROOT / "qwen3_deepresearch_tis_rl_onpolicy_bs128_local_rag_only_token-mean-seq-mean-temp_0.7_length_32k_nokl"
S2_ROLL = ROLLOUT_ROOT / "qwen3_deepresearch_tis_rl_stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k"


def find_file(root, step):
    cands = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            fp = sub / f"{step}.jsonl"
            if fp.exists() and fp.stat().st_size > 0:
                cands.append(fp)
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def trim(text, n=600):
    text = re.sub(r"\s+", " ", text).strip()
    return text[:n] + ("...[trim]" if len(text) > n else "")


def sample_traj(fp, n=4, pick="correct_short", seed=0):
    rows = []
    with open(fp) as f:
        for line in f:
            try: rows.append(json.loads(line))
            except: pass
    if pick == "correct_short":
        rows = [r for r in rows if r.get("correct")]
        rows.sort(key=lambda r: r.get("output_tokens", 1e9))
    elif pick == "incorrect_no_extract":
        rows = [r for r in rows if r.get("method") == "no_extraction"]
    elif pick == "long_reasoning":
        rows = [r for r in rows if r.get("correct")]
        rows.sort(key=lambda r: -r.get("total_turns", 0))
    elif pick == "first":
        pass
    return rows[:n]


def summarize_traj(d):
    out = d.get("output", "")
    tcs = parse_output_tools(out)
    queries = []
    for tc in tcs:
        q = tc.get("arguments", {}).get("query", None)
        if isinstance(q, list): queries.extend(q)
        elif isinstance(q, str): queries.append(q)
    thinks = THINK_RE.findall(out)
    ans = ANS_RE.findall(out)
    return {
        "question": trim(d.get("input", "").split("user\n")[-1] if "user\n" in d.get("input", "") else d.get("input", ""), 250),
        "score": d.get("score"),
        "correct": d.get("correct"),
        "method": d.get("method"),
        "output_tokens": d.get("output_tokens"),
        "total_turns": d.get("total_turns"),
        "pred_ans": d.get("pred_ans"),
        "n_tool_calls": len(tcs),
        "n_queries": len(queries),
        "first_3_queries": queries[:3],
        "all_tool_names": [tc.get("name", "?") for tc in tcs],
        "first_think": trim(thinks[0], 400) if thinks else "(no <think>)",
        "last_think": trim(thinks[-1], 400) if len(thinks) > 1 else "(only 1 think)",
        "answer": ans[0] if ans else "(NO <answer> TAG)",
    }


KEY_STEPS = {
    "S1_step1_baseline":   (S1_ROLL, 1,   "first"),
    "S1_step100_converge": (S1_ROLL, 100, "correct_short"),
    "S1_step220_ckpt":     (S1_ROLL, 220, "correct_short"),
    "S1_step480_peak":     (S1_ROLL, 480, "correct_short"),
    "S1_step510_collapse": (S1_ROLL, 510, "incorrect_no_extract"),
    "S1_step750_post_collapse": (S1_ROLL, 750, "correct_short"),
    "S2_step1_restart":    (S2_ROLL, 1,   "first"),
    "S2_step100":          (S2_ROLL, 100, "correct_short"),
    "S2_step400_strong":   (S2_ROLL, 400, "long_reasoning"),
    "S2_step570_final":    (S2_ROLL, 570, "long_reasoning"),
}

out = {}
for label, (root, step, pick) in KEY_STEPS.items():
    fp = find_file(root, step)
    if not fp:
        print(f"{label}: NO FILE", flush=True); continue
    samples = sample_traj(fp, n=3, pick=pick)
    out[label] = [summarize_traj(d) for d in samples]
    print(f"{label}: {fp.name} got {len(samples)} samples", flush=True)

with open("trajectory_examples.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("Saved trajectory_examples.json")
