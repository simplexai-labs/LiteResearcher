#!/usr/bin/env python3
"""Run extractor across many checkpoints from S1 and S2 rollouts and bench results."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from extract_behaviors import (
    load_rollout_step, load_bench_step, aggregate_metrics,
)

ROLLOUT_ROOT = Path("/share/project/wanli/Search_Agent/verl/rollout_trajectory")
BENCH_ROOT = Path("/share/project/wanli/Search_Agent/DeepResearch/bench_results/qwen3-4B-RL")

S1_ROLL = ROLLOUT_ROOT / "qwen3_deepresearch_tis_rl_onpolicy_bs128_local_rag_only_token-mean-seq-mean-temp_0.7_length_32k_nokl"
S2_ROLL = ROLLOUT_ROOT / "qwen3_deepresearch_tis_rl_stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k"

S1_BENCH = BENCH_ROOT / "onpolicy_bs128_local_rag_only_token-mean-seq-mean-temp_0.7_length_32k_nokl"
S2_BENCH = BENCH_ROOT / "stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k"


def find_rollout_file(root: Path, step: int):
    """Find <step>.jsonl across timestamped subdirs (first non-empty)."""
    candidates = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        fp = sub / f"{step}.jsonl"
        if fp.exists() and fp.stat().st_size > 0:
            candidates.append(fp)
    # prefer the latest (largest mtime) - usually the most recent run
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_bench_dir(root: Path, step: int):
    """Find global_step_<step>/GAIA_pass@4/ckpt dir."""
    cands = [
        root / f"global_step_{step}" / "GAIA_pass@4" / "ckpt",
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def get_bench_summary(root: Path, step: int):
    sp = root / f"global_step_{step}" / "GAIA_pass@4" / "summary.json"
    if not sp.exists():
        return None
    with open(sp) as f:
        d = json.load(f)
    return {
        "avg_accuracy": d.get("accuracy") or d.get("pass_at_k_stats", {}).get("avg_accuracy"),
        "pass_at_k_accuracy": d.get("pass_at_k_stats", {}).get("pass_at_k_accuracy"),
    }


def collect_stage(stage_name, roll_root, bench_root, steps, max_rollout=400, max_bench=None):
    rows = []
    for s in steps:
        row = {"stage": stage_name, "step": s}
        rf = find_rollout_file(roll_root, s)
        if rf:
            try:
                per = load_rollout_step(str(rf), max_lines=max_rollout)
                ag = aggregate_metrics(per)
                row["rollout_path"] = str(rf)
                row["rollout_n"] = len(per)
                for k, v in ag.items():
                    row["roll_" + k] = v
                print(f"[{stage_name}] step {s} rollout n={len(per)} corr={ag.get('correct_frac', 0):.3f}", flush=True)
            except Exception as e:
                print(f"[{stage_name}] step {s} rollout ERR: {e}", flush=True)
        bd = find_bench_dir(bench_root, s)
        if bd:
            try:
                per_b = load_bench_step(str(bd), max_files=max_bench)
                ag_b = aggregate_metrics(per_b)
                row["bench_path"] = str(bd)
                row["bench_n"] = len(per_b)
                for k, v in ag_b.items():
                    row["bench_" + k] = v
                summ = get_bench_summary(bench_root, s) or {}
                row["bench_summary"] = summ
                print(f"[{stage_name}] step {s} bench n={len(per_b)} acc={ag_b.get('correct_frac', 0):.3f} pass@4={summ.get('pass_at_k_accuracy')}", flush=True)
            except Exception as e:
                print(f"[{stage_name}] step {s} bench ERR: {e}", flush=True)
        rows.append(row)
    return rows


if __name__ == "__main__":
    # User wants the continuous training path: S1 0->220, then S2 continues.
    # Use S1 0..220 only (the abandoned >220 branch is NOT shown).
    # S2 will be shifted by +220 on the global axis at plot time.
    S1_STEPS = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]
    S2_STEPS = [1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240,
                260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480,
                500, 520, 540, 570]

    out = {}
    out["stage1"] = collect_stage("stage1", S1_ROLL, S1_BENCH, S1_STEPS,
                                   max_rollout=300, max_bench=None)
    out["stage2"] = collect_stage("stage2", S2_ROLL, S2_BENCH, S2_STEPS,
                                   max_rollout=300, max_bench=None)
    with open("behavior_timeline.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved behavior_timeline.json")
