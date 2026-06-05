#!/usr/bin/env python3
"""Generate behavior-evolution figures from behavior_timeline.json."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = json.load(open(os.path.join(os.path.dirname(__file__), "behavior_timeline.json")))


def series(stage, key):
    rows = D[stage]
    xs = [r["step"] for r in rows if r.get(key) is not None]
    ys = [r[key] for r in rows if r.get(key) is not None]
    return xs, ys


COLORS = {"stage1": "#1f77b4", "stage2": "#d62728"}
COLLAPSE_BAND = (490, 540)


def add_collapse_band(ax, stage):
    if stage == "stage1":
        ax.axvspan(*COLLAPSE_BAND, color="orange", alpha=0.12,
                   label="S1 collapse window" if not ax.get_legend() else None)


def plot_panel(ax, key, ylabel, title, stages=("stage1", "stage2"), marker="o"):
    for st in stages:
        xs, ys = series(st, key)
        if xs:
            ax.plot(xs, ys, marker=marker, color=COLORS[st], label=st.upper(),
                    linewidth=1.8, markersize=4)
    if "stage1" in stages:
        ax.axvspan(*COLLAPSE_BAND, color="orange", alpha=0.10)
    ax.axvline(220, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


# ===== Figure 1: Accuracy & length / turns / truncation =====
fig, axes = plt.subplots(3, 2, figsize=(13, 11))
plot_panel(axes[0][0], "roll_correct_frac",   "rollout reward",      "(a) Sim train accuracy (correct frac)")
plot_panel(axes[0][1], "bench_correct_frac",  "GAIA pass@1 acc",     "(b) Online GAIA pass@1 accuracy (412 tasks)")
plot_panel(axes[1][0], "roll_output_tokens_mean", "mean output tokens", "(c) Mean response length (rollout)")
plot_panel(axes[1][1], "bench_output_tokens_mean", "mean output tokens", "(d) Mean response length (GAIA bench)")
plot_panel(axes[2][0], "roll_total_turns_mean", "mean turns",        "(e) Mean turns per question (rollout)")
plot_panel(axes[2][1], "bench_total_turns_mean", "mean turns",       "(f) Mean turns per question (GAIA bench)")
plt.suptitle("Figure 1 · Accuracy & response budget across stages (vertical line = S1 ckpt 220 used to start S2)", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig1_acc_length.png"), dpi=130, bbox_inches="tight")
plt.close()


# ===== Figure 2: Tool-usage shape =====
fig, axes = plt.subplots(3, 2, figsize=(13, 11))
plot_panel(axes[0][0], "roll_num_search_mean", "search calls / traj", "(a) Search tool calls per trajectory (rollout)")
plot_panel(axes[0][1], "roll_num_visit_mean", "visit calls / traj",   "(b) Visit/browse tool calls per trajectory (rollout)")
plot_panel(axes[1][0], "roll_num_queries_mean", "queries / traj",     "(c) Total queries emitted per trajectory")
plot_panel(axes[1][1], "roll_queries_per_search_call_mean", "queries / search call",
           "(d) Query-batching density (multi-query per search)")
plot_panel(axes[2][0], "roll_avg_query_len_mean", "avg query length (chars)", "(e) Average query length")
plot_panel(axes[2][1], "roll_quoted_query_frac_mean", "frac queries with \"...\"",
           "(f) Fraction of queries using exact-phrase quotes")
plt.suptitle("Figure 2 · Tool usage and search-query shape", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig2_tools.png"), dpi=130, bbox_inches="tight")
plt.close()


# ===== Figure 3: Truncation, answer-format breakdown =====
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
plot_panel(axes[0][0], "roll_truncated_frac", "truncated frac",
           "(a) Truncation rate (output ≥ 31.5k tokens)")
plot_panel(axes[0][1], "roll_no_extraction_frac", "no-answer-tag frac",
           "(b) Trajectories missing <answer> tag (forced score=0)")
plot_panel(axes[1][0], "bench_truncated_frac", "truncated frac (bench)",
           "(c) Bench truncation rate (real GAIA inference)")
plot_panel(axes[1][1], "bench_no_extraction_frac", "no-answer frac (bench)",
           "(d) Bench: missing-answer frac (real GAIA inference)")
plt.suptitle("Figure 3 · Truncation & answer-format failures: where reward signal goes blind", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig3_truncation.png"), dpi=130, bbox_inches="tight")
plt.close()


# ===== Figure 4: Thinking pattern & language =====
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
plot_panel(axes[0][0], "roll_num_think_mean", "<think> blocks / traj",
           "(a) Number of <think> blocks per trajectory")
plot_panel(axes[0][1], "roll_think_chars_mean", "thinking chars / traj",
           "(b) Total characters inside <think>")
plot_panel(axes[1][0], "roll_full_chinese_frac_mean", "Chinese-character frac",
           "(c) Chinese-character fraction in full output")
plot_panel(axes[1][1], "roll_hedge_count_mean", "hedge tokens / traj",
           "(d) Self-reflection / hedging tokens (wait, actually, etc.)")
plt.suptitle("Figure 4 · Reasoning style: thinking, language, self-reflection", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig4_thinking.png"), dpi=130, bbox_inches="tight")
plt.close()


# ===== Figure 5: Sim-to-Real gap =====
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
# (a) sim vs real accuracy on same step
for st, sty in [("stage1", "-"), ("stage2", "-")]:
    rows = D[st]
    xs = [r["step"] for r in rows if r.get("roll_correct_frac") is not None and r.get("bench_correct_frac") is not None]
    sim = [r["roll_correct_frac"] for r in rows if r.get("roll_correct_frac") is not None and r.get("bench_correct_frac") is not None]
    real = [r["bench_correct_frac"] for r in rows if r.get("roll_correct_frac") is not None and r.get("bench_correct_frac") is not None]
    axes[0].plot(xs, sim, sty + "o", color=COLORS[st], label=f"{st.upper()} sim",  alpha=0.9)
    axes[0].plot(xs, real, sty + "s", color=COLORS[st], label=f"{st.upper()} real", alpha=0.5)
axes[0].axvline(220, color="gray", ls="--", lw=0.8, alpha=0.5)
axes[0].axvspan(*COLLAPSE_BAND, color="orange", alpha=0.10)
axes[0].set_xlabel("step"); axes[0].set_ylabel("accuracy")
axes[0].set_title("(a) Sim (rollout) vs Real (GAIA pass@1) accuracy")
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

# (b) sim-real gap
for st in ["stage1", "stage2"]:
    rows = D[st]
    xs, gap = [], []
    for r in rows:
        if r.get("roll_correct_frac") is not None and r.get("bench_correct_frac") is not None:
            xs.append(r["step"]); gap.append(r["roll_correct_frac"] - r["bench_correct_frac"])
    axes[1].plot(xs, gap, "o-", color=COLORS[st], label=st.upper(), linewidth=1.8)
axes[1].axhline(0, color="black", lw=0.6)
axes[1].axvline(220, color="gray", ls="--", lw=0.8, alpha=0.5)
axes[1].axvspan(*COLLAPSE_BAND, color="orange", alpha=0.10)
axes[1].set_xlabel("step"); axes[1].set_ylabel("sim − real (acc gap)")
axes[1].set_title("(b) Sim-to-Real generalization gap (positive = sim flatters real)")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
plt.suptitle("Figure 5 · Sim-to-real consistency: when does training reward stop predicting real eval?", y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig5_sim2real.png"), dpi=130, bbox_inches="tight")
plt.close()


# ===== Figure 6: Bench pass@k summary =====
fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
for st in ["stage1", "stage2"]:
    rows = D[st]
    xs, p1, pk = [], [], []
    for r in rows:
        bs = r.get("bench_summary")
        if bs:
            xs.append(r["step"])
            p1.append(bs.get("avg_accuracy") or 0)
            pk.append(bs.get("pass_at_k_accuracy") or 0)
    ax.plot(xs, p1, "o-", color=COLORS[st], label=f"{st.upper()} pass@1", linewidth=1.8)
    ax.plot(xs, pk, "s--", color=COLORS[st], label=f"{st.upper()} pass@4", linewidth=1.5, alpha=0.7)
ax.axvline(220, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.axvspan(*COLLAPSE_BAND, color="orange", alpha=0.10)
ax.set_xlabel("step"); ax.set_ylabel("GAIA accuracy (%)")
ax.set_title("Figure 6 · GAIA pass@1 and pass@4 across stages")
ax.legend(fontsize=8); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "fig6_passk.png"), dpi=130, bbox_inches="tight")
plt.close()

print("OK: 6 figures generated.")
