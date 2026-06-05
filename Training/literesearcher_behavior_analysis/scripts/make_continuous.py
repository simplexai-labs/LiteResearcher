#!/usr/bin/env python3
"""
Continuous training-curve figures (S1 0..220 + S2 0..570 spliced at global step 220).
Academic publication style: serif font, minor grids, smoothed curves, ckpt220 separator.
"""
import json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ------------------- academic style -------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Bitstream Vera Serif"],
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
})

# ------------------- data -------------------
D = json.load(open(os.path.join(os.path.dirname(__file__), "behavior_timeline.json")))
STAGE_BOUNDARY = 220     # global step where S2 begins
S1_KEEP_LIMIT  = 220     # we drop S1 rows past 220 (abandoned branch)

def collect(metric, source="roll"):
    """Return (global_steps, values, stage_id) along the continuous training axis."""
    xs, ys, sid = [], [], []
    for r in D["stage1"]:
        if r["step"] > S1_KEEP_LIMIT: continue
        v = r.get(f"{source}_{metric}")
        if v is None: continue
        xs.append(r["step"]); ys.append(v); sid.append(1)
    for r in D["stage2"]:
        v = r.get(f"{source}_{metric}")
        if v is None: continue
        xs.append(r["step"] + STAGE_BOUNDARY); ys.append(v); sid.append(2)
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order], np.array(sid)[order]

def collect_bench_summary(field):
    """summary.json metrics (pass@1 percentage, pass@4 percentage)."""
    xs, ys = [], []
    for r in D["stage1"]:
        if r["step"] > S1_KEEP_LIMIT: continue
        bs = r.get("bench_summary")
        if not bs or bs.get(field) is None: continue
        xs.append(r["step"]); ys.append(bs[field])
    for r in D["stage2"]:
        bs = r.get("bench_summary")
        if not bs or bs.get(field) is None: continue
        xs.append(r["step"] + STAGE_BOUNDARY); ys.append(bs[field])
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]

C_TRAIN = "#2E5C8A"  # blue-grey
C_VAL   = "#B0413E"  # brick
C_S1    = "#777777"
C_S2    = "#444444"

def plot_panel(ax, metric_train, metric_val=None, ylabel="", title="",
               source_train="roll", source_val="bench",
               y_scale_train=1.0, y_scale_val=1.0, ylim=None,
               train_only=False, annotate_splice=True):
    xt, yt, _ = collect(metric_train, source_train)
    ax.plot(xt, yt * y_scale_train, "o-", color=C_TRAIN, lw=1.5, ms=3.5,
            label="train (rollout)", markeredgewidth=0)
    if metric_val and not train_only:
        xv, yv, _ = collect(metric_val, source_val)
        if len(xv):
            ax.plot(xv, yv * y_scale_val, "s--", color=C_VAL, lw=1.3, ms=3.8,
                    label="eval (GAIA)", markeredgewidth=0, alpha=0.95)
    ax.axvline(STAGE_BOUNDARY, color="black", lw=0.7, ls=":", alpha=0.6)
    if annotate_splice:
        y_lo, y_hi = ax.get_ylim()
        ax.text(STAGE_BOUNDARY + 6, y_lo + 0.04 * (y_hi - y_lo),
                "S2 starts", fontsize=8, color="black", alpha=0.55, ha="left", va="bottom")
    ax.set_xlabel("global training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlim(0, 800)
    ax.legend(loc="best")

# ------------------- big multi-panel academic figure -------------------
N_ROWS, N_COLS = 4, 3
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(15.5, 16),
                         sharex=False)
fig.subplots_adjust(hspace=0.55, wspace=0.30, top=0.945, bottom=0.05,
                    left=0.06, right=0.985)

panels = [
    # (train_key, val_key, ylabel, title)
    ("correct_frac",        "correct_frac",        "accuracy",
        "(a) Task-completion accuracy"),
    ("output_tokens_mean",  "output_tokens_mean",  "tokens",
        "(b) Mean response length"),
    ("total_turns_mean",    "total_turns_mean",    "turns",
        "(c) Mean # of agent turns"),
    ("num_search_mean",     "num_search_mean",     "calls / trajectory",
        "(d) Search-tool calls"),
    ("num_visit_mean",      "num_visit_mean",      "calls / trajectory",
        "(e) Browse / visit calls"),
    ("browse_ratio_mean",   "browse_ratio_mean",   "visit / (search+visit)",
        "(f) Browse share of tool usage"),
    ("queries_per_search_call_mean", "queries_per_search_call_mean",
        "queries / call", "(g) Search concurrency (queries per call)"),
    ("num_queries_mean",    "num_queries_mean",    "queries / trajectory",
        "(h) Total search queries emitted"),
    ("quoted_query_frac_mean", "quoted_query_frac_mean", "fraction",
        "(i) Quoted exact-phrase queries"),
    ("num_think_mean",      "num_think_mean",      "blocks / trajectory",
        "(j) Number of <think> reasoning blocks"),
    ("think_density_mean",  "think_density_mean",  "think chars / total chars",
        "(k) Thinking density (share of output)"),
    ("hedge_count_mean",    "hedge_count_mean",    "tokens / trajectory",
        "(l) Self-reflection hedges (wait / actually / reconsider)"),
]
for ax, (tk, vk, yl, ti) in zip(axes.flat, panels):
    # think-related metrics are train-only (bench strips <think> content)
    train_only = tk in {"num_think_mean", "think_density_mean"}
    plot_panel(ax, tk, vk, ylabel=yl, title=ti, train_only=train_only)

fig.suptitle("Behavioral Evolution along the Continuous Training Path  (S1 step 0–220, then S2 step 0–570 spliced at step 220)",
             y=0.985, fontsize=14, fontweight="bold")
fig.text(0.5, 0.005,
         "blue = training rollouts (sim, 300 traj/step) | red = held-out GAIA eval (pass@1, 412 tasks/step) | dotted vertical line = ckpt-220 splice point | (j),(k) train-only: GAIA logs strip <think> blocks",
         ha="center", va="bottom", fontsize=9.5, color="#333333")

out_path = os.path.join(os.path.dirname(__file__), "fig_continuous_full.png")
plt.savefig(out_path, dpi=150)
plt.savefig(out_path.replace(".png", ".pdf"))
plt.close()
print(f"Saved {out_path}")

# ------------------- compact 2x2 "headline" figure -------------------
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7))
fig.subplots_adjust(hspace=0.5, wspace=0.28, top=0.92, bottom=0.10,
                    left=0.08, right=0.985)
plot_panel(ax[0][0], "correct_frac", "correct_frac",
           ylabel="accuracy", title="(a) Accuracy: rollout vs GAIA")
plot_panel(ax[0][1], "num_visit_mean", "num_visit_mean",
           ylabel="calls / trajectory", title="(b) Browse / visit calls grow")
plot_panel(ax[1][0], "queries_per_search_call_mean", "queries_per_search_call_mean",
           ylabel="queries / call", title="(c) Search concurrency grows")
plot_panel(ax[1][1], "num_think_mean", "num_think_mean",
           ylabel="<think> blocks / traj", title="(d) Reasoning depth grows", train_only=True)
fig.suptitle("Headline behavioral shifts as training progresses through ckpt-220 splice",
             y=0.985, fontsize=12.5, fontweight="bold")
out2 = os.path.join(os.path.dirname(__file__), "fig_continuous_headline.png")
plt.savefig(out2, dpi=150)
plt.savefig(out2.replace(".png", ".pdf"))
plt.close()
print(f"Saved {out2}")

# ------------------- pass@1 & pass@4 vs train acc -------------------
fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
xt, yt, _ = collect("correct_frac", "roll")
ax.plot(xt, yt * 100, "o-", color=C_TRAIN, lw=1.6, ms=3.8, label="train rollout acc")
xv, yv, _ = collect("correct_frac", "bench")
ax.plot(xv, yv * 100, "s--", color=C_VAL, lw=1.4, ms=4.0, label="GAIA pass@1")
xp, yp = collect_bench_summary("pass_at_k_accuracy")
ax.plot(xp, yp, "^-.", color="#5B8C5A", lw=1.4, ms=4.2, label="GAIA pass@4")
ax.axvline(STAGE_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.6)
ax.text(STAGE_BOUNDARY + 4, 30, "S2 starts here", fontsize=9, color="black", alpha=0.7)
ax.set_xlabel("global training step")
ax.set_ylabel("accuracy (%)")
ax.set_title("Train reward and GAIA evaluation across the continuous training path")
ax.set_xlim(0, 800); ax.set_ylim(25, 95)
ax.legend(loc="lower right")
out3 = os.path.join(os.path.dirname(__file__), "fig_continuous_acc.png")
plt.savefig(out3, dpi=150)
plt.savefig(out3.replace(".png", ".pdf"))
plt.close()
print(f"Saved {out3}")
