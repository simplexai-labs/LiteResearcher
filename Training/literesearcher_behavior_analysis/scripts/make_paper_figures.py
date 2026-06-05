#!/usr/bin/env python3
"""
Paper-ready figures: 4 behavior mechanisms + accuracy outcome.
Continuous training axis: S1 step 0..220 then S2 step 0..570 spliced at global step 220.
"""
import json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

HERE = Path(__file__).parent
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)

# ------------------- style -------------------
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
D = json.load(open(HERE / "data" / "behavior_timeline.json"))
STAGE_BOUNDARY = 220
S1_KEEP_LIMIT  = 220
XLIM = (0, 800)

C_TRAIN = "#2E5C8A"   # blue
C_VAL   = "#B0413E"   # brick
C_K     = "#5B8C5A"   # green for pass@4
C_GREY  = "#666666"

def collect(metric_key, source="roll"):
    xs, ys = [], []
    for r in D["stage1"]:
        if r["step"] > S1_KEEP_LIMIT: continue
        v = r.get(f"{source}_{metric_key}")
        if v is None: continue
        xs.append(r["step"]); ys.append(v)
    for r in D["stage2"]:
        v = r.get(f"{source}_{metric_key}")
        if v is None: continue
        xs.append(r["step"] + STAGE_BOUNDARY); ys.append(v)
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]

def collect_derived(num_key, den_key, source="roll"):
    """num / den, point-wise (e.g. chars_per_think = think_chars / num_think)."""
    xs, ys = [], []
    for stage_label, offset in [("stage1", 0), ("stage2", STAGE_BOUNDARY)]:
        for r in D[stage_label]:
            if stage_label == "stage1" and r["step"] > S1_KEEP_LIMIT: continue
            n = r.get(f"{source}_{num_key}"); d = r.get(f"{source}_{den_key}")
            if n is None or d is None or d == 0: continue
            xs.append(r["step"] + offset); ys.append(n / d)
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]

def collect_bench_summary(field):
    xs, ys = [], []
    for stage_label, offset in [("stage1", 0), ("stage2", STAGE_BOUNDARY)]:
        for r in D[stage_label]:
            if stage_label == "stage1" and r["step"] > S1_KEEP_LIMIT: continue
            bs = r.get("bench_summary")
            if not bs or bs.get(field) is None: continue
            xs.append(r["step"] + offset); ys.append(bs[field])
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]

def add_splice(ax, label="S2 starts", color="black"):
    ax.axvline(STAGE_BOUNDARY, color=color, lw=0.8, ls=":", alpha=0.6)
    y_lo, y_hi = ax.get_ylim()
    ax.text(STAGE_BOUNDARY + 6, y_lo + 0.04*(y_hi-y_lo),
            label, fontsize=8, color=color, alpha=0.6, ha="left", va="bottom")

def shade_stages(ax):
    y_lo, y_hi = ax.get_ylim()
    ax.axvspan(0, STAGE_BOUNDARY, alpha=0.04, color="steelblue")
    ax.axvspan(STAGE_BOUNDARY, XLIM[1], alpha=0.04, color="firebrick")
    ax.set_ylim(y_lo, y_hi)

def style_panel(ax, ylabel, title, ylim=None):
    ax.set_xlabel("global training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlim(*XLIM)
    if ylim: ax.set_ylim(*ylim)

def plot_train_val(ax, metric_key, source_val="bench", train_only=False, scale=1.0):
    xt, yt = collect(metric_key, "roll")
    ax.plot(xt, yt * scale, "o-", color=C_TRAIN, lw=1.6, ms=3.6,
            label="train rollout", markeredgewidth=0)
    if not train_only:
        xv, yv = collect(metric_key, source_val)
        if len(xv):
            ax.plot(xv, yv * scale, "s--", color=C_VAL, lw=1.3, ms=4.0,
                    label="GAIA eval", markeredgewidth=0, alpha=0.95)
    ax.legend(loc="best")

# =========================================================
# FIGURE 1  — Accuracy outcome (sim2real headline)
# =========================================================
fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
xt, yt = collect("correct_frac", "roll")
ax.plot(xt, yt*100, "o-", color=C_TRAIN, lw=1.8, ms=4, label="Train rollout reward")
xv, yv = collect("correct_frac", "bench")
ax.plot(xv, yv*100, "s--", color=C_VAL, lw=1.5, ms=4.5, label="GAIA pass@1")
xp, yp = collect_bench_summary("pass_at_k_accuracy")
ax.plot(xp, yp, "^-.", color=C_K, lw=1.5, ms=4.5, label="GAIA pass@4")
ax.axvline(STAGE_BOUNDARY, color="black", lw=0.9, ls=":", alpha=0.65)
ax.text(STAGE_BOUNDARY+6, 30, "S2 starts (ckpt-220 splice)",
        fontsize=9.5, color="black", alpha=0.7, ha="left")
ax.set_xlim(*XLIM); ax.set_ylim(25, 95)
ax.set_xlabel("global training step")
ax.set_ylabel("accuracy (%)")
ax.set_title("Sim2Real transfer along the continuous training path",
             loc="left", fontweight="bold")
ax.legend(loc="lower right")
plt.savefig(FIGDIR/"fig1_accuracy_sim2real.png", dpi=160)
plt.savefig(FIGDIR/"fig1_accuracy_sim2real.pdf")
plt.close()
print("Saved fig1_accuracy_sim2real")

# =========================================================
# FIGURE 2  — Four behavioral mechanisms (2x2)
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
fig.subplots_adjust(hspace=0.55, wspace=0.27, top=0.90, bottom=0.10,
                    left=0.08, right=0.985)

# (a) mechanism 1 — browse-over-search tool rebalance
ax = axes[0][0]
plot_train_val(ax, "browse_ratio_mean")
style_panel(ax, "visit / (search + visit)",
            "(a) Mechanism 1 · Tool-choice shift: search → browse",
            ylim=(0.40, 0.85))
add_splice(ax); shade_stages(ax)

# (b) mechanism 2 — trajectory length (U-shape: S1 compresses, S2 expands)
ax = axes[0][1]
plot_train_val(ax, "total_turns_mean")
style_panel(ax, "agent turns / trajectory",
            "(b) Mechanism 2 · Trajectory re-allocation (compress → expand)",
            ylim=(15, 50))
add_splice(ax); shade_stages(ax)

# (c) mechanism 3 — chars per <think> (depth per single reasoning episode)
ax = axes[1][0]
xt, yt = collect_derived("think_chars_mean", "num_think_mean", "roll")
ax.plot(xt, yt, "o-", color=C_TRAIN, lw=1.6, ms=3.6,
        label="train rollout", markeredgewidth=0)
style_panel(ax, "chars / <think> block",
            "(c) Mechanism 3 · Per-think reasoning depth grows",
            ylim=(300, 700))
ax.legend(loc="upper left")
add_splice(ax); shade_stages(ax)

# (d) mechanism 4 — self-correction volume (hedge tokens per trajectory)
ax = axes[1][1]
plot_train_val(ax, "hedge_count_mean", train_only=True)
style_panel(ax, "hedge tokens / trajectory",
            "(d) Mechanism 4 · Self-correction volume grows",
            ylim=(0, 7))
add_splice(ax); shade_stages(ax)

fig.suptitle("Four behavioral mechanisms underlying the Sim2Real gain  "
             "(S1 0–220, S2 0–570 spliced at step 220)",
             y=0.975, fontsize=13, fontweight="bold")
fig.text(0.5, 0.005,
         "Blue = train rollout (300 traj/step). Red = GAIA eval (412 tasks). "
         "Panel (d) is train-only because GAIA logs strip <think> blocks.",
         ha="center", va="bottom", fontsize=9, color="#444444")
plt.savefig(FIGDIR/"fig2_mechanisms.png", dpi=160)
plt.savefig(FIGDIR/"fig2_mechanisms.pdf")
plt.close()
print("Saved fig2_mechanisms")

# =========================================================
# FIGURE 3 — Per-turn normalized rates (supports mech 1)
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
fig.subplots_adjust(wspace=0.30, top=0.85, bottom=0.18, left=0.06, right=0.985)

# visits per turn
ax = axes[0]
xt, yt = collect_derived("num_visit_mean", "total_turns_mean", "roll")
xv, yv = collect_derived("num_visit_mean", "total_turns_mean", "bench")
ax.plot(xt, yt, "o-", color=C_TRAIN, lw=1.6, ms=3.6, label="train rollout", markeredgewidth=0)
if len(xv):
    ax.plot(xv, yv, "s--", color=C_VAL, lw=1.3, ms=4, label="GAIA eval", markeredgewidth=0)
style_panel(ax, "visits / turn",
            "(a) Per-turn browse rate", ylim=(0.15, 0.40))
ax.legend(loc="lower right")
add_splice(ax); shade_stages(ax)

# searches per turn
ax = axes[1]
xt, yt = collect_derived("num_search_mean", "total_turns_mean", "roll")
xv, yv = collect_derived("num_search_mean", "total_turns_mean", "bench")
ax.plot(xt, yt, "o-", color=C_TRAIN, lw=1.6, ms=3.6, label="train rollout", markeredgewidth=0)
if len(xv):
    ax.plot(xv, yv, "s--", color=C_VAL, lw=1.3, ms=4, label="GAIA eval", markeredgewidth=0)
style_panel(ax, "searches / turn",
            "(b) Per-turn search rate", ylim=(0.05, 0.35))
ax.legend(loc="upper right")
add_splice(ax); shade_stages(ax)

# think density (fraction of output that is reasoning)
ax = axes[2]
plot_train_val(ax, "think_density_mean", train_only=True)
style_panel(ax, "think_chars / output_chars",
            "(c) Reasoning share of output", ylim=(0.05, 0.18))
add_splice(ax); shade_stages(ax)

fig.suptitle("Normalized per-turn rates: tool-choice and reasoning share both shift",
             y=0.97, fontsize=12.5, fontweight="bold")
plt.savefig(FIGDIR/"fig3_normalized_rates.png", dpi=160)
plt.savefig(FIGDIR/"fig3_normalized_rates.pdf")
plt.close()
print("Saved fig3_normalized_rates")

# =========================================================
# FIGURE 4 — Full 12-panel appendix figure
# =========================================================
N_ROWS, N_COLS = 4, 3
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(15.5, 16), sharex=False)
fig.subplots_adjust(hspace=0.55, wspace=0.30, top=0.945, bottom=0.05,
                    left=0.06, right=0.985)
panels = [
    ("correct_frac",        "accuracy",                "(a) Task accuracy", None),
    ("output_tokens_mean",  "tokens",                  "(b) Mean response length", None),
    ("total_turns_mean",    "turns",                   "(c) # agent turns", None),
    ("num_search_mean",     "calls / trajectory",      "(d) Search-tool calls", None),
    ("num_visit_mean",      "calls / trajectory",      "(e) Browse / visit calls", None),
    ("browse_ratio_mean",   "visit / (search+visit)",  "(f) Browse share of tool use", None),
    ("queries_per_search_call_mean", "queries / call", "(g) Queries per search call", None),
    ("num_queries_mean",    "queries / trajectory",    "(h) Total search queries", None),
    ("quoted_query_frac_mean", "fraction",             "(i) Quoted exact-phrase queries", None),
    ("num_think_mean",      "blocks / trajectory",     "(j) # <think> blocks", "train_only"),
    ("think_density_mean",  "fraction",                "(k) Thinking share of output", "train_only"),
    ("hedge_count_mean",    "tokens / trajectory",     "(l) Self-correction hedges", "train_only"),
]
for ax, (key, ylab, title, mode) in zip(axes.flat, panels):
    train_only = (mode == "train_only")
    plot_train_val(ax, key, train_only=train_only)
    style_panel(ax, ylab, title)
    add_splice(ax); shade_stages(ax)
fig.suptitle("Full behavioral timeline across the continuous training path "
             "(S1 step 0–220, S2 step 0–570 spliced at step 220)",
             y=0.985, fontsize=14, fontweight="bold")
fig.text(0.5, 0.005,
         "Blue = train rollouts (300 traj/step) ; Red = GAIA eval (412 tasks/step) ; "
         "vertical dotted line = ckpt-220 splice. (j)(k)(l) train-only because GAIA logs strip <think> blocks.",
         ha="center", va="bottom", fontsize=9.5, color="#333333")
plt.savefig(FIGDIR/"fig4_appendix_12panel.png", dpi=160)
plt.savefig(FIGDIR/"fig4_appendix_12panel.pdf")
plt.close()
print("Saved fig4_appendix_12panel")
