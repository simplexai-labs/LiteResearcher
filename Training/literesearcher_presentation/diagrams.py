"""Generate custom decorative figures for the deck."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

# Claude palette
CREAM     = "#F5F1ED"
INK       = "#1F1F1F"
INK_SOFT  = "#6B6356"
ACCENT    = "#D97757"
ACCENT_DK = "#B85F44"
DIVIDER   = "#E8E0D5"
INK_PANEL = "#EBE4D7"
GREEN     = "#5B8C5A"
NAVY      = "#2E5C8A"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.8,
    "savefig.facecolor": CREAM,
    "figure.facecolor": CREAM,
})

# ============================================================
# 1. Architecture diagram — 3 pillars
# ============================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.5); ax.axis("off")

    title_props = dict(fontsize=12.5, color=INK, weight="bold", ha="center")
    body_props  = dict(fontsize=9.5, color=INK_SOFT, ha="center", va="top")

    pillars = [
        ("①  Co-construct\nData × Corpus",
         "Seed corpus  →  QA gen\n→  source masking\n→  rubric filter\n→  corpus expansion",
         0.5),
        ("②  Stable Local\nTool Environment",
         "32M pages · 1M+ domains\nLocal Search Engine (BGE-M3)\nLocal Browse (Postgres)\n~0.15s search · ~0.17s browse",
         4.4),
        ("③  Difficulty-Aware\nCurriculum Learning",
         "pass@8 ∈ [1, 7]  filter\nOn-policy GRPO\nMulti-stage difficulty + ctx\nStage1 32K  →  Stage2 48K",
         8.3),
    ]
    for title, body, x in pillars:
        rect = patches.FancyBboxPatch((x, 0.8), 3.2, 3.6,
                                      boxstyle="round,pad=0.02,rounding_size=0.15",
                                      linewidth=1.2, edgecolor=DIVIDER,
                                      facecolor="white")
        ax.add_patch(rect)
        ax.text(x + 1.6, 4.0, title, **title_props)
        ax.text(x + 1.6, 3.2, body, **body_props)

    # arrows
    for x in [3.7, 7.6]:
        ax.annotate("", xy=(x + 0.55, 2.6), xytext=(x, 2.6),
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))

    # output banner
    rect = patches.FancyBboxPatch((2, 0.05), 8, 0.55,
                                  boxstyle="round,pad=0.02,rounding_size=0.12",
                                  linewidth=0, facecolor=ACCENT)
    ax.add_patch(rect)
    ax.text(6, 0.33, "LiteResearcher-4B   ·   GAIA 71.3%   ·   Xbench-DS 78.0%",
            fontsize=12, color="white", weight="bold", ha="center", va="center")

    plt.savefig(HERE / "diag_architecture.png", dpi=200, bbox_inches="tight")
    plt.close()

# ============================================================
# 2. Five atomic capabilities — cards
# ============================================================
def fig_atomic_caps():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.5); ax.axis("off")
    caps = [
        ("Direct Information",
         "Single fact, single retrieval.\ne.g. \"When was X founded?\""),
        ("Aggregation",
         "Find target by multiple\nidentifying attributes."),
        ("Enumeration",
         "List & count entities that\nsatisfy a constraint."),
        ("Cross-verification",
         "Validate claims across\nmultiple independent sources."),
        ("Statistics",
         "Aggregate numerical metrics\nfrom multiple pages."),
    ]
    # 5 cards in a row
    card_w, card_h, gap = 2.15, 3.0, 0.15
    total = 5 * card_w + 4 * gap
    x0 = (12 - total) / 2
    for i, (t, b) in enumerate(caps):
        x = x0 + i * (card_w + gap)
        rect = patches.FancyBboxPatch((x, 1.2), card_w, card_h,
                                      boxstyle="round,pad=0.02,rounding_size=0.12",
                                      linewidth=1, edgecolor=DIVIDER,
                                      facecolor="white")
        ax.add_patch(rect)
        # accent number
        ax.text(x + card_w/2, 3.9, f"{i+1:02d}",
                fontsize=24, color=ACCENT, weight="bold", ha="center", va="center",
                family="Georgia")
        ax.text(x + card_w/2, 3.05, t,
                fontsize=10.5, color=INK, weight="bold", ha="center", va="center")
        ax.text(x + card_w/2, 2.0, b,
                fontsize=8.5, color=INK_SOFT, ha="center", va="center")

    plt.savefig(HERE / "diag_atomic_caps.png", dpi=200, bbox_inches="tight")
    plt.close()

# ============================================================
# 3. Cost comparison bar
# ============================================================
def fig_cost():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = ["Local\n(LiteResearcher)", "Serper API\n(commercial search)",
              "SerpAPI + Jina\n(commercial)", "Full proxy stack"]
    costs  = [0, 59, 168, 243]  # K USD for 73.2M tool calls
    colors = [ACCENT] + [INK_SOFT]*3
    bars = ax.barh(labels, costs, color=colors, edgecolor="none")
    for b, c in zip(bars, costs):
        ax.text(b.get_width() + 4, b.get_y() + b.get_height()/2,
                f"${c}K" if c else "$0   (zero marginal)",
                va="center", ha="left", fontsize=11, color=INK)
    ax.set_xlim(0, 285)
    ax.set_xlabel("cost in USD (×1000) for 73.2M tool calls", color=INK_SOFT, fontsize=10)
    ax.tick_params(axis="y", labelsize=10.5, colors=INK)
    ax.tick_params(axis="x", labelsize=9.5, colors=INK_SOFT)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(DIVIDER)
    ax.grid(axis="x", color=DIVIDER, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    plt.savefig(HERE / "diag_cost.png", dpi=200, bbox_inches="tight")
    plt.close()

# ============================================================
# 4. Causal funnel for the 4 mechanisms
# ============================================================
def fig_causal():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.5); ax.axis("off")

    nodes = [
        (1.3, 3.7, "M1\nTool-choice", "search → browse", ACCENT),
        (1.3, 1.7, "M2\nTraj. re-alloc", "compress → expand", ACCENT),
        (4.7, 2.7, "more evidence\nper trajectory", "4.6 → 16.4 visits", INK_PANEL),
        (7.5, 2.7, "M3 + M4\ndeeper reasoning", "+58% chars/think\n+3.4× hedges", INK_PANEL),
        (10.0, 2.7, "GAIA pass@1\n+9 pts", "0.55 → 0.68", ACCENT),
    ]
    for x, y, t, sub, fill in nodes:
        w, h = 1.85, 1.5
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                      boxstyle="round,pad=0.02,rounding_size=0.12",
                                      linewidth=1.0, edgecolor=DIVIDER,
                                      facecolor=fill)
        ax.add_patch(rect)
        text_color = "white" if fill == ACCENT else INK
        sub_color  = "white" if fill == ACCENT else INK_SOFT
        ax.text(x, y + 0.25, t, fontsize=10.5, color=text_color, weight="bold",
                ha="center", va="center")
        ax.text(x, y - 0.35, sub, fontsize=8.5, color=sub_color,
                ha="center", va="center")
    # arrows
    arrows = [(2.25, 3.5, 3.85, 2.85), (2.25, 1.9, 3.85, 2.55),
              (5.65, 2.7, 6.65, 2.7), (8.45, 2.7, 9.15, 2.7)]
    for x0, y0, x1, y1 in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=ACCENT_DK, lw=1.8))

    plt.savefig(HERE / "diag_causal.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fig_architecture()
    fig_atomic_caps()
    fig_cost()
    fig_causal()
    print("All decorative figures generated.")
