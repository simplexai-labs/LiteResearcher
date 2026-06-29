#!/usr/bin/env python3
"""Generate a Tongyi-style benchmark results figure for the README."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---- data: 8 benchmarks x 18 models (dash = not reported) ----
COLS = ["GAIA", "BrowseComp", "BrowseComp-ZH", "HLE", "Frames", "WebWalker", "Seal-0", "Xbench-DS"]

GROUPS = [
    ("Commercial Models", [
        ("Claude-4-Sonnet",      ["68.3", "12.2", "29.1", "20.3", "80.7", "61.7", "–",    "64.6"]),
        ("Claude-4.5-Sonnet",    ["71.2", "19.6", "40.8", "24.5", "85.0", "–",    "53.4", "66.0"]),
        ("DeepSeek-V3.2",        ["63.5", "67.6", "65.0", "40.8", "80.2", "–",    "38.5", "71.0"]),
        ("DeepSeek-V3.1",        ["63.1", "30.0", "49.2", "29.8", "83.7", "61.2", "–",    "71.0"]),
        ("Minimax-M2",           ["75.7", "44.0", "48.5", "31.8", "–",    "–",    "–",    "72.0"]),
        ("OpenAI-GPT-5-high",    ["76.4", "54.9", "65.0", "35.2", "–",    "–",    "51.4", "77.8"]),
        ("GLM-4.6",              ["71.9", "45.1", "49.5", "30.4", "–",    "–",    "–",    "70.0"]),
        ("Kimi-Researcher",      ["–",    "–",    "–",    "26.9", "78.8", "–",    "36.0", "69.0"]),
        ("Kimi-K2-0905",         ["60.2", "7.4",  "22.2", "21.7", "58.1", "–",    "25.2", "61.0"]),
    ]),
    ("Open-Source Models (>8B)", [
        ("Tongyi DeepResearch-30B", ["70.9", "43.4", "46.7", "32.9", "90.6", "72.2", "–", "75.0"]),
        ("ASearcher-QWQ-v2",     ["58.7", "–",    "–",    "–",    "74.5", "–",    "–",    "51.1"]),
        ("WebSailor-30B",        ["53.2", "–",    "–",    "–",    "–",    "–",    "–",    "53.3"]),
        ("WebDancer-QwQ",        ["51.5", "3.8",  "18.0", "–",    "–",    "47.9", "–",    "38.3"]),
        ("DeepMiner-32B",        ["58.7", "33.5", "40.1", "–",    "–",    "–",    "–",    "62.0"]),
        ("AFM-RL-32B",           ["55.3", "11.1", "–",    "18.0", "–",    "63.0", "–",    "–"]),
        ("SFR-DeepResearch",     ["66.0", "–",    "–",    "28.7", "82.8", "–",    "–",    "–"]),
    ]),
    ("Small Models (≤8B)", [
        ("Mirothinker-8B",       ["66.4", "31.1", "40.2", "21.5", "80.6", "60.6", "40.4", "60.6"]),
        ("WebExplorer-8B",       ["50.0", "15.7", "32.0", "17.3", "75.7", "62.7", "–",    "53.7"]),
        ("AgentCPM-Explore-4B",  ["63.9", "24.1", "29.1", "19.1", "82.7", "68.1", "40.5", "70.0"]),
        ("LiteResearcher-4B",    ["71.3", "27.5*", "32.5*", "22.0", "83.1", "72.7", "41.8", "78.0"]),
    ]),
]

# benchmarks where LiteResearcher is best among <=8B (bold/highlight in its row)
LITE_BEST = {"GAIA", "HLE", "Frames", "WebWalker", "Seal-0", "Xbench-DS"}

# ---- palette (navy blue, matching the Method Overview figure) ----
PURPLE      = "#3d5180"   # primary accent (navy, matches overview.png)
PURPLE_DARK = "#2c3c63"   # darker navy for emphasis
HEADER_BG   = "#ffffff"
GROUP_BG    = "#e9edf5"   # light navy band for group separators
ROW_ALT     = "#f5f7fb"   # very light navy for alternating rows
HILITE_BG   = "#dde4f0"   # LiteResearcher row highlight
TEXT        = "#2a3142"
MUTED       = "#9aa1b3"
BORDER      = "#d6deec"

MONO = "DejaVu Sans Mono"
SANS = "DejaVu Sans"

n_data_rows = sum(len(rows) for _, rows in GROUPS)
n_groups = len(GROUPS)
n_lines = 1 + n_groups + n_data_rows  # header + group headers + data rows

ROW_H = 0.40
HEADER_H = 0.60
TITLE_H = 1.5
W = 16.0
table_h = HEADER_H + (n_lines - 1) * ROW_H
H = TITLE_H + table_h + 0.5

fig = plt.figure(figsize=(W, H), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")

# ---- title: rounded "search bar" ----
bar_w, bar_h = 8.6, 0.78
bar_x = (W - bar_w) / 2
bar_y = H - 1.15
ax.add_patch(FancyBboxPatch((bar_x, bar_y), bar_w - 2.0, bar_h,
             boxstyle="round,pad=0.02,rounding_size=0.39",
             linewidth=2, edgecolor=PURPLE, facecolor="white", zorder=3))
ax.text(bar_x + 0.45, bar_y + bar_h / 2, "Deep Research Benchmarks",
        fontsize=17, fontweight="bold", color=PURPLE, va="center", ha="left",
        fontfamily=SANS, zorder=4)
btn_x = bar_x + bar_w - 2.0
ax.add_patch(FancyBboxPatch((btn_x - 0.05, bar_y), 2.05, bar_h,
             boxstyle="round,pad=0.02,rounding_size=0.39",
             linewidth=0, facecolor=PURPLE, zorder=3))
ax.text(btn_x + 0.97, bar_y + bar_h / 2, "Research It!",
        fontsize=14, fontweight="bold", color="white", va="center", ha="center",
        fontfamily=SANS, zorder=4)

# ---- table geometry ----
left = 0.5
right = W - 0.5
model_col_w = 3.5
val_x0 = left + model_col_w
val_w = (right - val_x0) / len(COLS)
top = bar_y - 0.4

# outer rounded card
ax.add_patch(FancyBboxPatch((left, top - table_h), right - left, table_h,
             boxstyle="round,pad=0.0,rounding_size=0.12",
             linewidth=1.4, edgecolor=BORDER, facecolor="white", zorder=1))

def col_center(i):
    return val_x0 + val_w * (i + 0.5)

y = top
# header row
ax.add_patch(Rectangle((left, y - HEADER_H), right - left, HEADER_H,
             facecolor=HEADER_BG, edgecolor="none", zorder=2))
ax.text(left + 0.35, y - HEADER_H / 2, "Benchmarks", fontsize=14,
        fontweight="bold", color=TEXT, va="center", ha="left", fontfamily=SANS, zorder=5)
for i, c in enumerate(COLS):
    ax.text(col_center(i), y - HEADER_H / 2, c, fontsize=12, fontweight="bold",
            color=TEXT, va="center", ha="center", fontfamily=SANS, zorder=5)
ax.plot([left, right], [y - HEADER_H, y - HEADER_H], color=PURPLE, lw=1.6, zorder=5)
y -= HEADER_H

alt = False
for gname, rows in GROUPS:
    # group separator row
    ax.add_patch(Rectangle((left, y - ROW_H), right - left, ROW_H,
                 facecolor=GROUP_BG, edgecolor="none", zorder=2))
    ax.text((left + right) / 2, y - ROW_H / 2, gname, fontsize=12.5,
            fontweight="bold", color=PURPLE_DARK, va="center", ha="center",
            fontfamily=MONO, zorder=5)
    y -= ROW_H
    for mname, vals in rows:
        is_lite = mname == "LiteResearcher-4B"
        if is_lite:
            bg = HILITE_BG
        else:
            bg = ROW_ALT if alt else "white"
        ax.add_patch(Rectangle((left, y - ROW_H), right - left, ROW_H,
                     facecolor=bg, edgecolor="none", zorder=2))
        mcolor = PURPLE_DARK if is_lite else TEXT
        mweight = "bold" if is_lite else "normal"
        ax.text(left + 0.35, y - ROW_H / 2, mname, fontsize=12,
                fontweight=mweight, color=mcolor, va="center", ha="left",
                fontfamily=SANS, zorder=5)
        for i, v in enumerate(vals):
            if v == "–":
                col = MUTED
                wt = "normal"
            elif is_lite:
                col = PURPLE_DARK
                wt = "bold" if COLS[i] in LITE_BEST else "normal"
            else:
                col = TEXT
                wt = "normal"
            ax.text(col_center(i), y - ROW_H / 2, v, fontsize=12, fontweight=wt,
                    color=col, va="center", ha="center", fontfamily=MONO, zorder=5)
        y -= ROW_H
        alt = not alt

# footnote
ax.text(left + 0.1, y - 0.05, "Accuracy (%). – = not reported. * = 64k ctx + summarization memory (others 128k). "
        "Bold = best among ≤8B models.",
        fontsize=8.5, color=MUTED, va="top", ha="left", fontfamily=SANS)

out = "figures/benchmark_results.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
