"""
LiteResearcher · Claude-style presentation builder.
Outputs LiteResearcher_Presentation.pptx (30 slides, 16:9).
"""
from pathlib import Path
from theme import *

HERE = Path(__file__).parent
TOTAL = 30  # will update if changes
OUT = HERE / "LiteResearcher_Presentation.pptx"

prs = make_prs()

def img(slide, fn, x, y, w=None, h=None):
    p = HERE / fn
    return slide.shapes.add_picture(str(p), x, y, width=w, height=h)

# =====================================================================
# Slide 1 — Title
# =====================================================================
def slide_01_title(idx):
    s = blank_slide(prs)
    # accent vertical bar on the left
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(2.4),
                             Inches(0.08), Inches(2.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    # tag
    add_text(s, Inches(0.9), Inches(2.35), Inches(8), Inches(0.35),
             "AGENTIC RL · DEEP RESEARCH", font=F_BODY, size=11,
             color=ACCENT, bold=True)
    # title
    add_text(s, Inches(0.9), Inches(2.75), Inches(11.5), Inches(1.4),
             "LiteResearcher", font=F_HEAD, size=64, color=INK, bold=True)
    # subtitle
    add_text(s, Inches(0.9), Inches(3.95), Inches(11.5), Inches(0.9),
             "A Scalable Agentic RL Training Framework\nfor Deep Research Agents",
             font=F_HEAD, size=24, color=INK_SOFT, italic=True,
             line_spacing=1.3)
    # footer
    add_hline(s, Inches(0.6), Inches(6.2), Inches(12.1))
    add_text(s, Inches(0.6), Inches(6.35), Inches(8), Inches(0.3),
             "arXiv:2604.17931  ·  github.com/simplex-ai-inc/LiteResearcher",
             font=F_BODY, size=11, color=INK_SOFT)
    add_text(s, Inches(0.6), Inches(6.7), Inches(8), Inches(0.3),
             "Zhejiang University  ·  Simplex AI  ·  HKPU",
             font=F_BODY, size=10.5, color=WARM_GRAY)
    add_text(s, Inches(11.5), Inches(6.7), Inches(1.3), Inches(0.3),
             f"{idx} / {TOTAL}", font=F_BODY, size=9, color=INK_SOFT,
             align=PP_ALIGN.RIGHT)
    return s

# =====================================================================
# Slide 2 — Authors / context
# =====================================================================
def slide_02_authors(idx):
    s = blank_slide(prs)
    page_header(s, "Paper Information", "Authors, links & one-line summary")
    # box: authors
    box1 = add_rect(s, Inches(0.6), Inches(2.0), Inches(6.0), Inches(2.4),
                    fill=CREAM_SOFT, line=DIVIDER)
    add_text(s, Inches(0.85), Inches(2.15), Inches(5.5), Inches(0.35),
             "AUTHORS", font=F_BODY, size=10.5, color=ACCENT, bold=True)
    add_text(s, Inches(0.85), Inches(2.5), Inches(5.5), Inches(1.6),
             "Wanli Li¹², Bince Qu¹², Bo Pan¹, Jianyu Zhang¹,\nZheng Liu³, Pan Zhang², Wei Chen¹, Bo Zhang¹²",
             font=F_BODY, size=13, color=INK, line_spacing=1.5)
    add_text(s, Inches(0.85), Inches(3.7), Inches(5.5), Inches(0.7),
             "¹ Zhejiang University    ² Simplex AI    ³ The Hong Kong Polytechnic University",
             font=F_BODY, size=10, color=INK_SOFT)

    # box: links
    box2 = add_rect(s, Inches(6.8), Inches(2.0), Inches(5.95), Inches(2.4),
                    fill=CREAM_SOFT, line=DIVIDER)
    add_text(s, Inches(7.05), Inches(2.15), Inches(5.5), Inches(0.35),
             "RESOURCES", font=F_BODY, size=10.5, color=ACCENT, bold=True)
    add_paragraphs(s, Inches(7.05), Inches(2.55), Inches(5.5), Inches(1.8),
                   [("Paper  ·  arXiv:2604.17931", {"size":13, "color":INK, "bold":True}),
                    ("Code   ·  github.com/simplex-ai-inc/LiteResearcher",
                     {"size":12, "color":INK_SOFT}),
                    ("Weights  ·  huggingface.co/simplex-ai-inc/LiteResearcher-4B",
                     {"size":12, "color":INK_SOFT}),
                    ("", {"space_after":2}),
                    ("Venue  ·  Preprint, under review", {"size":11, "color":WARM_GRAY, "italic":True})],
                   line_spacing=1.5, space_after=8)

    # tldr
    box3 = add_rect(s, Inches(0.6), Inches(4.7), Inches(12.15), Inches(1.9),
                    fill=ACCENT, line=None)
    add_text(s, Inches(0.85), Inches(4.85), Inches(11.5), Inches(0.35),
             "TL;DR", font=F_BODY, size=10.5, color=RGBColor(0xFF,0xE9,0xDE), bold=True)
    add_text(s, Inches(0.85), Inches(5.20), Inches(11.6), Inches(1.5),
             "A virtual world that mirrors live web search but isolates its noise lets a tiny 4B model\n"
             "match or beat 8× larger open-source and commercial agents on deep-research benchmarks.",
             font=F_HEAD, size=18, color=RGBColor(0xFF,0xFF,0xFF), italic=True,
             line_spacing=1.4)
    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 3 — Agenda
# =====================================================================
def slide_03_agenda(idx):
    s = blank_slide(prs)
    page_header(s, "Agenda", "What we will cover today")
    items = [
        ("01", "Problem & prior approaches",      "Why scaling Agentic RL for deep research is hard"),
        ("02", "Method overview",                  "The 3-pillar LiteResearcher pipeline"),
        ("03", "Pillar 1 · Data × Corpus",         "Co-construct synthetic data and local corpus"),
        ("04", "Pillar 2 · Local tool environment","32M-page virtual web at <0.2 s latency"),
        ("05", "Pillar 3 · Curriculum RL",         "On-policy GRPO + difficulty filtering"),
        ("06", "Main results",                     "GAIA, Xbench, BrowseComp, HLE, …"),
        ("07", "Ablations",                        "Cost, data, on/off-policy, two-stage, RL vs SFT"),
        ("08", "Behavior analysis · appendix",     "Four mechanisms behind the Sim2Real gain"),
    ]
    # 2-column layout
    col_w = Inches(5.95); gap = Inches(0.25)
    x0 = Inches(0.6); y0 = Inches(2.1); row_h = Inches(1.1)
    for i, (n, t, d) in enumerate(items):
        col = i % 2
        row = i // 2
        x = x0 + (col_w + gap) * col
        y = y0 + row_h * row
        bx = add_rect(s, x, y, col_w, Inches(1.0), fill=CREAM_SOFT, line=DIVIDER)
        # number
        add_text(s, x + Inches(0.25), y + Inches(0.18), Inches(0.7), Inches(0.6),
                 n, font=F_HEAD, size=22, color=ACCENT, bold=True)
        # title
        add_text(s, x + Inches(1.05), y + Inches(0.16), col_w - Inches(1.2), Inches(0.4),
                 t, font=F_BODY, size=13, color=INK, bold=True)
        # subtitle
        add_text(s, x + Inches(1.05), y + Inches(0.52), col_w - Inches(1.2), Inches(0.4),
                 d, font=F_BODY, size=10.5, color=INK_SOFT)
    page_footer(s, idx, TOTAL)
    return s

# Build first 3 slides and verify
slide_01_title(1)
slide_02_authors(2)
slide_03_agenda(3)

# Save partial deck for verification
prs.save(OUT)
print(f"Saved partial deck with {len(prs.slides)} slides at {OUT}")

# =====================================================================
# Slide 4 — The Problem
# =====================================================================
def slide_04_problem(idx):
    s = blank_slide(prs)
    page_header(s, "01 · Problem", "Why scaling Agentic RL for Deep Research is hard")

    # Big quoted statement
    box = add_rect(s, Inches(0.6), Inches(2.05), Inches(12.15), Inches(1.6),
                   fill=CREAM_SOFT, line=DIVIDER)
    add_text(s, Inches(0.9), Inches(2.2), Inches(0.3), Inches(0.6),
             "“", font=F_HEAD, size=46, color=ACCENT, bold=True)
    add_text(s, Inches(1.35), Inches(2.35), Inches(11.2), Inches(1.2),
             "RL has delivered sustained gains for closed-world reasoners; "
             "Agentic RL for deep research has not. Why?",
             font=F_HEAD, size=20, color=INK, italic=True, line_spacing=1.3)

    # Two coupled challenges
    y0 = Inches(3.95)
    titles = ["Hand-crafted synthetic data", "Real-web RL dependency"]
    bodies = [
        "Over-engineered task templates that do not elicit\n"
        "the diverse atomic search skills used in the real world\n"
        "— cross-verification, enumeration, statistics, …",
        "Live-internet rollouts inject high variance and prohibitive\n"
        "cost into RL training, capping the number of stable\n"
        "updates and the scale of training data."
    ]
    tags = ["A · DATA BOTTLENECK", "B · ENVIRONMENT BOTTLENECK"]
    for i in range(2):
        x = Inches(0.6 + i * 6.275)
        bx = add_rect(s, x, y0, Inches(5.875), Inches(2.7),
                      fill=CREAM_SOFT, line=DIVIDER)
        add_text(s, x + Inches(0.3), y0 + Inches(0.25), Inches(5.5), Inches(0.3),
                 tags[i], font=F_BODY, size=10.5, color=ACCENT, bold=True)
        add_text(s, x + Inches(0.3), y0 + Inches(0.6), Inches(5.5), Inches(0.55),
                 titles[i], font=F_HEAD, size=18, color=INK, bold=True)
        add_text(s, x + Inches(0.3), y0 + Inches(1.25), Inches(5.5), Inches(1.3),
                 bodies[i], font=F_BODY, size=12, color=INK_SOFT,
                 line_spacing=1.4)

    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 5 — Prior approaches comparison
# =====================================================================
def slide_05_prior(idx):
    s = blank_slide(prs)
    page_header(s, "01 · Prior approaches", "Three families, three trade-offs")

    cols = [
        ("Online RL", "WebThinker · Kimi-Researcher",
         "Live internet  ·  high realism",
         ["Realistic interactions", "Unbounded knowledge"],
         ["Non-deterministic reward", "Cost grows with steps",
          "Variance bottlenecks scale"], "#B85F44"),
        ("Local retrieval", "Search-R1 · Search-o1",
         "Wikipedia-scale corpus",
         ["Cheap, fast, deterministic"],
         ["Narrow, homogeneous", "Misses real-web dynamics",
          "Caps the search-skill ceiling"], "#6B6356"),
        ("Simulated search", "ZeroSearch",
         "LLM-mimicked search engine",
         ["Fully controllable"],
         ["No page-level fidelity", "Hallucinated documents",
          "Limited transfer to real web"], "#8A8070"),
    ]
    x0 = Inches(0.6); col_w = Inches(4.05); gap = Inches(0.0)
    y0 = Inches(2.1)
    for i, (name, ex, sub, pros, cons, _) in enumerate(cols):
        x = x0 + Inches(i * 4.075)
        bx = add_rect(s, x, y0, col_w, Inches(4.7),
                      fill=CREAM_SOFT, line=DIVIDER)
        add_text(s, x + Inches(0.3), y0 + Inches(0.25), col_w - Inches(0.6), Inches(0.3),
                 ex.upper(), font=F_BODY, size=9, color=ACCENT, bold=True)
        add_text(s, x + Inches(0.3), y0 + Inches(0.6), col_w - Inches(0.6), Inches(0.55),
                 name, font=F_HEAD, size=20, color=INK, bold=True)
        add_text(s, x + Inches(0.3), y0 + Inches(1.2), col_w - Inches(0.6), Inches(0.5),
                 sub, font=F_BODY, size=11, color=INK_SOFT, italic=True)
        # divider
        add_hline(s, x + Inches(0.3), y0 + Inches(1.85), col_w - Inches(0.6))
        # pros
        add_text(s, x + Inches(0.3), y0 + Inches(1.95), col_w - Inches(0.6), Inches(0.3),
                 "STRENGTHS", font=F_BODY, size=9, color=GREEN, bold=True)
        for j, p in enumerate(pros):
            add_text(s, x + Inches(0.3), y0 + Inches(2.30 + j * 0.32), col_w - Inches(0.6), Inches(0.3),
                     "+ " + p, font=F_BODY, size=11, color=INK)
        # cons
        y_cons = y0 + Inches(2.30 + len(pros) * 0.32 + 0.25)
        add_text(s, x + Inches(0.3), y_cons, col_w - Inches(0.6), Inches(0.3),
                 "LIMITATIONS", font=F_BODY, size=9, color=ACCENT_DK, bold=True)
        for j, c in enumerate(cons):
            add_text(s, x + Inches(0.3), y_cons + Inches(0.35 + j * 0.32), col_w - Inches(0.6), Inches(0.3),
                     "−  " + c, font=F_BODY, size=11, color=INK_SOFT)

    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 6 — Our thesis
# =====================================================================
def slide_06_thesis(idx):
    s = blank_slide(prs)
    page_header(s, "01 · Thesis", "Mirror the web, isolate the noise")

    # Two halves: left text, right diagram of the goal
    add_text(s, Inches(0.6), Inches(2.2), Inches(7.0), Inches(0.5),
             "OUR ARGUMENT", font=F_BODY, size=10.5, color=ACCENT, bold=True)
    add_text(s, Inches(0.6), Inches(2.55), Inches(7.4), Inches(2.5),
             "Scaling Agentic RL for Deep Research requires a twin architecture:",
             font=F_HEAD, size=18, color=INK, line_spacing=1.35)

    items = [
        ("Mirroring", "Local environment behaves like the live web — same page-level fidelity, "
                      "same retrieval distributions, same browse dynamics."),
        ("Isolation", "Training is decoupled from the real internet — no rate limits, no API cost, "
                      "no non-deterministic reward, no rollout variance."),
    ]
    y = Inches(3.65)
    for t, b in items:
        add_text(s, Inches(0.6), y, Inches(1.5), Inches(0.45),
                 t, font=F_HEAD, size=15, color=ACCENT, bold=True)
        add_text(s, Inches(2.2), y + Inches(0.04), Inches(5.5), Inches(1.2),
                 b, font=F_BODY, size=12, color=INK_SOFT, line_spacing=1.4)
        y += Inches(1.25)

    # Right: outcome callout
    bx = add_rect(s, Inches(8.4), Inches(2.2), Inches(4.35), Inches(4.5),
                  fill=ACCENT, line=None)
    add_text(s, Inches(8.7), Inches(2.4), Inches(4), Inches(0.3),
             "OUTCOME", font=F_BODY, size=10.5, color=RGBColor(0xFF,0xE9,0xDE), bold=True)
    add_text(s, Inches(8.7), Inches(2.75), Inches(4), Inches(0.6),
             "LiteResearcher-4B", font=F_HEAD, size=24, color=RGBColor(0xFF,0xFF,0xFF), bold=True)
    add_text(s, Inches(8.7), Inches(3.45), Inches(4), Inches(0.4),
             "A 4B model that scales like a 30B+ one",
             font=F_BODY, size=12, color=RGBColor(0xFF,0xE9,0xDE), italic=True)
    add_hline(s, Inches(8.7), Inches(4.0), Inches(3.8), color=RGBColor(0xFF,0xC8,0xA8))
    # stats
    stats = [("71.3%", "GAIA"), ("78.0%", "Xbench-DS"), ("83.1%", "Frames")]
    y = Inches(4.2)
    for v, k in stats:
        add_text(s, Inches(8.7), y, Inches(2.5), Inches(0.55),
                 v, font=F_HEAD, size=22, color=RGBColor(0xFF,0xFF,0xFF), bold=True)
        add_text(s, Inches(10.7), y + Inches(0.18), Inches(2), Inches(0.35),
                 k, font=F_BODY, size=11.5, color=RGBColor(0xFF,0xE9,0xDE))
        y += Inches(0.72)

    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 7 — System overview (3 pillars)
# =====================================================================
def slide_07_overview(idx):
    s = blank_slide(prs)
    page_header(s, "02 · Method overview", "Three pillars of the LiteResearcher pipeline")
    img(s, "diag_architecture.png", Inches(0.6), Inches(2.05), w=Inches(12.15), h=Inches(4.85))
    page_footer(s, idx, TOTAL)
    return s

slide_04_problem(4); slide_05_prior(5); slide_06_thesis(6); slide_07_overview(7)
prs.save(OUT)
print(f"Now {len(prs.slides)} slides")

# =====================================================================
# Slide 8 — Pillar 1 intro
# =====================================================================
def slide_08_pillar1_intro(idx):
    s = blank_slide(prs)
    page_header(s, "03 · Pillar 1", "Co-construct training data and local corpus")

    add_text(s, Inches(0.6), Inches(2.15), Inches(11.5), Inches(0.45),
             "WHY CO-CONSTRUCTION", font=F_BODY, size=10.5, color=ACCENT, bold=True)
    add_text(s, Inches(0.6), Inches(2.55), Inches(12), Inches(1.4),
             "Tasks and corpus evolve together so that every synthetic query is\n"
             "answerable by browsing the local environment — no shortcut, no leak.",
             font=F_HEAD, size=19, color=INK, line_spacing=1.4)

    # 4-step pipeline strip
    steps = [
        ("Seed Corpus", "Wikipedia + BBC News\n→ structured QA"),
        ("Source Masking", "Delete the source page\nso the answer is non-trivial"),
        ("Rubric Filtering", "7-criteria LLM filter\n(independence, verifiability, ...)"),
        ("Corpus Expansion", "Fetch related web pages\n→ enrich the local pool"),
    ]
    y = Inches(4.3); w = Inches(2.9); gap = Inches(0.15); total = 4*w + 3*gap
    x0 = (SLIDE_W - total) / 2
    for i, (t, b) in enumerate(steps):
        x = x0 + i * (w + gap)
        bx = add_rect(s, x, y, w, Inches(2.05), fill=CREAM_SOFT, line=DIVIDER)
        add_text(s, x + Inches(0.25), y + Inches(0.25), w - Inches(0.5), Inches(0.4),
                 f"STEP {i+1:02d}", font=F_BODY, size=9.5, color=ACCENT, bold=True)
        add_text(s, x + Inches(0.25), y + Inches(0.65), w - Inches(0.5), Inches(0.5),
                 t, font=F_HEAD, size=15, color=INK, bold=True)
        add_text(s, x + Inches(0.25), y + Inches(1.15), w - Inches(0.5), Inches(0.9),
                 b, font=F_BODY, size=11, color=INK_SOFT, line_spacing=1.4)
    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 9 — Five atomic capabilities
# =====================================================================
def slide_09_atomic(idx):
    s = blank_slide(prs)
    page_header(s, "03 · Pillar 1", "Five atomic search capabilities")
    add_text(s, Inches(0.6), Inches(2.0), Inches(12), Inches(0.5),
             "The taxonomy of skills our data must elicit",
             font=F_BODY, size=12, color=INK_SOFT, italic=True)
    img(s, "diag_atomic_caps.png", Inches(0.4), Inches(2.5), w=Inches(12.5))
    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 10 — Pillar 2: Local tool environment
# =====================================================================
def slide_10_pillar2(idx):
    s = blank_slide(prs)
    page_header(s, "04 · Pillar 2", "Stable local tool environment")

    # Big numbers
    add_text(s, Inches(0.6), Inches(2.1), Inches(11.5), Inches(0.45),
             "THE VIRTUAL WEB", font=F_BODY, size=10.5, color=ACCENT, bold=True)
    nums = [("32 M", "pages"), ("1 M+", "domains"), ("73.2 M", "tool calls\nover full RL run")]
    y = Inches(2.55); x = Inches(0.6); col_w = Inches(3.7)
    for v, k in nums:
        add_text(s, x, y, col_w, Inches(0.7),
                 v, font=F_HEAD, size=34, color=INK, bold=True)
        add_text(s, x, y + Inches(0.95), col_w, Inches(0.6),
                 k, font=F_BODY, size=11.5, color=INK_SOFT, line_spacing=1.3)
        x += col_w

    # Two tool cards
    y = Inches(4.55); w = Inches(5.95); gap = Inches(0.25)
    cards = [
        ("Local Search Engine",
         "BGE-M3 hybrid dense + sparse\nMilvus index · DiskANN on disk\n~0.15 s / query  (≈ 10× faster than online)"),
        ("Local Browse Tool",
         "Full Markdown content in PostgreSQL\nKeyed by URL · tuned for 1,000 concurrent conn.\n~0.17 s / page  (≈ 46× faster than Jina Reader)"),
    ]
    for i, (t, b) in enumerate(cards):
        x = Inches(0.6) + i * (w + gap)
        bx = add_rect(s, x, y, w, Inches(2.1), fill=CREAM_SOFT, line=DIVIDER)
        add_text(s, x + Inches(0.3), y + Inches(0.2), w - Inches(0.6), Inches(0.35),
                 ["INDEX · RETRIEVE", "STORE · SERVE"][i],
                 font=F_BODY, size=9.5, color=ACCENT, bold=True)
        add_text(s, x + Inches(0.3), y + Inches(0.6), w - Inches(0.6), Inches(0.5),
                 t, font=F_HEAD, size=18, color=INK, bold=True)
        add_text(s, x + Inches(0.3), y + Inches(1.2), w - Inches(0.6), Inches(0.9),
                 b, font=F_BODY, size=11.5, color=INK_SOFT, line_spacing=1.45)
    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 11 — Pillar 3: Curriculum RL
# =====================================================================
def slide_11_pillar3(idx):
    s = blank_slide(prs)
    page_header(s, "05 · Pillar 3", "Difficulty-aware curriculum learning")

    # left: bullet content
    add_text(s, Inches(0.6), Inches(2.15), Inches(7), Inches(0.45),
             "THE PROBLEM · TRAINING SATURATION", font=F_BODY, size=10.5,
             color=ACCENT, bold=True)
    add_text(s, Inches(0.6), Inches(2.55), Inches(7.2), Inches(1.6),
             "When too easy → no gradient. When too hard → no signal.\n"
             "Single-stage RL plateaus at the model's current difficulty band.",
             font=F_BODY, size=13, color=INK_SOFT, line_spacing=1.45)

    add_text(s, Inches(0.6), Inches(4.25), Inches(7), Inches(0.45),
             "THE FIX · TWO LEVERS", font=F_BODY, size=10.5,
             color=ACCENT, bold=True)
    levers = [
        ("Difficulty filter",
         "Before each stage: K=8 rollouts per query, keep only those\n"
         "with correct-count c ∈ [1, 7]. Discard trivial and impossible."),
        ("Multi-stage curriculum",
         "Each stage  →  increase task difficulty + context length\n"
         "(Stage 1 = 32K  →  Stage 2 = 48K, with harder data mix)."),
    ]
    y = Inches(4.65)
    for t, b in levers:
        add_text(s, Inches(0.6), y, Inches(2), Inches(0.4),
                 t, font=F_HEAD, size=13, color=ACCENT, bold=True)
        add_text(s, Inches(2.5), y - Inches(0.02), Inches(5.4), Inches(1.0),
                 b, font=F_BODY, size=11, color=INK_SOFT, line_spacing=1.4)
        y += Inches(1.1)

    # right: formula card
    bx = add_rect(s, Inches(8.4), Inches(2.15), Inches(4.35), Inches(4.55),
                  fill=CREAM_SOFT, line=DIVIDER)
    add_text(s, Inches(8.65), Inches(2.35), Inches(4), Inches(0.35),
             "ON-POLICY GRPO", font=F_BODY, size=10.5, color=ACCENT, bold=True)
    add_text(s, Inches(8.65), Inches(2.7), Inches(4), Inches(0.5),
             "The training objective", font=F_HEAD, size=15, color=INK, bold=True)
    # render formula as bitmap-like styled text
    add_text(s, Inches(8.65), Inches(3.35), Inches(4), Inches(2.0),
             "J(θ) = E [ (1/K) Σ min ( rᵢ·Aᵢ ,",
             font=F_MONO, size=11, color=INK)
    add_text(s, Inches(8.65), Inches(3.65), Inches(4), Inches(2.0),
             "       clip(rᵢ, 1-ε_lo, 1+ε_hi)·Aᵢ ) ]",
             font=F_MONO, size=11, color=INK)
    add_text(s, Inches(8.65), Inches(4.15), Inches(4), Inches(2.0),
             "rᵢ(θ) = π_θ(oᵢ|q) / π_old(oᵢ|q)",
             font=F_MONO, size=11, color=INK_SOFT)
    add_hline(s, Inches(8.65), Inches(4.85), Inches(3.85))
    add_paragraphs(s, Inches(8.65), Inches(4.95), Inches(4), Inches(1.5),
                   [("· No KL penalty", {"size":11, "color":INK_SOFT}),
                    ("· No entropy regularizer", {"size":11, "color":INK_SOFT}),
                    ("· Strictly on-policy: each rollout used once", {"size":11, "color":INK_SOFT})],
                   line_spacing=1.5, space_after=4)
    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 12 — Section divider for Experiments
# =====================================================================
def slide_12_divider_exp(idx):
    s = blank_slide(prs)
    # giant section number on left
    add_text(s, Inches(0.6), Inches(2.4), Inches(3), Inches(2.5),
             "06", font=F_HEAD, size=140, color=ACCENT, bold=True)
    # title
    add_text(s, Inches(4.5), Inches(2.7), Inches(8), Inches(0.5),
             "EXPERIMENTS", font=F_BODY, size=11, color=ACCENT, bold=True)
    add_text(s, Inches(4.5), Inches(3.15), Inches(8.5), Inches(1.5),
             "Main results & ablations", font=F_HEAD, size=44, color=INK, bold=True)
    add_text(s, Inches(4.5), Inches(4.4), Inches(8.5), Inches(2),
             "8 benchmarks · ablations on data, environment,\n"
             "on/off-policy, multi-stage, SFT-only",
             font=F_BODY, size=14, color=INK_SOFT, line_spacing=1.5, italic=True)
    add_hline(s, Inches(4.5), Inches(6.2), Inches(8.3), color=ACCENT, weight=1.2)
    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 13 — Experimental setup
# =====================================================================
def slide_13_setup(idx):
    s = blank_slide(prs)
    page_header(s, "06 · Setup", "Training recipe at a glance")
    items = [
        ("Base model",         "Qwen3-4B-Thinking-2507"),
        ("Cold-start SFT",     "trajectories distilled from Tongyi DeepResearch"),
        ("Batch size",         "global 128  ·  8 rollouts per query  ·  on-policy"),
        ("Learning rate",      "1 × 10⁻⁶  (constant)"),
        ("Loss",               "GRPO  ·  no KL  ·  no entropy regularizer"),
        ("Context length",     "Stage 1 = 32 K  →  Stage 2 = 48 K"),
        ("Eval tools",         "Serper for search  ·  Jina Reader for browse (online APIs)"),
        ("Benchmarks",         "GAIA · Xbench-DS · BrowseComp · HLE · Frames · WebWalker · Seal-0"),
    ]
    y0 = Inches(2.15); row_h = Inches(0.55)
    add_text(s, Inches(0.6), y0, Inches(11.5), Inches(0.4),
             "RL TRAINING CONFIGURATION", font=F_BODY, size=10.5, color=ACCENT, bold=True)
    y = y0 + Inches(0.5)
    for i, (k, v) in enumerate(items):
        bg = CREAM_SOFT if i % 2 == 0 else CREAM
        if bg != CREAM:
            add_rect(s, Inches(0.6), y, Inches(12.15), row_h - Inches(0.05),
                     fill=bg, line=None)
        add_text(s, Inches(0.9), y + Inches(0.13), Inches(3.4), Inches(0.4),
                 k, font=F_BODY, size=12, color=INK_SOFT)
        add_text(s, Inches(4.4), y + Inches(0.13), Inches(8.3), Inches(0.4),
                 v, font=F_BODY, size=12.5, color=INK, bold=True)
        y += row_h
    page_footer(s, idx, TOTAL)
    return s

# Build slides 8-13
slide_08_pillar1_intro(8)
slide_09_atomic(9)
slide_10_pillar2(10)
slide_11_pillar3(11)
slide_12_divider_exp(12)
slide_13_setup(13)
prs.save(OUT)
print(f"Now {len(prs.slides)} slides")

# =====================================================================
# Slide 14 — Main results table (highlight LiteResearcher)
# =====================================================================
def slide_14_main_results(idx):
    s = blank_slide(prs)
    page_header(s, "06 · Main results", "8 benchmarks · 4B model · open-source SOTA")

    # Compact table: model / size / GAIA / Xbench / Frames / BrowseComp
    rows = [
        ("Model",                        "Size", "GAIA",  "Xbench-DS", "Frames", "BrowseComp"),
        ("Claude-4.5-Sonnet",            "—",    "71.2",  "66.0",      "80.7",   "—"),
        ("OpenAI GPT-5-high",            "—",    "—",     "77.8",      "—",      "—"),
        ("GLM-4.6",                      "—",    "71.9",  "70.0",      "—",      "—"),
        ("DeepSeek-V3.2",                "—",    "—",     "—",         "80.2",   "—"),
        ("Tongyi DeepResearch",          "30B",  "70.9",  "75.0",      "—",      "—"),
        ("WebSailor",                    "30B",  "53.2",  "53.3",      "—",      "—"),
        ("AgentCPM-Explore",             "4B",   "63.9",  "70.0",      "—",      "—"),
        ("LiteResearcher (ours)",        "4B",   "71.3",  "78.0",      "83.1",   "32.5"),
    ]
    col_widths = [Inches(4.1), Inches(1.0), Inches(1.5), Inches(1.5), Inches(1.5), Inches(1.6)]
    x0 = Inches(0.6); y0 = Inches(2.15)
    row_h = Inches(0.42)
    for r, row in enumerate(rows):
        is_header = r == 0
        is_ours = "ours" in row[0]
        x = x0
        for c, val in enumerate(row):
            # background
            if is_ours:
                bg = ACCENT
                fg = RGBColor(0xFF,0xFF,0xFF)
            elif is_header:
                bg = INK_PANEL
                fg = INK
            else:
                bg = CREAM_SOFT if r % 2 else CREAM
                fg = INK
            if bg != CREAM:
                add_rect(s, x, y0 + r * row_h,
                         col_widths[c], row_h - Inches(0.02),
                         fill=bg, line=None)
            align = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            sz = 12 if not is_header else 11.5
            add_text(s, x + Inches(0.18 if c == 0 else 0),
                     y0 + r * row_h + Inches(0.10),
                     col_widths[c] - Inches(0.18 if c == 0 else 0),
                     Inches(0.4),
                     str(val), font=F_BODY, size=sz, color=fg,
                     bold=is_ours or is_header, align=align)
            x += col_widths[c]
    # caption
    add_text(s, Inches(0.6), y0 + len(rows) * row_h + Inches(0.2), Inches(12),
             Inches(0.4),
             "All scores in %. LiteResearcher-4B matches or exceeds models 8× its size and most commercial systems.",
             font=F_BODY, size=10.5, color=INK_SOFT, italic=True)
    # callout: 4B vs 30B
    bx = add_rect(s, Inches(0.6), Inches(6.3), Inches(12.15), Inches(0.65),
                  fill=INK_PANEL, line=None)
    add_text(s, Inches(0.8), Inches(6.43), Inches(11.7), Inches(0.4),
             "4 B beats 30 B  ·  open-source SOTA on GAIA · Xbench · Frames  ·  zero marginal cost during training",
             font=F_BODY, size=12.5, color=INK, bold=True, align=PP_ALIGN.CENTER)
    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 15 — Cost ablation
# =====================================================================
def slide_15_cost(idx):
    s = blank_slide(prs)
    page_header(s, "07 · Ablation", "Local vs. live internet — cost & latency")

    # left: chart
    img(s, "diag_cost.png", Inches(0.4), Inches(2.15), w=Inches(7.5))
    # right: callout
    bx = add_rect(s, Inches(8.4), Inches(2.15), Inches(4.35), Inches(4.55),
                  fill=ACCENT, line=None)
    add_text(s, Inches(8.65), Inches(2.4), Inches(4), Inches(0.35),
             "TOTAL TOOL CALLS", font=F_BODY, size=10.5,
             color=RGBColor(0xFF,0xE9,0xDE), bold=True)
    add_text(s, Inches(8.65), Inches(2.78), Inches(4), Inches(0.7),
             "73.2 M", font=F_HEAD, size=42, color=RGBColor(0xFF,0xFF,0xFF), bold=True)
    add_text(s, Inches(8.65), Inches(3.55), Inches(4), Inches(0.45),
             "45.8 M search + 27.4 M browse\nover the full RL run",
             font=F_BODY, size=11.5, color=RGBColor(0xFF,0xE9,0xDE), line_spacing=1.4)
    add_hline(s, Inches(8.65), Inches(4.4), Inches(3.85), color=RGBColor(0xFF,0xC8,0xA8))
    add_text(s, Inches(8.65), Inches(4.55), Inches(4), Inches(0.4),
             "SPEED-UP", font=F_BODY, size=10.5,
             color=RGBColor(0xFF,0xE9,0xDE), bold=True)
    add_text(s, Inches(8.65), Inches(4.95), Inches(4), Inches(1.5),
             "10× faster search\n46× faster browse",
             font=F_HEAD, size=20, color=RGBColor(0xFF,0xFF,0xFF), bold=True,
             line_spacing=1.35)
    add_text(s, Inches(8.65), Inches(6.05), Inches(4), Inches(0.5),
             "Throughput is what makes\nscalable on-policy RL feasible.",
             font=F_BODY, size=11, color=RGBColor(0xFF,0xE9,0xDE),
             italic=True, line_spacing=1.4)
    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 16 — Synthetic data ablation
# =====================================================================
def slide_16_data_ablation(idx):
    s = blank_slide(prs)
    page_header(s, "07 · Ablation", "Effect of our synthetic data")

    add_text(s, Inches(0.6), Inches(2.15), Inches(11.5), Inches(0.5),
             "DOES SCALING INFORMATION SOURCES PAY OFF?",
             font=F_BODY, size=10.5, color=ACCENT, bold=True)

    # comparison table
    rows = [
        ("Training data",                 "GAIA", "Xbench-DS"),
        ("Multi-hop data only",           "58.7", "66.3"),
        ("Multi-hop + our synthetic",     "66.8", "71.0"),
    ]
    col_widths = [Inches(8.0), Inches(2.0), Inches(2.15)]
    x0 = Inches(0.6); y0 = Inches(2.85); row_h = Inches(0.6)
    for r, row in enumerate(rows):
        is_header = r == 0
        is_best   = r == 2
        x = x0
        for c, val in enumerate(row):
            bg = INK_PANEL if is_header else (ACCENT if is_best else CREAM_SOFT)
            fg = RGBColor(0xFF,0xFF,0xFF) if is_best else INK
            add_rect(s, x, y0 + r * row_h,
                     col_widths[c], row_h - Inches(0.05), fill=bg, line=None)
            align = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            add_text(s, x + Inches(0.25 if c == 0 else 0),
                     y0 + r * row_h + Inches(0.18),
                     col_widths[c] - Inches(0.25 if c == 0 else 0),
                     Inches(0.4),
                     val, font=F_BODY, size=13.5, color=fg,
                     bold=is_header or is_best, align=align)
            x += col_widths[c]

    # delta callout
    add_text(s, Inches(0.6), Inches(5.0), Inches(11.5), Inches(0.45),
             "INTERPRETATION", font=F_BODY, size=10.5, color=ACCENT, bold=True)
    add_text(s, Inches(0.6), Inches(5.4), Inches(12), Inches(1.5),
             "Adding our synthetic data lifts both benchmarks by +8.1 / +4.7 points.\n"
             "A scaled-and-filtered information source covers the long-tail of search\n"
             "patterns that hand-crafted multi-hop data misses.",
             font=F_BODY, size=13, color=INK_SOFT, line_spacing=1.5)
    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 17 — On-policy vs off-policy
# =====================================================================
def slide_17_onpolicy(idx):
    s = blank_slide(prs)
    page_header(s, "07 · Ablation", "On-policy vs. off-policy GRPO")

    # left text
    add_text(s, Inches(0.6), Inches(2.15), Inches(7), Inches(0.5),
             "SETUP", font=F_BODY, size=10.5, color=ACCENT, bold=True)
    add_text(s, Inches(0.6), Inches(2.55), Inches(7.3), Inches(2.5),
             "Off-policy variant: each rollout batch (256) is split into 4 mini-batches\n"
             "and consumed in 4 successive updates.\n\n"
             "On-policy: each rollout batch is consumed in a single update and discarded.",
             font=F_BODY, size=12, color=INK_SOFT, line_spacing=1.5)

    add_text(s, Inches(0.6), Inches(4.45), Inches(7), Inches(0.5),
             "RESULT", font=F_BODY, size=10.5, color=ACCENT, bold=True)
    add_text(s, Inches(0.6), Inches(4.85), Inches(7.3), Inches(2),
             "Off-policy gains reward earlier but eventually declines.\n"
             "On-policy keeps improving — higher final GAIA validation.",
             font=F_BODY, size=13, color=INK, line_spacing=1.45)

    # right: numbers
    bx = add_rect(s, Inches(8.4), Inches(2.15), Inches(4.35), Inches(4.55),
                  fill=CREAM_SOFT, line=DIVIDER)
    add_text(s, Inches(8.65), Inches(2.35), Inches(4), Inches(0.4),
             "GAIA VALIDATION ACCURACY", font=F_BODY, size=10.5,
             color=ACCENT, bold=True)
    add_hline(s, Inches(8.65), Inches(2.95), Inches(3.85))
    y = Inches(3.15)
    for name, val, c in [("Off-policy", "66.8 %", INK_SOFT),
                          ("On-policy",  "68.9 %", ACCENT)]:
        add_text(s, Inches(8.65), y, Inches(2), Inches(0.5),
                 name, font=F_BODY, size=13, color=c, bold=True)
        add_text(s, Inches(10.5), y, Inches(2.2), Inches(0.5),
                 val, font=F_HEAD, size=22, color=c, bold=True,
                 align=PP_ALIGN.RIGHT)
        y += Inches(0.95)
    add_hline(s, Inches(8.65), Inches(5.10), Inches(3.85))
    add_text(s, Inches(8.65), Inches(5.30), Inches(4), Inches(1.5),
             "Multiple updates per batch  →  policy lag  →  trajectory mismatch  →  ceiling drops.",
             font=F_BODY, size=11.5, color=INK_SOFT, italic=True, line_spacing=1.4)
    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 18 — Multi-stage training
# =====================================================================
def slide_18_multistage(idx):
    s = blank_slide(prs)
    page_header(s, "07 · Ablation", "Effectiveness of multi-stage training")

    # left: image
    img(s, "fig1_accuracy_sim2real.png", Inches(0.4), Inches(2.15),
        w=Inches(8.5))

    # right: numbers
    bx = add_rect(s, Inches(9.2), Inches(2.15), Inches(3.55), Inches(4.55),
                  fill=ACCENT, line=None)
    add_text(s, Inches(9.4), Inches(2.4), Inches(3.2), Inches(0.4),
             "GAIA · EMA-SMOOTHED", font=F_BODY, size=10.5,
             color=RGBColor(0xFF,0xE9,0xDE), bold=True)
    add_hline(s, Inches(9.4), Inches(2.95), Inches(3.2), color=RGBColor(0xFF,0xC8,0xA8))
    y = Inches(3.15)
    for label, val, sub in [
        ("Stage 1 plateau", "64.7 %", "at ~step 220"),
        ("Stage 2 end",      "68.3 %", "+3.6 pts"),
    ]:
        add_text(s, Inches(9.4), y, Inches(3.2), Inches(0.35),
                 label, font=F_BODY, size=11, color=RGBColor(0xFF,0xE9,0xDE), bold=True)
        add_text(s, Inches(9.4), y + Inches(0.38), Inches(3.2), Inches(0.55),
                 val, font=F_HEAD, size=28, color=RGBColor(0xFF,0xFF,0xFF), bold=True)
        add_text(s, Inches(9.4), y + Inches(0.95), Inches(3.2), Inches(0.4),
                 sub, font=F_BODY, size=11, color=RGBColor(0xFF,0xE9,0xDE), italic=True)
        y += Inches(1.55)
    add_hline(s, Inches(9.4), Inches(6.05), Inches(3.2), color=RGBColor(0xFF,0xC8,0xA8))
    add_text(s, Inches(9.4), Inches(6.15), Inches(3.2), Inches(0.6),
             "Stage 2 lifts the plateau by\nraising difficulty + context length.",
             font=F_BODY, size=11, color=RGBColor(0xFF,0xE9,0xDE),
             italic=True, line_spacing=1.4)
    page_footer(s, idx, TOTAL)
    return s

# =====================================================================
# Slide 19 — RL over SFT
# =====================================================================
def slide_19_rl_vs_sft(idx):
    s = blank_slide(prs)
    page_header(s, "07 · Ablation", "Contribution of RL over SFT")

    add_text(s, Inches(0.6), Inches(2.15), Inches(11.5), Inches(0.5),
             "SFT TEACHER  ·  TONGYI DEEPRESEARCH (30B)",
             font=F_BODY, size=10.5, color=ACCENT, bold=True)

    # 3 columns of numbers
    cols = [
        ("Teacher (30B)", "70.9 %", "where the teacher caps", INK_SOFT),
        ("SFT only (4B)", "55.6 %", "−15.3 pts vs teacher", INK_SOFT),
        ("+ RL  (4B, ours)", "71.3 %", "+15.7 pts over SFT", ACCENT),
    ]
    y0 = Inches(2.85); w = Inches(4.05); gap = Inches(0.05)
    x0 = Inches(0.6)
    for i, (n, v, sub, color) in enumerate(cols):
        x = x0 + i * (w + gap)
        bg = ACCENT if color == ACCENT else CREAM_SOFT
        bx = add_rect(s, x, y0, w, Inches(2.95), fill=bg, line=DIVIDER if bg != ACCENT else None)
        fg = RGBColor(0xFF,0xFF,0xFF) if bg == ACCENT else INK
        sub_fg = RGBColor(0xFF,0xE9,0xDE) if bg == ACCENT else INK_SOFT
        add_text(s, x + Inches(0.3), y0 + Inches(0.3), w - Inches(0.6), Inches(0.4),
                 n.upper(), font=F_BODY, size=10.5, color=sub_fg, bold=True)
        add_text(s, x + Inches(0.3), y0 + Inches(0.95), w - Inches(0.6), Inches(0.9),
                 v, font=F_HEAD, size=44, color=fg, bold=True)
        add_text(s, x + Inches(0.3), y0 + Inches(2.15), w - Inches(0.6), Inches(0.5),
                 sub, font=F_BODY, size=12, color=sub_fg, italic=True)

    # bottom insight
    bx2 = add_rect(s, Inches(0.6), Inches(6.05), Inches(12.15), Inches(0.7),
                   fill=INK_PANEL, line=None)
    add_text(s, Inches(0.85), Inches(6.20), Inches(11.7), Inches(0.45),
             "The primary performance driver is the RL framework — not teacher distillation.",
             font=F_HEAD, size=14, color=INK, italic=True, align=PP_ALIGN.CENTER)
    page_footer(s, idx, TOTAL)
    return s

# Build slides 14-19
slide_14_main_results(14)
slide_15_cost(15)
slide_16_data_ablation(16)
slide_17_onpolicy(17)
slide_18_multistage(18)
slide_19_rl_vs_sft(19)
prs.save(OUT)
print(f"Now {len(prs.slides)} slides")

# ─── APPENDIX DIVIDER + BEHAVIOR ANALYSIS ───────────────────────────────

def slide_20_appendix_divider(idx):
    s = blank_slide(prs)
    # eyebrow
    add_text(s, Inches(0.6), Inches(0.65), Inches(6), Inches(0.4),
             "APPENDIX  ·  BEHAVIOR ANALYSIS", font=F_BODY, size=11.5,
             color=ACCENT, bold=True)
    add_hline(s, Inches(0.6), Inches(1.05), Inches(1.2), color=ACCENT, weight=2.2)

    # giant number
    add_text(s, Inches(0.6), Inches(1.5), Inches(7), Inches(3.6),
             "08", font=F_HEAD, size=240, color=ACCENT, bold=True)

    # title block at right
    add_text(s, Inches(7.4), Inches(2.5), Inches(5.4), Inches(1.4),
             "Why does the policy improve?", font=F_HEAD, size=36,
             color=INK, bold=True)
    add_paragraphs(s, Inches(7.4), Inches(3.8), Inches(5.4), Inches(2.2), [
        ("A behavior-level autopsy of 41 checkpoints across", dict(size=15, color=INK_SOFT, font=F_BODY)),
        ("S1 (steps 0-220) → S2 (steps 0-570), spliced at step 220.", dict(size=15, color=INK_SOFT, font=F_BODY)),
    ], line_spacing=1.35)

    add_hline(s, Inches(7.4), Inches(5.4), Inches(4.5), color=ACCENT, weight=2.2)
    add_text(s, Inches(7.4), Inches(5.55), Inches(5.4), Inches(0.5),
             "30 metrics  ·  300 traj/step  ·  GAIA every 20 steps",
             font=F_BODY, size=12, color=INK_SOFT, italic=True)

    page_footer(s, idx, TOTAL)
    return s


def slide_21_motivation(idx):
    s = blank_slide(prs)
    page_header(s, "Appendix · Behavior", "What changed about the model's behavior?")

    # Left: question card
    add_rect(s, Inches(0.6), Inches(2.15), Inches(5.8), Inches(4.85),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.04)
    add_text(s, Inches(0.95), Inches(2.45), Inches(5.1), Inches(0.45),
             "MOTIVATION", font=F_BODY, size=11, color=ACCENT, bold=True)
    add_paragraphs(s, Inches(0.95), Inches(2.95), Inches(5.1), Inches(3.9), [
        ("The aggregate reward curve goes up — but what does the policy", dict(size=14, color=INK, font=F_BODY)),
        ("actually do differently at step 790 vs step 0?", dict(size=14, color=INK, font=F_BODY, italic=True)),
        ("", dict(size=8)),
        ("We log 30 per-trajectory behavior signals at every", dict(size=13, color=INK_SOFT, font=F_BODY)),
        ("checkpoint, then look for monotone shifts that correlate", dict(size=13, color=INK_SOFT, font=F_BODY)),
        ("with the GAIA accuracy gain.", dict(size=13, color=INK_SOFT, font=F_BODY)),
        ("", dict(size=8)),
        ("Goal: isolate causal patterns from incidental noise.", dict(size=13, color=ACCENT_DK, font=F_BODY, italic=True, bold=True)),
    ], line_spacing=1.30)

    # Right: setup card
    add_rect(s, Inches(6.65), Inches(2.15), Inches(6.1), Inches(4.85),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.04)
    add_text(s, Inches(7.0), Inches(2.45), Inches(5.4), Inches(0.45),
             "ANALYSIS SETUP", font=F_BODY, size=11, color=ACCENT, bold=True)

    # data table
    rows = [
        ("Checkpoints sampled", "41"),
        ("Train rollouts per ckpt", "300 trajectories"),
        ("GAIA-eval rollouts per ckpt", "412 tasks  ×  pass@1, pass@4"),
        ("Behavior metrics extracted", "30 per trajectory"),
        ("Splice point", "step 220 — S1 end → S2 start"),
        ("Total trajectories analyzed", "~ 24,000"),
    ]
    y = Inches(3.0); rh = Inches(0.52)
    for i, (k, v) in enumerate(rows):
        bg = CREAM if i % 2 == 0 else CREAM_SOFT
        add_rect(s, Inches(7.0), y, Inches(5.4), rh, fill=bg, line=None)
        add_text(s, Inches(7.15), y + Inches(0.1), Inches(2.4), Inches(0.32),
                 k, font=F_BODY, size=11, color=INK_SOFT)
        add_text(s, Inches(9.55), y + Inches(0.1), Inches(2.85), Inches(0.32),
                 v, font=F_BODY, size=11.5, color=INK, bold=True)
        y += rh

    page_footer(s, idx, TOTAL)
    return s


def slide_22_outcome(idx):
    s = blank_slide(prs)
    page_header(s, "Appendix · Behavior", "What we measured at the outcome level")

    # full-width image with breathing room
    s.shapes.add_picture("fig1_accuracy_sim2real.png",
                         Inches(0.9), Inches(2.05),
                         width=Inches(8.8), height=Inches(4.85))

    # right narrative
    add_rect(s, Inches(10.05), Inches(2.05), Inches(2.7), Inches(4.85),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.04)
    add_text(s, Inches(10.25), Inches(2.30), Inches(2.4), Inches(0.4),
             "OBSERVATION", font=F_BODY, size=10.5, color=ACCENT, bold=True)
    add_paragraphs(s, Inches(10.25), Inches(2.75), Inches(2.4), Inches(4.0), [
        ("Train reward and GAIA", dict(size=12, color=INK, font=F_BODY, bold=True)),
        ("evaluation co-improve", dict(size=12, color=INK, font=F_BODY, bold=True)),
        ("through both stages.", dict(size=12, color=INK, font=F_BODY, bold=True)),
        ("", dict(size=8)),
        ("S1 saturates near", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("step 220 — S2 lifts", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("the plateau without", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("regression.", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("", dict(size=8)),
        ("→ Sim2Real transfer", dict(size=11.5, color=ACCENT_DK, font=F_BODY, italic=True)),
        ("    is sustained.", dict(size=11.5, color=ACCENT_DK, font=F_BODY, italic=True)),
    ], line_spacing=1.30)

    page_footer(s, idx, TOTAL)
    return s


slide_20_appendix_divider(20)
slide_21_motivation(21)
slide_22_outcome(22)
prs.save(OUT)
print(f"Now {len(prs.slides)} slides")

# ─── THE FOUR MECHANISMS ────────────────────────────────────────────────

def slide_23_mechanisms_overview(idx):
    s = blank_slide(prs)
    page_header(s, "Appendix · Behavior", "Four mechanisms behind the gain")

    # picture takes up most of the slide
    s.shapes.add_picture("fig2_mechanisms.png",
                         Inches(0.6), Inches(2.05),
                         width=Inches(8.6), height=Inches(4.95))

    # right narrative card
    add_rect(s, Inches(9.4), Inches(2.05), Inches(3.35), Inches(4.95),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.04)
    add_text(s, Inches(9.6), Inches(2.30), Inches(3.0), Inches(0.4),
             "THE FOUR LEVERS", font=F_BODY, size=10.5, color=ACCENT, bold=True)

    items = [
        ("M1", "Tool-choice shift", "search → browse",       "0.46 → 0.76"),
        ("M2", "Trajectory re-alloc", "compress → expand",   "28 → 19 → 45"),
        ("M3", "Per-think depth",   "richer reasoning blocks", "387 → 626 chars"),
        ("M4", "Self-correction",   "more hedge volume",       "1.5 → 5.0 tokens"),
    ]
    y = Inches(2.75); rh = Inches(1.05)
    for tag, name, sub, num in items:
        add_text(s, Inches(9.6), y, Inches(0.55), Inches(0.4),
                 tag, font=F_HEAD, size=15, color=ACCENT, bold=True)
        add_text(s, Inches(10.2), y, Inches(2.5), Inches(0.4),
                 name, font=F_BODY, size=12, color=INK, bold=True)
        add_text(s, Inches(10.2), y + Inches(0.38), Inches(2.5), Inches(0.32),
                 sub, font=F_BODY, size=10.5, color=INK_SOFT, italic=True)
        add_text(s, Inches(10.2), y + Inches(0.65), Inches(2.5), Inches(0.32),
                 num, font=F_MONO, size=10.5, color=ACCENT_DK, bold=True)
        y += rh

    page_footer(s, idx, TOTAL)
    return s


def slide_24_m1_tool_choice(idx):
    s = blank_slide(prs)
    page_header(s, "M1 · Tool-choice shift", "From keyword search to deep browse")

    # left: panel-(a) only
    add_rect(s, Inches(0.6), Inches(2.05), Inches(7.6), Inches(4.9),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.03)
    s.shapes.add_picture("panel_a.png",
                         Inches(0.85), Inches(2.30),
                         width=Inches(7.1), height=Inches(4.4))

    # right: numbers + interpretation
    add_rect(s, Inches(8.4), Inches(2.05), Inches(4.35), Inches(4.9),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.03)
    add_text(s, Inches(8.7), Inches(2.30), Inches(4.0), Inches(0.4),
             "browse / (search + browse)", font=F_MONO, size=11, color=ACCENT, bold=True)

    # 4 stat rows
    stats = [
        ("step  0",    "0.46", INK),
        ("step 220",   "0.56", INK),
        ("step 460",   "0.72", INK),
        ("step 790",   "0.76", ACCENT_DK),
    ]
    y = Inches(2.95); rh = Inches(0.55)
    for k, v, color in stats:
        bg = CREAM_SOFT if color == INK else INK_PANEL
        add_rect(s, Inches(8.7), y, Inches(3.75), rh, fill=bg, line=None)
        add_text(s, Inches(8.85), y + Inches(0.13), Inches(1.5), Inches(0.32),
                 k, font=F_MONO, size=12, color=INK_SOFT)
        add_text(s, Inches(10.6), y + Inches(0.13), Inches(1.8), Inches(0.32),
                 v, font=F_HEAD, size=14, color=color, bold=True)
        y += rh + Inches(0.05)

    # interpretation
    y = Inches(5.55)
    add_text(s, Inches(8.7), y, Inches(4.0), Inches(0.35),
             "INTERPRETATION", font=F_BODY, size=10, color=ACCENT, bold=True)
    add_paragraphs(s, Inches(8.7), y + Inches(0.4), Inches(4.0), Inches(2.2), [
        ("Policy stops treating search as the answer", dict(size=11.5, color=INK, font=F_BODY)),
        ("and starts using it as a pointer toward", dict(size=11.5, color=INK, font=F_BODY)),
        ("evidence — which it then opens and reads.", dict(size=11.5, color=INK, font=F_BODY)),
    ], line_spacing=1.25)

    page_footer(s, idx, TOTAL)
    return s


def slide_25_m2_reallocation(idx):
    s = blank_slide(prs)
    page_header(s, "M2 · Trajectory re-allocation", "Compress, then expand into harder problems")

    add_rect(s, Inches(0.6), Inches(2.05), Inches(7.6), Inches(4.9),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.03)
    s.shapes.add_picture("panel_b.png",
                         Inches(0.85), Inches(2.30),
                         width=Inches(7.1), height=Inches(4.4))

    # right narrative
    add_rect(s, Inches(8.4), Inches(2.05), Inches(4.35), Inches(4.9),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.03)
    add_text(s, Inches(8.7), Inches(2.30), Inches(4.0), Inches(0.4),
             "agent turns / trajectory", font=F_MONO, size=11, color=ACCENT, bold=True)

    # U-shape callout
    add_text(s, Inches(8.7), Inches(2.85), Inches(4.0), Inches(0.5),
             "28  →  19  →  45", font=F_HEAD, size=28, color=ACCENT_DK, bold=True)
    add_text(s, Inches(8.7), Inches(3.50), Inches(4.0), Inches(0.4),
             "S1 start  ·  S1 end  ·  S2 end", font=F_BODY, size=11, color=INK_SOFT, italic=True)

    add_hline(s, Inches(8.7), Inches(4.05), Inches(3.85), color=DIVIDER, weight=1)

    add_paragraphs(s, Inches(8.7), Inches(4.20), Inches(4.0), Inches(2.7), [
        ("Phase A · S1 (0→220)", dict(size=12, color=INK, font=F_BODY, bold=True)),
        ("Policy learns to compress —", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("fewer wasted turns per task.", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("", dict(size=6)),
        ("Phase B · S2 (220→790)", dict(size=12, color=ACCENT_DK, font=F_BODY, bold=True)),
        ("Same policy meets harder tasks", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("and an enlarged 48k context — it", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("re-invests the budget for depth.", dict(size=11, color=INK_SOFT, font=F_BODY)),
    ], line_spacing=1.20)

    page_footer(s, idx, TOTAL)
    return s


def slide_26_m3_think_depth(idx):
    s = blank_slide(prs)
    page_header(s, "M3 · Per-think depth", "Each reasoning block carries more content")

    add_rect(s, Inches(0.6), Inches(2.05), Inches(7.6), Inches(4.9),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.03)
    s.shapes.add_picture("panel_c.png",
                         Inches(0.85), Inches(2.30),
                         width=Inches(7.1), height=Inches(4.4))

    # right narrative
    add_rect(s, Inches(8.4), Inches(2.05), Inches(4.35), Inches(4.9),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.03)
    add_text(s, Inches(8.7), Inches(2.30), Inches(4.0), Inches(0.4),
             "chars / <think> block", font=F_MONO, size=11, color=ACCENT, bold=True)

    add_text(s, Inches(8.7), Inches(2.85), Inches(4.0), Inches(0.7),
             "387  →  626", font=F_HEAD, size=40, color=ACCENT_DK, bold=True)
    add_text(s, Inches(8.7), Inches(3.70), Inches(4.0), Inches(0.4),
             "≈ +62 % per reasoning block", font=F_BODY, size=12, color=INK_SOFT, italic=True)

    add_hline(s, Inches(8.7), Inches(4.20), Inches(3.85), color=DIVIDER, weight=1)

    add_text(s, Inches(8.7), Inches(4.35), Inches(4.0), Inches(0.35),
             "WHY DEPTH, NOT FREQUENCY?", font=F_BODY, size=10, color=ACCENT, bold=True)
    add_paragraphs(s, Inches(8.7), Inches(4.75), Inches(4.0), Inches(2.2), [
        ("Number of <think> blocks per turn", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("stays flat — what grows is the", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("content inside each block.", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("", dict(size=6)),
        ("The policy reasons more carefully,", dict(size=11.5, color=INK, font=F_BODY, italic=True)),
        ("not more often.", dict(size=11.5, color=INK, font=F_BODY, italic=True)),
    ], line_spacing=1.20)

    page_footer(s, idx, TOTAL)
    return s


def slide_27_m4_self_correct(idx):
    s = blank_slide(prs)
    page_header(s, "M4 · Self-correction volume", "More hedging and verification moves")

    add_rect(s, Inches(0.6), Inches(2.05), Inches(7.6), Inches(4.9),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.03)
    s.shapes.add_picture("panel_d.png",
                         Inches(0.85), Inches(2.30),
                         width=Inches(7.1), height=Inches(4.4))

    # right narrative
    add_rect(s, Inches(8.4), Inches(2.05), Inches(4.35), Inches(4.9),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.03)
    add_text(s, Inches(8.7), Inches(2.30), Inches(4.0), Inches(0.4),
             "hedge tokens / trajectory", font=F_MONO, size=11, color=ACCENT, bold=True)

    add_text(s, Inches(8.7), Inches(2.85), Inches(4.0), Inches(0.7),
             "1.5  →  5.0", font=F_HEAD, size=40, color=ACCENT_DK, bold=True)
    add_text(s, Inches(8.7), Inches(3.70), Inches(4.0), Inches(0.4),
             '"wait / verify / actually / let me check"', font=F_BODY, size=11.5, color=INK_SOFT, italic=True)

    add_hline(s, Inches(8.7), Inches(4.20), Inches(3.85), color=DIVIDER, weight=1)

    add_text(s, Inches(8.7), Inches(4.35), Inches(4.0), Inches(0.35),
             "HONEST FRAMING", font=F_BODY, size=10, color=ACCENT, bold=True)
    add_paragraphs(s, Inches(8.7), Inches(4.75), Inches(4.0), Inches(2.2), [
        ("Hedges per token barely changes —", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("the trajectory simply gets longer", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("and contains more verification.", dict(size=11, color=INK_SOFT, font=F_BODY)),
        ("", dict(size=6)),
        ("Self-correction is a volume effect,", dict(size=11.5, color=INK, font=F_BODY, italic=True)),
        ("driven by the expanded budget.", dict(size=11.5, color=INK, font=F_BODY, italic=True)),
    ], line_spacing=1.20)

    page_footer(s, idx, TOTAL)
    return s


slide_23_mechanisms_overview(23)
slide_24_m1_tool_choice(24)
slide_25_m2_reallocation(25)
slide_26_m3_think_depth(26)
slide_27_m4_self_correct(27)
prs.save(OUT)
print(f"Now {len(prs.slides)} slides")

# ─── SYNTHESIS + CLOSING ────────────────────────────────────────────────

def slide_28_discarded(idx):
    s = blank_slide(prs)
    page_header(s, "Appendix · Behavior", "Candidates we ruled out — for transparency")

    # Left table: discarded
    add_text(s, Inches(0.6), Inches(2.05), Inches(7.6), Inches(0.4),
             "RULED OUT BY THE DATA", font=F_BODY, size=11, color=ACCENT, bold=True)

    rows = [
        ("Answer-length growth",     "Confounded with task difficulty in S2"),
        ("Tool calls per turn",      "Stable across both stages — no shift"),
        ("Think frequency per turn", "Flat — the depth, not the count, grows"),
        ("Hedge density (per token)","Almost unchanged — only volume grows"),
        ("Vocabulary diversity",     "No monotone trend across checkpoints"),
        ("Error/retry rate",         "Drops in S1, rebounds in S2 — non-monotone"),
    ]
    y = Inches(2.55); rh = Inches(0.55)
    for i, (k, v) in enumerate(rows):
        bg = CREAM_SOFT if i % 2 == 0 else CREAM
        add_rect(s, Inches(0.6), y, Inches(7.6), rh, fill=bg, line=None)
        add_text(s, Inches(0.85), y + Inches(0.13), Inches(2.8), Inches(0.32),
                 k, font=F_BODY, size=11.5, color=INK, bold=True)
        add_text(s, Inches(3.75), y + Inches(0.13), Inches(4.3), Inches(0.32),
                 v, font=F_BODY, size=11, color=INK_SOFT)
        y += rh

    # Right: causal funnel diagram
    add_rect(s, Inches(8.4), Inches(2.05), Inches(4.35), Inches(4.95),
             fill=CREAM_SOFT, line=DIVIDER, radius=0.04)
    add_text(s, Inches(8.7), Inches(2.30), Inches(4.0), Inches(0.4),
             "CAUSAL CHAIN", font=F_BODY, size=11, color=ACCENT, bold=True)

    # 5-step vertical chain
    steps = [
        ("M1", "Tool-choice", "search → browse"),
        ("M2", "Re-allocation", "expand budget"),
        ("·",  "More evidence", "per trajectory"),
        ("M3+M4", "Deeper reason &", "self-correction"),
        ("→", "GAIA +9 pts", "(60 → 69)"),
    ]
    y = Inches(2.85); rh = Inches(0.78)
    for i, (tag, name, sub) in enumerate(steps):
        is_outcome = (i == len(steps) - 1)
        bg = ACCENT if is_outcome else CREAM
        add_rect(s, Inches(8.7), y, Inches(3.75), Inches(0.68),
                 fill=bg, line=DIVIDER if not is_outcome else None, radius=0.05)
        fg = RGBColor(0xFF,0xFF,0xFF) if is_outcome else INK
        sub_fg = RGBColor(0xFF,0xE9,0xDE) if is_outcome else INK_SOFT
        add_text(s, Inches(8.85), y + Inches(0.06), Inches(0.7), Inches(0.6),
                 tag, font=F_HEAD, size=14,
                 color=ACCENT if not is_outcome else fg, bold=True)
        add_text(s, Inches(9.50), y + Inches(0.05), Inches(2.9), Inches(0.32),
                 name, font=F_BODY, size=11.5, color=fg, bold=True)
        add_text(s, Inches(9.50), y + Inches(0.32), Inches(2.9), Inches(0.32),
                 sub, font=F_BODY, size=10, color=sub_fg, italic=True)
        y += rh

    page_footer(s, idx, TOTAL)
    return s


def slide_29_insights(idx):
    s = blank_slide(prs)
    page_header(s, "09 · Takeaways", "Three things to remember")

    items = [
        ("01",
         "Local environments are enough.",
         "A 32 M-page Wikipedia mirror + a lightweight browse tool replaces "
         "$59 K-$243 K of commercial APIs, gives 10×/46× speed-ups, and unlocks "
         "the trajectory volume needed for on-policy RL at scale."),
        ("02",
         "Simple-but-scalable beats clever-but-fragile.",
         "Atomic-capability synthesis, on-policy GRPO, and a 2-stage curriculum — "
         "each individually familiar — combine to lift a 4 B model past 30 B "
         "teachers and most commercial systems."),
        ("03",
         "The gain has a behavioral signature.",
         "Two structural mechanisms (tool-choice shift, trajectory re-allocation) "
         "act first; depth and self-correction emerge as their downstream effect. "
         "Sim2Real is sustained because the behavior, not just the score, transfers."),
    ]
    y = Inches(2.10); rh = Inches(1.55)
    for tag, head, body in items:
        # accent bar on the left
        add_rect(s, Inches(0.6), y, Inches(0.1), Inches(1.4),
                 fill=ACCENT, line=None)
        add_text(s, Inches(0.85), y, Inches(1.4), Inches(0.6),
                 tag, font=F_HEAD, size=32, color=ACCENT, bold=True)
        add_text(s, Inches(2.4), y + Inches(0.05), Inches(10.3), Inches(0.55),
                 head, font=F_HEAD, size=18, color=INK, bold=True)
        add_text(s, Inches(2.4), y + Inches(0.65), Inches(10.3), Inches(0.85),
                 body, font=F_BODY, size=12, color=INK_SOFT)
        y += rh

    page_footer(s, idx, TOTAL)
    return s


def slide_30_thanks(idx):
    s = blank_slide(prs)

    # large centered serif "Thank you"
    add_text(s, Inches(0.6), Inches(1.95), Inches(12.15), Inches(1.7),
             "Thank you", font=F_HEAD, size=88, color=INK, bold=True,
             align=PP_ALIGN.CENTER)

    # subtitle (pushed below descenders of "y")
    add_text(s, Inches(0.6), Inches(3.85), Inches(12.15), Inches(0.55),
             "Questions & discussion welcome.",
             font=F_BODY, size=18, color=INK_SOFT, italic=True,
             align=PP_ALIGN.CENTER)

    # accent divider
    add_hline(s, Inches(5.5), Inches(4.65), Inches(2.3), color=ACCENT, weight=2.5)

    # three resource cards
    res = [
        ("PAPER", "arXiv : 2604.17931"),
        ("CODE",  "github.com/wanli/LiteResearcher"),
        ("MODEL", "hf.co/litereresearcher-4b"),
    ]
    y0 = Inches(5.00); w = Inches(3.85); gap = Inches(0.20)
    total_w = w * 3 + gap * 2
    x0 = (Inches(13.333) - total_w) / 2
    for i, (k, v) in enumerate(res):
        x = x0 + i * (w + gap)
        add_rect(s, x, y0, w, Inches(1.2),
                 fill=CREAM_SOFT, line=DIVIDER, radius=0.04)
        add_text(s, x + Inches(0.3), y0 + Inches(0.22), w - Inches(0.6), Inches(0.35),
                 k, font=F_BODY, size=11, color=ACCENT, bold=True,
                 align=PP_ALIGN.CENTER)
        add_text(s, x + Inches(0.3), y0 + Inches(0.60), w - Inches(0.6), Inches(0.45),
                 v, font=F_MONO, size=12, color=INK, align=PP_ALIGN.CENTER)

    # tiny footer with author + acknowledgement
    add_text(s, Inches(0.6), Inches(6.65), Inches(12.15), Inches(0.4),
             "Wanli Lee  ·  with thanks to all co-authors and reviewers",
             font=F_BODY, size=11, color=INK_SOFT, italic=True,
             align=PP_ALIGN.CENTER)

    return s


slide_28_discarded(28)
slide_29_insights(29)
slide_30_thanks(30)
prs.save(OUT)
print(f"Final deck: {len(prs.slides)} slides at {OUT}")
