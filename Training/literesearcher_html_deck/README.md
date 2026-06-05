# LiteResearcher · HTML Presentation Deck

A 30-slide, WeaveBench-inspired HTML deck for the **LiteResearcher** paper (arXiv 2604.17931). Renders cleanly in any modern browser and exports to a fixed-layout PDF via Playwright.

## Deliverables

| File | Description |
| --- | --- |
| `deck.html` | Single-file HTML deck — all 30 slides as `<section class="slide">` blocks |
| `styles.css` | Palette, typography, layout primitives, print-mode rules |
| `LiteResearcher.pdf` | Final exported PDF, one 1280×720 page per slide |
| `render.py` | Playwright script — renders deck.html → per-slide PNGs + composite PDF |
| `preview/slide-NN.png` | 30 individual high-res slide previews (2× device pixel ratio) |
| `fig1..4_*.png`, `panel_[a-d].png` | Embedded figures |

## Design system

- **Palette** — cream `#F5F1ED` bg · card `#FAF7F2` · ink `#1F1F1F` · accent orange `#D97757` · soft rule `#E8E0D5`
- **Typography**
  - Headlines · `Source Serif Pro` (Georgia fallback)
  - Body · `Inter` (system sans fallback)
  - Numbers / mono · `JetBrains Mono` (Consolas fallback)
- **Slide template** (consistent on every body slide)
  - eyebrow tag · accent rule · serif title · italic subtitle · soft `<hr>`
  - content band: 1136 px wide × ~480 px tall
  - footer: `LITERESEARCHER` left · `NN / 30` right
- **Inspirations** — WeaveBench's dense data-first layouts, numbered M1/M2 pictograms, monospaced ID tags, bold "best row" highlighting in tables; Claude's warm orange palette and serif headlines.

## Deck flow (30 slides)

| # | Section | Slides |
| --- | --- | --- |
| 1 | **Opening** | Cover · Paper info & TL;DR · Agenda |
| 2 | **Problem** | Cost wall · Prior approaches (3-col) · Thesis (Mirror + Isolate) |
| 3 | **Method** | 3-pillar overview · Data pipeline · 5 atomic capabilities · Local env · Curriculum + GRPO |
| 4 | **Experiments** | §-divider · Setup · Main 8-bench table · Cost · Data · On-policy · Multi-stage · RL vs SFT |
| 5 | **Appendix — Behavior** | §-divider · Motivation + setup · Outcome figure · 4-mechanism overview · M1 · M2 · M3 · M4 · Discarded + causal chain |
| 6 | **Closing** | 3 takeaways · Thank-you with resources |

## Rebuilding

```bash
# install once
pip install playwright
playwright install chromium

# regenerate previews + PDF
python3 render.py
# → LiteResearcher.pdf (1280×720 per page)
# → preview/slide-01.png … slide-30.png (2× DPI)
```

To tweak a slide, edit the `<section id="sNN">…</section>` in `deck.html` and re-run `render.py`.

## Editorial / review notes

This deck was reviewed slide-by-slide by an Opus reviewer (claude-opus-4.7). Three issues flagged and fixed:

- **Slide 18** — explicit "§5.3 plateau-averaged" framing on the 64.7 → 68.3 cards so they don't appear to contradict the chart's per-checkpoint pass@1 trace.
- **Slide 16** — disambiguation note ties the ablation's SFT-recipe rows to the vanilla `Qwen3-4B · SFT only` baseline on slide 14.
- **Slide 1** — page indicator `01 / 30` added to keep audience orientation consistent.

A v2 verification pass confirmed all three fixes landed without regressions.

## Key numbers (kept internally consistent)

- 4 B model · GAIA 71.3 · Xbench-DS 78.0 · Frames 83.1 · BrowseComp 32.5
- Local env: 32 M pages · 1 M+ domains · 73.2 M tool calls
- Cost: $0 local vs $59 K – $243 K commercial; latency speed-up 10× (search) / 46× (browse)
- Curriculum: S1 plateau 64.7 % → S2 lift 68.3 % (paper §5.3 averaged values)
- RL vs distillation: Tongyi teacher 70.9 → SFT 55.6 → +RL 71.3 (+15.7 pts)
- On-policy 68.9 % vs off-policy 66.8 %
- Behavior shifts: M1 0.46 → 0.76 · M2 28 → 19 → 45 · M3 387 → 626 chars · M4 1.5 → 5.0 hedges

## License

Internal — for paper presentation use. Figures © the authors.
