# LiteResearcher · Presentation Deck

A 30-slide, Claude-style presentation that walks through the full **LiteResearcher** pipeline (arXiv:2604.17931) — problem framing, three-pillar method, experiments, and the behavior-analysis appendix that explains *why* the policy improves during multi-stage RL.

## Deliverables

| File | Description |
| --- | --- |
| `LiteResearcher_Presentation.pptx` | 16:9, 30 slides — edit in PowerPoint / Keynote / LibreOffice |
| `LiteResearcher_Presentation.pdf`  | Rendered preview (1.3 MB) |
| `build_ppt.py`                     | Master builder (~30 slide functions) |
| `theme.py`                         | Palette, fonts, header/footer, helpers |
| `diagrams.py`                      | Matplotlib custom diagrams generator |
| `panel_[a-d].png`                  | Slices of `fig2_mechanisms.png` for per-mechanism slides |
| `fig*.png`, `diag_*.png`           | All embedded figures |

## Deck outline

| # | Section | Slides |
| --- | --- | --- |
| 1 | **Opening**     | Title · Authors/resources · Agenda |
| 2 | **Problem**     | Cost wall · Prior approaches · Mirror + Isolate thesis |
| 3 | **Method**      | Architecture · Pillar 1 atomic data · Atomic capabilities · Pillar 2 local env · Pillar 3 curriculum |
| 4 | **Experiments** | Setup · Main 8-benchmark table · Cost ablation · Synthetic-data ablation · On-policy vs off-policy · Multi-stage · RL vs SFT |
| 5 | **Appendix · Behavior analysis** | Motivation · 41-ckpt setup · Outcome figure · 4-mechanism overview · M1 tool-choice · M2 trajectory re-allocation · M3 per-think depth · M4 self-correction · Discarded candidates + causal chain |
| 6 | **Closing**     | Three takeaways · Thank you (paper/code/model links) |

## Visual system

- **Background**: cream `#F5F1ED`
- **Card surface**: `#FAF7F2`
- **Accent**: Claude orange `#D97757`
- **Ink**: `#1F1F1F` primary, `#6B6356` secondary
- **Fonts**: Georgia (headlines) · Calibri (body) · Consolas (mono)
- **Grid**: 13.333" × 7.5" with 0.6" side margins, header band 0.55"–1.78", content 2.05"–6.95", footer 7.05"

## Rebuilding the deck

```bash
# regenerate the four custom matplotlib diagrams (optional)
python3 diagrams.py

# rebuild the .pptx
python3 build_ppt.py
# → LiteResearcher_Presentation.pptx

# preview as PDF (requires libreoffice + poppler)
soffice --headless --convert-to pdf LiteResearcher_Presentation.pptx
pdftoppm -png -r 90 LiteResearcher_Presentation.pdf preview/slide
```

## Key numbers used (kept consistent across slides)

- **4 B model**, GAIA 71.3 · Xbench-DS 78.0 · Frames 83.1 · BrowseComp 32.5
- **Local env**: 32 M pages · 1 M+ domains · 73.2 M tool calls (45.8 M search + 27.4 M browse)
- **Cost**: $0 local vs $59 K–$243 K commercial; **latency**: ≈ 0.15 s search (10× online), 0.17 s browse (46× Jina)
- **Curriculum**: S1 plateau 64.7 % → S2 lift 68.3 % (+3.6 pts)
- **RL vs distillation**: Tongyi teacher 70.9 → SFT 55.6 → +RL 71.3 (+15.7 pts)
- **On-policy vs off-policy**: 68.9 % vs 66.8 %
- **Behavior shifts** (S1 start → S2 end):
  - M1 browse ratio 0.46 → 0.76
  - M2 turns 28 → 19 → 45 (U-shape)
  - M3 chars per `<think>` 387 → 626
  - M4 hedge tokens / trajectory 1.5 → 5.0

## License

Internal — for paper presentation use. Figures © the authors.
