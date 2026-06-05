# LiteResearcher — Behavior Analysis (Sim→Real Mechanisms)

Supplementary analysis package for the **LiteResearcher** paper (arXiv:2604.17931).
Investigates the question: *why does Stage-2 sim→real RL produce real improvements on
GAIA / Xbench-DS / Frames / BrowseComp* — by tracing model **behavioral changes** from
Stage-1 start to Stage-2 end on the actual rollout trajectories.

---

## Layout

```
literesearcher_behavior_analysis/
├── BEHAVIOR_MECHANISMS.md         # ★ headline analysis — 4 mechanisms (EN)
├── BEHAVIOR_MECHANISMS_zh.md      # ★ headline analysis — 4 mechanisms (ZH)
├── BEHAVIOR_EVOLUTION.md          # longer evolution narrative (EN)
├── BEHAVIOR_EVOLUTION_zh.md       # longer evolution narrative (ZH)
├── figures/                       # 4 paper figures + 7 exploratory + continuous-trend
│   ├── fig1_accuracy_sim2real.png/.pdf
│   ├── fig2_mechanisms.png/.pdf         # 4-mechanism composite
│   ├── fig3_normalized_rates.png/.pdf   # normalized per-turn rates
│   ├── fig4_appendix_12panel.png/.pdf   # full per-mechanism panels
│   ├── fig_continuous_*                  # continuous training curves
│   └── fig{1..6}_*.png                   # earlier exploratory passes
├── scripts/
│   ├── extract_behaviors.py       # parse rollout jsonl → per-step behavior signals
│   ├── make_figs.py               # exploratory figures
│   ├── make_continuous.py         # continuous-trend smoothing
│   ├── make_paper_figures.py      # final 4 figures for paper
│   ├── mine_examples.py           # qualitative trajectory mining
│   └── run_all.py                 # end-to-end pipeline
├── data/
│   ├── behavior_timeline.json     # full per-checkpoint metric timeline
│   └── behavior_timeline_paperpack.json   # slim version used by make_paper_figures
└── trajectory_examples/
    └── trajectory_examples.json   # qualitative before/after rollout pairs
```

---

## The Four Mechanisms (paper §5 supplement)

Reading **`BEHAVIOR_MECHANISMS.md`** (EN) or **`BEHAVIOR_MECHANISMS_zh.md`** (ZH) is
the recommended entry point. Key findings:

| ID | Mechanism                                  | S1-start → S2-end                          |
|----|--------------------------------------------|--------------------------------------------|
| M1 | Higher tool-use per turn                   | 0.46 → 0.76 calls/turn                     |
| M2 | Smarter retry & self-correction            | retry rate 28% → 19% (down), recovered-pass rate up |
| M3 | Longer, more structured deliberation       | avg `<think>` block 387 → 626 chars         |
| M4 | More calibrated hedging / uncertainty      | hedge markers per answer 1.5 → 5.0          |

Together these explain the 64.7 % → 68.3 % plateau-averaged sim→real lift reported in §5.3.

---

## Reproduce

```bash
# 1. extract behavior signals from raw rollout jsonl
python scripts/extract_behaviors.py \
    --rollout_root /path/to/verl/rollout_trajectory \
    --bench_root   /path/to/DeepResearch/bench_results

# 2. rebuild the 4 paper figures
python scripts/make_paper_figures.py
```

Trajectory sources used (paths are local to author's env):
- Stage-1 RL rollouts: `verl/rollout_trajectory/qwen3_deepresearch_tis_rl_onpolicy_bs128_local_rag_only_token-mean-seq-mean-temp_0.7_length_32k_nokl`
- Stage-2 RL rollouts: `verl/rollout_trajectory/qwen3_deepresearch_tis_rl_stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k`
- Stage-1 online evals: `DeepResearch/bench_results/qwen3-4B-RL/onpolicy_bs128_local_rag_only_token-mean-seq-mean-temp_0.7_length_32k_nokl`
- Stage-2 online evals: `DeepResearch/bench_results/qwen3-4B-RL/stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k`

---

## Related artifacts in this repo

- `literesearcher_html_deck/` — full HTML presentation (30 slides, Claude × WeaveBench style)
- `literesearcher_presentation/` — earlier .pptx version of the same deck

The 4 mechanism figures here (`fig1`–`fig4` paper variants) are the same files used as
slides 22-27 of the HTML deck.

---

*Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>*
