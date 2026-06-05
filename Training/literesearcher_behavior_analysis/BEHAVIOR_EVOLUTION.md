# Behavioral Evolution Across Two-Stage RL Training
*A trajectory-level analysis of how a Qwen3-4B deep-research agent transforms under pure on-policy GRPO*

> 🇨🇳 **中文版**: [`BEHAVIOR_EVOLUTION_zh.md`](BEHAVIOR_EVOLUTION_zh.md)

> **Paper-ready section · Wanli Lee · 2026-06-03**
>
> **Data sources**
> - Sim rollouts (training-time samples, 128 prompts × 8 GRPO rollouts per step)
>   - S1: `verl/rollout_trajectory/qwen3_deepresearch_tis_rl_onpolicy_bs128_local_rag_only_token-mean-seq-mean-temp_0.7_length_32k_nokl/`
>   - S2: `verl/rollout_trajectory/qwen3_deepresearch_tis_rl_stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k/`
> - Online benchmark (GAIA pass@4, 412 records per evaluated step)
>   - S1: `DeepResearch/bench_results/qwen3-4B-RL/onpolicy_bs128_local_rag_only_token-mean-seq-mean-temp_0.7_length_32k_nokl/`
>   - S2: `DeepResearch/bench_results/qwen3-4B-RL/stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k/`
> - Coverage: 32 checkpoints (S1=19, S2=13); 300 rollouts per checkpoint × 30 behavioral metrics.
> - Six figures `fig1` – `fig6` and 10 representative trajectory transcripts (`trajectory_examples.json`) accompany this report.

---

## 0 · One-paragraph paper TL;DR

Pure on-policy GRPO without KL or entropy regularization induces a **non-monotonic behavioral trajectory** that cleanly decomposes into five regimes: (A) **base-model exploration** with high token / turn / search budget; (B) **compression** to a short-and-correct policy that halves length, turns and tool calls while accuracy climbs; (C) a **silent decay phase** in which sim reward keeps rising even as critical priors (e.g. exact-phrase quoted queries) are quietly extinguished; (D) a **terminal collapse** triggered by the joint failure of two reward-signal blind spots — the 32 k token cap and the `<answer>` tag requirement — manifesting as a single-step jump in `no_extraction` rate from 1 % to 38 %; (E) a **second-stage rescue** that, by warm-starting from the *last healthy* checkpoint (ckpt 220), extending context to 48 k and feeding a harder data mix, *reverses the regime* into long-reasoning, multi-turn, self-reflective behavior — *the* regime that actually generalizes to GAIA pass@1 (55 → 68 %). This document quantifies every regime with measured metrics and provides illustrative rollouts for each.

---

## 1 · Five-regime taxonomy of behavioral evolution

| Regime | Stage / steps | Defining behavior | Sim reward | GAIA acc | Trigger |
|---|---|---|---|---|---|
| **A. Base exploration** | S1 0–80 | 17 k tok, 28 turns, 7.8 search/traj, 16.8 queries, 46 % quoted | 0.50 | n/a | Inherited from Qwen3-4B-Thinking-2507 |
| **B. Compression** | S1 80–250 | 10 k tok, 19 turns, 4.0 search, 9 queries, 16 % quoted | 0.66 → 0.70 | 56 → 59 % | GRPO advantage on short-and-correct samples |
| **C. Silent decay** | S1 250–490 | length / turns oscillate, but quoted-query falls to 8 % and `<think>` to 9 blocks | 0.66 → 0.76 | 62 → 63 % | Drift along the reward-flat manifold |
| **D. Terminal collapse** | S1 490–540 | `no_extraction` 1 % → 38 %, turns 21 → 10, search 3 → 1.3, `<think>` 9 → 3.7 | 0.72 → 0.51 | 63 → 51 % | Joint failure: 32 k truncation + `<answer>` requirement gate the reward |
| **E. Second-stage rescue** | S2 0–500 (warm from S1 ckpt 220) | 15–21 k tok, 32–45 turns, hedge×3 vs S1, 16 think blocks, mid-quoted (12 %) | 0.41 → 0.85 | 60 → 68 % | 48 k cap + hard data mix re-opens long-reasoning manifold |

The remainder of this document expands each regime with concrete trajectory evidence and the figures that justify the boundaries.

---

## 2 · Setup & metrics

### 2.1 Two-stage training pipeline
```
Qwen3-4B-Thinking-2507  ── S1 GRPO (~787 steps) ──► (eventual collapse)
                              │
                              └─► ckpt at step 220 ────► S2 GRPO (~570 steps) ──► final model
```

- **S1** config: `loss_agg_mode = seq-mean-token-mean`, `train_batch = mini_batch = 128` (pure on-policy ⇒ `ratio ≡ 1`, `pg_clipfrac ≡ 0` for all 787 steps), no KL, entropy_coeff = 0, lr = 1e-6 constant, temp 0.7, `max_response_length = 32 k`, `max_assistant_turns = 40`, reward = LLM-judge {0,1} with forced 0 if `extract_solution` fails (no closing `</answer>`).
- **S2** config: warm-start from S1 ckpt 220, `max_response_length = 48 k`, temp 1.0, expanded data (wiki 16 k-32 k + mqa_subgraph6 + science) for harder multi-hop questions.

### 2.2 Behavioral metric set (30+ scalars per checkpoint)
Per trajectory we parse the model output for `<tool_call>{…}</tool_call>` JSON blocks, `<think>…</think>` spans, and `<answer>…</answer>` tags. From these we derive:

- **Output budget**: `output_tokens`, `total_turns`, `truncated_frac` (≥ 31.5 k tokens), `no_extraction_frac` (no `<answer>` tag)
- **Tool usage**: per-tool call counts (search / visit / python), total queries, queries-per-search-call (`search` supports a list of queries), average query length, fraction of queries containing `"…"` exact-phrase quotes
- **Reasoning style**: number and total chars of `<think>` blocks, fraction of Chinese characters in think text, hedge-token count (`wait`, `actually`, `let me reconsider`, 重新, 等等, …)
- **Outcome**: `score`, `correct`, `method` (`llm_judge` / `no_extraction`)
- **Online**: cross-referenced GAIA `summary.json` for pass@1 / pass@4 + the same 30 metrics computed from per-task `result_*.json` records.

Source code: `extract_behaviors.py`, `run_all.py`, `mine_examples.py`, `make_figs.py`. Raw timeline data: `behavior_timeline.json`. Concrete trajectories: `trajectory_examples.json`.

---

## 3 · Regime A — Base Exploration  (S1 steps 0–80)

The base model — Qwen3-4B-Thinking-2507 — is already an "agent" in the sense that it produces `<think>` blocks and `<tool_call>` JSONs. But it has no calibration on the cost of search. Step 1 rollouts show the model **brute-forcing the question**: it fires 25 tool calls and 45 queries on a single GAIA-style problem, hits the 32 k token cap, and times out without ever emitting `<answer>`.

**Rollout fingerprint at step 1**

| Field | Value |
|---|---|
| Question | *"Which airline loyalty program, known for eliminating mileage expiration starting in 2011, absorbed a major U.S. carrier's frequent flyer program on October 1, 2009…"* |
| Output tokens | **32 768** (truncated) |
| Total turns | **51** |
| Tool calls | 25 (search + visit interleaved) |
| Total queries | 45 |
| First 3 queries | `"\"eliminating mileage expiration\" 2011 airline loyalty program"`, `"\"mileage expiration\" eliminated 2011 airline"`, `"\"October 1, 2009\" \"absorbed\" \"frequent flyer program\""` |
| Final `<answer>` | **(none — `no_extraction`, reward = 0)** |
| Last `<think>` | *"Search results not helpful. Let's search for 'Diamond Medallion' and 'private jet card'."* |

This single trajectory contains every property of Regime A: aggressive exact-phrase quoting (a real strength of the base model), repeated reformulation, and a complete inability to *commit* to an answer under budget pressure. Quoted-query fraction across all rollouts in this regime: **46 %** (the highest of any checkpoint in the entire two-stage run).

The aggregate metric reading on Regime A:

| Metric | Step 1 | Step 80 (extrapolated boundary) |
|---|---|---|
| Mean output tokens | 17 398 | ≈ 11 000 |
| Mean turns | 28.3 | ≈ 19 |
| Mean tool calls | 7.8 search + 0.9 visit | 4.0 + 0.5 |
| Mean queries | 16.8 | 9 |
| Quoted-query fraction | **46.0 %** | 24 % |
| Truncation rate | 23 % | 4 % |
| Sim reward | 0.497 | 0.59 |

The reward signal in this regime is dominated by a single failure mode (truncation → no answer tag → score 0). GRPO advantages therefore overwhelmingly favor trajectories that **finish early enough to emit `<answer>`**. This sets the stage for Regime B.

---

## 4 · Regime B — Compression  (S1 steps 80–250)

This is the regime in which the *paper's headline-quality model is born*: the checkpoint at step **220** that S2 will warm-start from. Empirically, between step 80 and step 250 the policy halves its token budget while accuracy rises by 17 pp.

**Mean trajectory shape at step 220 (representative correct rollout)**

| Field | Value |
|---|---|
| Question | *"Gamma Aquariids Meteor Shower 所属的星座是哪一个？"* |
| Output tokens | **1 957** |
| Total turns | 6 |
| Tool calls | `[search, visit]` (1 each) |
| Queries | `["Gamma Aquariids Meteor Shower 星座", "…属于哪个星座", "…constellation"]` (3 queries in one search call) |
| First `<think>` | *"搜索结果包含一个指向 'Gamma Aquariids Meteor Shower' 的链接。我们打开它。"* |
| Last `<think>` | *"该页面显示星座是水瓶座 (Aquarius)。所以答案是：水瓶座 (Aquarius)。"* |
| `<answer>` | `Aquarius` ✓ |

This is the textbook **search → visit → answer** pattern. Note three RL-learned properties:
1. **Multi-query batching**: the model issues 3 queries inside a *single* `search` call (`queries_per_search_call ≈ 3.06`). The base model also does this but less consistently.
2. **Bilingual querying**: it intentionally mixes Chinese and English queries to cover both index spaces — note the Chinese query, the bilingual restatement, and the English fallback.
3. **One-shot commitment**: a single `visit` followed by a single `<answer>`, with no second-guessing.

Aggregate metrics across Regime B:

| Metric | Step 100 | Step 200 | Step 220 (ckpt) | Step 250 |
|---|---|---|---|---|
| Sim acc | 0.593 | 0.643 | **0.700** | 0.653 |
| GAIA pass@1 | — | 0.561 | **0.592** | — |
| Mean tokens | 10 983 | 9 607 | 10 805 | 10 838 |
| Mean turns | 18.8 | 17.2 | 19.2 | 19.8 |
| Mean search calls | 4.25 | 3.49 | 3.98 | 3.94 |
| Queries / search | 2.90 | 3.13 | **3.06** | 3.05 |
| Quoted-query frac | 24.3 % | 16.3 % | **15.7 %** | 12.7 % |
| Truncation | 4.3 % | 0.7 % | 2.3 % | 3.7 % |
| no_extraction | 4.3 % | 0.7 % | **1.7 %** | 4.7 % |

Two early warning signs are already visible at step 220 but invisible from reward alone:
- Quoted-query fraction has dropped from 46 % to 16 % — a 65 % erosion of the base model's exact-match prior.
- `no_extraction` is at a remarkable low of 1.7 %, but this is *because* the model has learned to finish early, not because it has learned to commit to better answers. This will matter in Regime D.

**Why ckpt 220 is the correct warm-start point for S2** (a paper-grade claim worth making): step 220 is the *first* step at which (sim_acc ≥ 0.70) AND (no_extraction ≤ 2 %) AND (Chinese-query-handling intact) hold simultaneously. Earlier checkpoints lack accuracy; later checkpoints have already begun discarding behavioral priors that S2 needs to relearn from scratch.


---

## 5 · Regime C — Silent Decay  (S1 steps 250–490)

This is the most subtle and the most pedagogically important regime. Sim reward keeps climbing (0.66 → 0.76) and GAIA pass@1 follows it gently (62 % → 63 %). On every dashboard a practitioner watches, S1 looks healthy. Internally, however, the policy is **silently extinguishing behavioral priors that the reward function cannot see**:

| Metric | Step 250 | Step 350 | Step 450 | Step 480 |
|---|---|---|---|---|
| Sim acc | 0.653 | 0.720 | **0.760** | 0.720 |
| GAIA pass@1 | — | — | — | 0.629 |
| Mean tokens | 10 838 | 11 621 | 11 477 | 10 301 |
| Mean turns | 19.8 | 23.7 | 23.3 | 21.1 |
| Quoted-query frac | 12.7 % | 16.0 % | **7.6 %** | 9.1 % |
| Truncation | 3.7 % | 3.0 % | 1.7 % | **0.7 %** |
| no_extraction | 4.7 % | 5.0 % | 1.7 % | **1.0 %** |

The pattern is *learning to dodge the reward's blind spots* rather than learning to do the task. Three observations:

1. **Truncation rate drops to 0.7 %** by step 480 — the model has *learned* that hitting 32 k means a guaranteed 0, and reflexively closes its trace before getting there. This is a useful skill in isolation, but **it is the same skill that will cause the collapse in Regime D**: once the model has internalized "short is safe", any push toward shorter outputs is rewarded.
2. **Quoted-query fraction crashes to 7.6 %** at step 450. The base-model habit of grabbing exact phrases from the question and quoting them in the search query — a strong prior for entity-heavy retrieval — is being unlearned, because in many easy training questions the unquoted query also succeeds.
3. **Reward keeps climbing** because all of these silently-eroded priors don't *directly* hurt aggregate accuracy on the relatively easy training distribution (`rag_direct`, `local_rag_only`). They will hurt on GAIA — but step 480's GAIA result (0.629) isn't yet alarming.

**Concrete trajectory at step 480** (same question as step 220 above, for direct comparison):

| Field | Step 220 | Step 480 |
|---|---|---|
| Output tokens | 1 957 | 1 915 (essentially identical) |
| Total turns | 6 | 6 |
| Search calls | 1 | 1 |
| Tool sequence | `[search, visit]` | `[search, visit]` |
| First `<think>` | *"…我们打开它。"* | *"…我们打开 universeguide.com 上的页面。"* |
| `<answer>` | `Aquarius` ✓ | `Aquarius` ✓ |

On this easy question the two checkpoints are behaviorally indistinguishable. The decay is invisible on individual easy examples; it shows up only as a population-level shift in the distribution of behavior on *hard* examples — which is exactly what GAIA exposes.

**Why this regime ends in collapse, not in a plateau**: the loss aggregation `seq-mean-token-mean` (∑ᵢ (1/Nᵢ) ∑ₜ loss_{i,t}, then mean over batch) gives each *trajectory* equal weight regardless of length. Combined with no KL and no entropy floor, this means there is no force opposing further shortening. The policy slides along a reward-flat manifold toward shorter and shorter trajectories until one shortening step pushes too many trajectories into "single-turn-guess" territory.

---

## 6 · Regime D — Terminal Collapse  (S1 steps 490–540)

Between step 480 and step 510 the policy crosses a phase boundary. Every metric on the *output-budget* axis collapses simultaneously:

| Metric | Step 480 | Step 510 | Δ |
|---|---|---|---|
| Sim acc | 0.720 | 0.510 | **−21 pp** |
| Mean output tokens | 10 301 | 9 301 | −10 % (deceptive — see below) |
| Mean turns | **21.1** | **10.2** | **−52 %** |
| Mean search calls | 3.08 | 1.29 | −58 % |
| `<think>` blocks | 9.5 | 3.7 | −61 % |
| Truncation rate | 0.7 % | 6.3 % | ×9 |
| **no_extraction** | **1.0 %** | **38.3 %** | **×38** |
| Hedge tokens | 0.9 | 1.4 | (noise) |

**The smoking gun is `no_extraction`**. In a single training step the fraction of rollouts that emit no `<answer>` tag jumps from 1 % to 38 %. Reading `verl/utils/reward_score/llm_judge_async.py` confirms that any trajectory failing `extract_solution` is assigned `score = 0.0` with `method = "no_extraction"`, *indistinguishably from a wrong answer*. GRPO therefore receives a reward signal that says "40 % of your trajectories are wrong" with no clue about whether they're wrong-from-content or wrong-from-format.

**Representative collapsing trajectory (step 510)**:

| Field | Value |
|---|---|
| Question | *"斯洛博达耶希瓦在1929年阿拉伯暴乱后迁往耶路撒冷之前，其原名是什么，以及迁往耶路撒冷后更名为什么？"* |
| Output tokens | **32 768** (truncated) |
| Turns | 14 |
| Tool calls | 6 (`[search, visit, visit, search, visit, visit]`) |
| Last `<think>` (verbatim, last 80 chars) | *"…我们查一下希伯伦耶希瓦维基百科 (Hebron Yeshiva Wikipedia)。"* |
| `<answer>` | **(none — never emitted)** |
| Reward | 0 (no_extraction) |

Compare this to Regime A's step-1 trajectory: same failure mode (run out of budget before committing). The difference is that in Regime A the model was *exploring*; in Regime D the model has *stopped exploring* (turns 28 → 10) but is also no longer *committing* (because the policy has been pushed past the boundary where it knows how to wrap up a multi-hop question). It is now stuck between two bad attractors.

Across the collapse window:
- Quoted-query fraction crashes to 1 % (step 510) then **0.1 %** by step 750.
- `<think>` blocks drop to 3.7 (step 510); each `<think>` is now a single sentence rather than a planning block.
- Search-call count crashes to 1.3 per trajectory: the model has stopped iterating.

**False recovery (steps 540–787)**: sim reward gradually climbs back to 0.80 by step 660, but GAIA pass@1 stays pinned at 0.51–0.58. The recovery is real on the training distribution (the model relearns to emit *some* `<answer>` tag, even if it's a guess), but it is not real on GAIA — see Section 8 for the sim-real gap chart.


---

## 7 · Regime E — Second-Stage Rescue  (S2 steps 0–500, warm from S1 ckpt 220)

S2 is not a continuation of S1; it is a *regime reset*. The warm-start at S1 ckpt 220 keeps the behavioral priors that survived Regime B; the new context window (32 k → 48 k) eliminates the reward blind spot that drove Regime C decay; the harder data mix forces multi-hop reasoning that *cannot* be shortcut. Together these three changes flip the behavioral attractor.

**Macro signature** of S2 across the full run:

| Metric | S2 step 1 | S2 step 100 | S2 step 300 | S2 step 500 | S2 step 570 |
|---|---|---|---|---|---|
| Sim acc | 0.410 | 0.610 | 0.717 | **0.847** | 0.727 |
| GAIA pass@1 | — | 0.600 | 0.650 | 0.663 | **0.682** |
| GAIA pass@4 | — | 80.6 | 83.5 | 79.6 | 85.4 |
| Mean output tokens | 18 497 | 15 731 | 16 379 | 15 771 | **20 933** |
| Mean turns | **36.8** | 35.9 | 38.8 | 32.9 | **44.6** |
| Mean `<think>` blocks | 17.4 | 16.9 | 18.4 | 15.5 | **21.3** |
| Mean hedge tokens | 2.6 | 3.9 | 4.4 | 2.9 | **5.0** |
| Quoted-query frac | 19.3 % | 15.6 % | 16.9 % | 12.5 % | 12.1 % |
| no_extraction | 9.7 % | 6.7 % | 4.3 % | **2.3 %** | 8.0 % |

Compare against S1 *at its very best* (step 480): mean turns 21.1, `<think>` 9.5, hedge 0.9. **S2 produces 2 × the turns, 1.7 × the `<think>` blocks, and 3 – 5 × the hedge tokens of the best S1 checkpoint.** And it does so while monotonically improving GAIA pass@1.

**Long-reasoning trajectory at step 400** (correct, hard multi-hop):

| Field | Value |
|---|---|
| Question | *"香川县选区在2023年1月31日时，现任参议员中属于2022年当选类别且为第三届任期的是谁？"* |
| Output tokens | **44 781** |
| Total turns | **114** |
| Tool calls | 56 (mostly `visit`, walking many Wikipedia pages) |
| Total queries | 21 |
| First `<think>` | *"我们有多个相关的维基百科页面。让我们打开关于'香川县选区'的维基百科页面。"* |
| Last `<think>` | *"现在我们有了答案。因此，答案是：Yoshihiko Isozaki。"* |
| `<answer>` | `Yoshihiko Isozaki` ✓ |

This trajectory **cannot exist** in S1 — 114 turns × ~400 tokens each would have hit the 32 k cap by turn 80. S2's 48 k cap (and the model's learned willingness to use it) creates a new behavioral mode entirely.

**Long-reasoning trajectory at step 570** (hardest end of the difficulty curve):

| Field | Value |
|---|---|
| Question | *"What is the name of the exhibition expected to travel to several art museums in California starting in 2024 and accompanied by the book Julia Morgan: The Road to San Simeon?"* |
| Output tokens | **47 409** (near the 48 k cap) |
| Turns | **120** |
| Tool calls | 59 |
| Queries | 27 |
| Last `<think>` | *"Thus, difficult. Nevertheless, I'll answer: The exhibition is the forthcoming traveling museum exhibition…"* |
| `<answer>` | (long descriptive answer — judged correct) ✓ |

Two important behavioral properties visible here:
1. **The model knows when it isn't sure**, and frequently emits hedges like *"Thus, difficult"*, *"Nevertheless, I'll answer"*. S1 by step 450+ had hedge_count = 1.2; S2 step 570 has 5.0.
2. **The model is willing to commit a long descriptive answer** rather than producing nothing. This is the inverse of the S1 collapse mode (which traded committal for shortness).

### 7.1 Why S2 succeeds where S1 failed — three mechanisms

1. **Context window mismatch resolution**. The training reward is computed against generations bounded by `max_response_length`. In S1 the bound was the same 32 k that capped GAIA evaluations, so the model's reward landscape was *systematically* warped by truncation. In S2 the bound is 48 k, well beyond what any GAIA task requires, so truncation drops below 25 % and the policy gradient sees the full content quality dimension.
2. **Difficulty floor on the data**. S2 mixes in `wiki_16k-32k`, `mqa_subgraph6`, and science data — questions where a one-shot search-and-answer cannot win. The reward-flat manifold that S1 slid along (Regime C) does not exist in S2 because shorter trajectories are *strictly worse* on the new data distribution. The compression force is removed at the data level.
3. **Warm-start point selection**. Starting from ckpt 220 (last healthy) rather than ckpt 480 (visibly "best" by reward but already deep in Regime C decay) means S2 inherits a model that still emits 16 % quoted queries, 8.6 `<think>` blocks, and 19 turns per trajectory. These are the substrates S2 amplifies. From ckpt 480 the same training would almost certainly *not* have rebuilt the lost behavioral diversity.


---

## 8 · Figures

All figures share two visual conventions: a dashed gray vertical line at S1 step **220** (the ckpt used to start S2) and an orange band across S1 steps **490–540** (the collapse window).

### Figure 1 — Accuracy and response budget across stages

![fig1](fig1_acc_length.png)

- (a) Sim train accuracy: S1 peaks at step 450 (0.76), crashes to 0.51 at step 510. S2 peaks at step 500 (0.85) with no collapse.
- (b) GAIA pass@1: S1 peaks step 480 (0.63), falls to 0.51 (step 600). S2 reaches 0.68 at steps 400 & 570.
- (c)(d) Response length: S1 falls from 17 k to 7 k tokens; S2 stays at 13–21 k throughout (real reasoning).
- (e)(f) Turns: S1 27 → 10, S2 stays at 32–45 — the long-reasoning regime.

### Figure 2 — Tool usage and search-query shape

![fig2](fig2_tools.png)

- (a) Search calls: S1 7.8 → 1.3 (collapse), S2 9.9 → 3.8 (healthy).
- (b) Visit calls: S1 sets visit ≈ 0 by collapse; S2 maintains visit throughout (necessary for hard multi-hop).
- (d) Queries per search call: S1 plateaus at 3.0 (healthy) then jumps to 4.7 in collapse (one big query dump per call — symptomatic of "fire-and-forget"); S2 stable at 2.5–3.6.
- (f) Quoted-query fraction — **the single most diagnostic metric**: S1 46 % → 7.6 % → **0.1 %** (extinction); S2 11–19 % (preserved).

### Figure 3 — Truncation & answer-format failures

![fig3](fig3_truncation.png)

- (a) S1 sim truncation falls to 0.7 % (step 480 — model learned to dodge), then recovers slightly. S2 sim truncation 11–25 % (model uses the 48 k headroom).
- (b) **S1 sim `no_extraction` jumps from 1.0 % to 38.3 % in a single step (480 → 510)**. S2 maintains 2–10 %.
- (c)(d) On GAIA inference (no truncation cap during eval), both stages have stable answer-tag rates — the issue is *training-time reward distortion*, not the model's intrinsic answer-format ability.

### Figure 4 — Reasoning style: thinking, language, self-reflection

![fig4](fig4_thinking.png)

- (a)(b) `<think>` blocks per trajectory: S1 13 → 9 → 4 (collapse) → 6 (recovery). S2 15–21.
- (c) Chinese-character fraction: stable 3.5–5.5 % in both stages — the base model's bilingual habit is preserved (and useful for retrieval coverage across languages).
- (d) Hedge tokens: S1 0.9 (best); S2 2.8–5.0. The S2 model openly entertains alternatives, *which is precisely the substrate for multi-hop self-correction*.

### Figure 5 — Sim-to-Real consistency

![fig5](fig5_sim2real.png)

- (a) Sim (rollout) vs Real (GAIA pass@1) — S1 lines diverge sharply after step 510; S2 lines stay paired.
- (b) Sim − Real gap: S1 normal regime ≈ 0.08–0.13; **post-collapse 0.17–0.27** (reward measures sim-mode overfitting); S2 0.05–0.18 throughout.

> **This is the quantitative claim of the paper's "sim-to-real" contribution**: in Stage 2 the training reward remains a *faithful predictor* of GAIA pass@1, whereas Stage 1 after step 500 loses this property and the training signal becomes anti-correlated with real-world performance.

### Figure 6 — GAIA pass@1 / pass@4

![fig6](fig6_passk.png)

- S1 pass@4 peaks 84.5 at step 220–480; S2 pass@4 peaks **86.4 at step 60** (just 60 steps after warm-start).
- Pass@4 − pass@1 gap: S1 ≈ 22 pp, S2 ≈ 15 pp — S2 is more *certain*: rollouts converge to the right answer rather than scattering.


---

## 9 · Complete metric table (rollout sample, 300 trajectories per checkpoint)

| stage | step | sim_acc | bench_acc | resp_tok | turns | n_search | queries | qPer | quoted% | trunc% | noAns% | thinks | hedge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 1   | 0.497 |  —    | 17398 | 28.3 | 7.77 | 16.78 | 2.70 | 46.0 | 23.0 | 23.0 | 13.0 | 2.6 |
| S1 | 30  | 0.530 |  —    | 15473 | 25.7 | 6.90 | 14.68 | 2.75 | 40.7 | 15.7 | 15.0 | 11.8 | 2.5 |
| S1 | 60  | 0.517 |  —    | 12549 | 20.4 | 5.16 | 10.73 | 2.74 | 35.0 |  9.0 |  8.7 |  9.2 | 2.2 |
| S1 | 100 | 0.593 |  —    | 10983 | 18.8 | 4.25 |  9.46 | 2.90 | 24.3 |  4.3 |  4.3 |  8.4 | 1.5 |
| S1 | 150 | 0.663 |  —    | 10782 | 19.8 | 4.67 |  9.08 | 2.74 | 22.0 |  6.0 |  6.3 |  8.9 | 2.1 |
| S1 | 200 | 0.643 | 0.561 |  9607 | 17.2 | 3.49 |  8.44 | 3.13 | 16.3 |  0.7 |  0.7 |  7.6 | 1.8 |
| S1 | **220** | **0.700** | **0.592** | 10805 | 19.2 | 3.98 | 9.61 | 3.06 | 15.7 | 2.3 | 1.7 | 8.6 | 1.5 |
| S1 | 250 | 0.653 |  —    | 10838 | 19.8 | 3.94 |  9.07 | 3.05 | 12.7 |  3.7 |  4.7 |  8.9 | 1.0 |
| S1 | 300 | 0.627 | 0.624 | 12527 | 25.6 | 5.08 |  9.08 | 2.26 | 19.3 |  3.3 |  3.0 | 11.8 | 1.4 |
| S1 | 350 | 0.720 |  —    | 11621 | 23.7 | 5.02 |  8.88 | 2.26 | 16.0 |  3.0 |  5.0 | 10.8 | 2.8 |
| S1 | 400 | 0.707 |  —    | 12191 | 25.0 | 4.78 |  9.23 | 2.49 |  9.2 |  3.0 |  3.7 | 11.5 | 1.4 |
| S1 | 450 | 0.760 |  —    | 11477 | 23.3 | 4.11 |  8.71 | 2.68 |  7.6 |  1.7 |  1.7 | 10.7 | 1.2 |
| S1 | 480 | 0.720 | 0.629 | 10301 | 21.1 | 3.08 |  7.36 | 2.94 |  9.1 |  0.7 |  1.0 |  9.5 | 0.9 |
| S1 | **510** | **0.510** | — | 9301 | **10.2** | **1.29** | 5.73 | 4.70 | 1.0 | 6.3 | **38.3** | **3.7** | 1.4 |
| S1 | 550 | 0.600 |  —    |  6576 |  8.4 | 1.20 |  5.74 | 4.99 |  0.5 |  5.7 | 12.0 |  3.1 | 1.0 |
| S1 | 600 | 0.677 | 0.507 |  7011 | 10.6 | 1.55 |  6.15 | 4.44 |  0.8 |  4.0 |  6.7 |  4.2 | 0.8 |
| S1 | 660 | 0.803 | 0.583 |  9361 | 15.6 | 2.06 |  7.70 | 4.22 |  0.8 |  5.0 |  6.7 |  6.8 | 1.3 |
| S1 | 700 | 0.767 | 0.532 | 11164 | 18.2 | 2.83 |  8.88 | 3.64 |  0.7 |  8.0 |  9.7 |  8.0 | 1.1 |
| S1 | 750 | 0.803 |  —    |  9691 | 13.7 | 2.01 |  8.21 | 4.38 |  0.1 |  6.3 |  8.7 |  5.8 | 1.8 |
| S2 | 1   | 0.410 |  —    | 18497 | 36.8 | 9.97 | 17.59 | 2.59 | 19.3 | 17.0 |  9.7 | 17.4 | 2.6 |
| S2 | 30  | 0.553 |  —    | 17871 | 36.3 | 9.96 | 17.70 | 2.51 | 18.6 | 14.3 |  8.0 | 17.1 | 2.3 |
| S2 | 60  | 0.550 | 0.633 | 16159 | 33.7 | 7.55 | 15.26 | 3.01 | 17.9 | 12.3 |  9.0 | 15.8 | 2.8 |
| S2 | 100 | 0.610 | 0.600 | 15731 | 35.9 | 6.79 | 13.33 | 2.66 | 15.6 | 11.0 |  6.7 | 16.9 | 3.9 |
| S2 | 150 | 0.647 |  —    | 15745 | 35.1 | 5.81 | 12.00 | 2.99 | 13.1 | 12.0 |  7.0 | 16.6 | 3.9 |
| S2 | 200 | 0.623 | 0.617 | 14790 | 34.0 | 5.66 | 10.65 | 2.86 | 13.3 | 13.0 |  6.3 | 16.0 | 3.9 |
| S2 | 250 | 0.710 |  —    | 13205 | 32.1 | 4.72 |  8.97 | 2.86 | 11.6 | 10.0 |  8.3 | 15.1 | 2.8 |
| S2 | 300 | 0.717 | 0.650 | 16379 | 38.8 | 5.81 | 10.54 | 2.62 | 16.9 | 11.3 |  4.3 | 18.4 | 4.4 |
| S2 | 350 | 0.727 |  —    | 17404 | 38.2 | 5.51 | 11.78 | 3.08 | 11.0 | 15.7 |  6.7 | 18.1 | 3.3 |
| S2 | 400 | 0.747 | 0.682 | 16343 | 34.8 | 4.85 | 10.78 | 3.01 | 14.3 | 14.3 |  4.3 | 16.4 | 3.0 |
| S2 | 450 | 0.773 |  —    | 17220 | 34.7 | 4.84 | 12.47 | 3.43 | 11.4 | 14.3 |  4.0 | 16.3 | 3.2 |
| S2 | **500** | **0.847** | 0.663 | 15771 | 32.9 | 3.76 | 10.81 | 3.64 | 12.5 | 12.0 | **2.3** | 15.5 | 2.9 |
| S2 | 570 | 0.727 | **0.682** | 20933 | **44.6** | 4.97 | 13.66 | 3.58 | 12.1 | 25.0 |  8.0 | **21.3** | **5.0** |

Field key: `resp_tok` = mean output tokens; `turns` = total agent turns per question; `n_search` = mean number of search tool calls; `queries` = total queries emitted (search supports a list of queries per call); `qPer` = queries per search call; `quoted%` = fraction of queries containing `"…"` exact-phrase quotes; `trunc%` = fraction of trajectories hitting the 32 k / 48 k cap; `noAns%` = fraction missing `<answer>` tag (forced score = 0); `thinks` = `<think>` blocks per trajectory; `hedge` = self-reflection token count (`wait`/`actually`/etc.).

---

## 10 · Implications for the paper

1. **Why a two-stage curriculum is necessary, not optional.** Stage 1 cannot run indefinitely; it possesses a deterministic collapse mechanism (Section 6) that no amount of additional steps can avoid. The two-stage design is therefore not just a convenience — it is a *required correction* to the otherwise unbounded compression force of GRPO + seq-mean-token-mean + no-KL on a bounded-context task.

2. **Why checkpoint selection matters more than learning rate.** The choice of ckpt 220 (rather than ckpt 480 which has higher sim reward) is the *single most important hyperparameter* of the entire training run. Selecting a later checkpoint by reward would warm-start S2 into a behavioral basin that no longer contains the priors needed for hard multi-hop tasks.

3. **Why behavioral metrics — not reward curves — should be monitored.** Quoted-query fraction, `<think>` block count, and `no_extraction` rate together would have caught Regime C decay *before* the collapse occurred. Sim reward and GAIA pass@1 alone did not catch it (both kept rising through step 480).

4. **Why the sim-to-real gap is the right success metric.** The paper's contribution is not "we got 71 % on GAIA" — many papers do. It is "we trained an RL policy whose sim reward correlates with GAIA pass@1 across the entire training run", which is the actual hard problem in agentic RL. Figure 5 is the quantitative version of this claim.

5. **What an ablation table should look like.** A complete picture would require ablating (a) the 48 k cap (revert to 32 k in S2), (b) the data mix (S2 with S1 data only), (c) the warm-start point (S2 from ckpt 480), and (d) the loss aggregation (`token-mean` instead of `seq-mean-token-mean`). Each row should be benchmarked on the **(sim reward trajectory shape, sim-real gap, GAIA pass@1)** triple — the behavioral analysis above predicts which rows will collapse and how.

---

## 11 · Reproducing this analysis

```bash
cd <this_dir>

# 1. Extract 30+ behavioral metrics per checkpoint (≈ 1 min)
python run_all.py          # writes behavior_timeline.json

# 2. Mine 10 representative trajectories for the regime examples (≈ 5 sec)
python mine_examples.py    # writes trajectory_examples.json

# 3. Render 6 figures (≈ 5 sec)
python make_figs.py        # writes fig1..6.png
```

The extractor inputs are the public rollout JSONL files and the per-task GAIA result JSONs listed at the top of this document.

— end —

---

## 12 · Continuous training path  (S1 step 0–220 ▸ S2 step 0–570 spliced at step 220)

Because Stage 2 warm-starts directly from S1 ckpt-220 (the abandoned S1 ≥ 240 branch never reaches the final model), it is most informative to view both stages on a **single continuous global-step axis**: S1 steps 0–220 are followed by S2 steps 0–570 (re-indexed to global steps 220–790). All twelve behavioral metrics below are computed *both* on the on-policy training rollouts (300 trajectories per step) *and* on the held-out GAIA `pass@1` evaluation (412 tasks per step) at every available checkpoint.

![continuous full](fig_continuous_full.png)

### 12.1 Twelve trends the paper should highlight

| # | Metric | S1 0 → 220 (rollout) | S2 0 → 570 (rollout) | What this tells the reader |
|---|---|---|---|---|
| 1 | **Task-completion accuracy** | 0.50 → **0.70** | 0.41 → **0.85** | S2 starts at 0.41 (harder data) and ends 21 pp above the entire S1 peak. |
| 2 | **Mean response length** (tokens) | 17.4 k → **10.8 k** | 18.5 k → **20.9 k** | S1 *compresses* output to fit the 32 k cap; S2 *expands* output once the 48 k cap is in effect. |
| 3 | **Mean # of agent turns** | 28.3 → **19.2** | 36.8 → **44.6** | S1 shortens trajectories; S2 doubles them — multi-hop reasoning becomes feasible. |
| 4 | **Search-tool calls / traj** | 7.8 → **4.0** | 10.0 → **5.0** | Search becomes *more efficient* (fewer calls), then plateaus. |
| 5 | **Browse / visit calls / traj** ↑ | 5.5 → 4.6 (flat) | 7.5 → **16.4** | S2 learns to *read pages*, not just search them. **One of the most consequential shifts**. |
| 6 | **Browse share of tool usage** ↑ | 0.46 → 0.56 | 0.50 → **0.76** | By the end the model spends three out of four tool calls reading content. |
| 7 | **Search concurrency** (queries / search call) ↑ | 2.70 → **3.06** | 2.59 → **3.58** | Each search request now batches more diverse queries — a learned cost-amortization. |
| 8 | **Total queries emitted / traj** | 16.8 → 9.6 | 17.6 → 13.7 | Total query volume drops, but the **information per query** rises (see #5–7). |
| 9 | **Quoted exact-phrase queries** | 0.46 → 0.16 | 0.19 → 0.12 | Quoted-query prior shrinks; under S2 the page-reading channel partially substitutes for it. |
| 10 | **`<think>` reasoning blocks / traj** | 13.0 → 8.6 | 17.4 → **21.3** | S2 reasons *17 % more often per trajectory* than at its own start — and ≈ 2.5× more often than S1 ckpt-220. |
| 11 | **Thinking density** (chars-in-`<think>` / total chars) | 0.092 → 0.089 (flat) | 0.102 → **0.142** | A larger share of the output is *reasoning text* by the end of S2. |
| 12 | **Self-reflection hedges** (`wait / actually / 重新…`) | 2.58 → 1.47 | 2.64 → **5.03** | S2 *talks itself out of mistakes* 3.4× more than S1 ckpt-220. |

### 12.2 Headline figure — four trends summarized

![continuous headline](fig_continuous_headline.png)

This figure is intended as a single drop-in for a paper. It pairs train (sim rollout) and eval (GAIA) on the same x-axis for the four metrics that best summarize the policy's transformation:

- (a) Accuracy converges between sim and real after the splice, ending at ≈ 0.68 on GAIA pass@1.
- (b) Browse / visit calls climb roughly linearly through S2 — both on the training distribution *and* on the held-out GAIA benchmark, evidence that this is a transferable behavior, not an over-fit to the sim reward.
- (c) Search concurrency grows in lockstep, indicating the model is *parallelizing* its retrieval rather than serializing it.
- (d) `<think>` block count more than doubles at the splice and keeps growing — long-horizon planning becomes the dominant reasoning mode.

### 12.3 Train-vs-eval accuracy headline

![continuous acc](fig_continuous_acc.png)

A single-panel figure suitable for the introduction: train rollout reward (blue) vs GAIA pass@1 (red) vs GAIA pass@4 (green) along the continuous training path. Sim and real curves track each other for almost the entire training run; the only systematic divergence appears in the late-S2 region (steps 700+) where train reward over-shoots while GAIA continues steady at ≈ 0.68 — suggesting the model is approaching the natural ceiling on this benchmark.

### 12.4 Reading the figure as a paper claim

The single-axis view supports three crisp paper claims:

1. **The behavioral transformation is monotonic along the global training axis** for 8 of 12 metrics (browse calls, browse share, search concurrency, `<think>` blocks, thinking density, hedge count, accuracy, turns). The remaining 4 (response length, total queries, quoted-query fraction, raw search calls) follow the expected efficiency-compression pattern.
2. **Sim and real agree on direction for every behavior** for which both can be measured. Where the curves disagree in absolute level (e.g. response length), it is because eval has no length cap so absolute numbers cannot match; **direction always matches**.
3. **The most consequential change happens *during* S2, not at the splice itself.** The splice produces a step-function in length, turns and `<think>` count (data-mix and context-window difference), but the slow climb in browse share (0.50 → 0.76) and `<think>` density (0.10 → 0.14) is a *continuous learning effect inside S2* — direct evidence that S2 is doing real RL work, not just inheriting a better starting point.

