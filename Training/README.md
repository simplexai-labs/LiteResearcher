# LiteResearcher — Open-Source Training Code

> **Companion code release for the LiteResearcher paper**
> *A scalable agentic-RL framework for deep-research agents — Qwen3-4B trained from
> SFT cold-start → Stage-1 RL → Stage-2 RL → 71.3 % GAIA / 78.0 % Xbench-DS / 83.1 %
> Frames / 32.5 % BrowseComp, at zero search API cost on local infra.*

This branch (`release/literesearcher`) is a **minimal, sanitized snapshot** of the
[verl](https://github.com/volcengine/verl) tree that we used to train
LiteResearcher. It contains exactly the code needed to reproduce our
**two-stage GRPO+TIS** training pipeline. Upstream verl variants
(DAPO, R1, langgraph, vLLM mode, etc.) and our own ablation scripts are pruned.

| Stage | Script | Setup | Data |
|---|---|---|---|
| **Stage 1** | `examples/sglang_multiturn/search_browser/stage1_rag_only.sh` | 8 × H20 (1 node) | [`stage1/train.parquet`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data) (pure local-RAG, 10,398 rows) |
| **Stage 2** | `examples/sglang_multiturn/search_browser/stage_2_mix_rag_on_policy_48k.sh` | 16 × H20 (2 nodes) | [`stage2/train.parquet`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data) (25-bucket curriculum, 16,199 rows) |

Stage 2 continues from a Stage-1 checkpoint (we used `global_step_220`).

---

## Related repositories

| Repo / branch | Purpose |
|---|---|
| **this branch** (`release/literesearcher`) | Training code — what's documented here |
| `feat/qwen3_5-support` | Active development (qwen3.5 support, debug scripts) |
| `literesearcher_html_deck/` | 30-slide presentation deck (Claude × WeaveBench style) |
| `literesearcher_presentation/` | Earlier .pptx version of the deck |
| `literesearcher_behavior_analysis/` | Bilingual analysis of the 4 behavioral mechanisms driving sim2real |

---

## Quick layout

```
verl/                                         ← all verl framework code (training loop, Ray, FSDP, SGLang)
├── verl/                                     ← Python package
│   ├── trainer/main_ppo.py                  ← training entry point
│   ├── trainer/ppo/ray_trainer.py           ← RayPPOTrainer.fit()
│   ├── experimental/agent_loop/             ← multi-turn tool-calling agent
│   │   ├── agent_loop.py                    ← AgentLoopManager + AgentLoopWorker
│   │   ├── tool_agent_loop.py               ← state machine (PENDING→GENERATING→TOOLS→…)
│   │   └── tool_parser.py / tool_registry.py
│   ├── tools/                                ← google_search_tool.py + browse_tool.py
│   ├── workers/fsdp_workers.py              ← actor + ref FSDP worker
│   ├── workers/rollout/sglang_rollout/      ← SGLang HTTP-server-mode rollout
│   └── utils/reward_score/llm_judge_async.py ← async LLM-judge reward
│
├── examples/sglang_multiturn/
│   ├── config/                               ← Hydra configs
│   │   ├── google_search_browse_multiturn_grpo.yaml   ← main training config
│   │   └── tool_config/google_search_browse_tool_config.yaml
│   └── search_browser/
│       ├── stage1_rag_only.sh               ← ★ Stage 1 RL launcher
│       ├── stage_2_mix_rag_on_policy_48k.sh ← ★ Stage 2 RL launcher
│       ├── ray_setup_head.sh                ← multi-node Ray head bring-up
│       ├── MULTINODE_SETUP.md               ← multi-node setup details
│       ├── QUICKSTART_MULTINODE.md
│       ├── rag_diskann/                     ← local-RAG diskANN server (Google-Search backend)
│       │   ├── start.sh / stop.sh           ← service control
│       │   ├── local_rag_diskann_server.py  ← Flask HTTP server
│       │   └── embedding_server_diskann.py  ← embedding service
│       ├── tool_backend/                    ← Browse service + LLM-Judge config
│       │   ├── browser_service.py           ← Browse tool HTTP server
│       │   ├── start_browse.sh / ray_launch.sh
│       │   ├── .env.example                 ← ← copy to .env and fill in
│       │   └── benchmark/                   ← tool-latency benchmarks
│       └── sgl_serve/                       ← SGLang FP8 serving (used to host the LLM-Judge model)
│
└── data/deepresearch_rl/
    ├── README_data_processing.md            ← data-pipeline overview
    ├── merge_rag_with_mask_url.py           ← main data-prep script (Stage 1)
    ├── sample_parquet.py                    ← quick parquet inspector
    └── stage2/
        ├── merge_data.py / merge_wiki.py / merge_science.py
        └── tools_kwargs_flow.md             ← schema notes
```

> The `*.parquet` data files themselves are **not** checked in (gitignored).
> See [Data preparation](#data-preparation) for how to regenerate them.

---

## Environment

```bash
# 1. Conda env
conda create -n verl-v060 python=3.11 -y
conda activate verl-v060

# 2. Install verl + backends (CUDA 12.6, torch 2.8, sglang 0.5.2)
pip install -e .                  # core
pip install -e .[sglang]          # SGLang backend
pip install -e .[gpu]             # flash-attn + liger
pip install -e .[math]            # math verification (optional)

# 3. Verify
python -c "import verl, sglang; print(verl.__version__)"
```

### Required external services

LiteResearcher training calls **three external services** per rollout step. Spin them
up *before* launching either stage script:

| Service | Why | Where |
|---|---|---|
| **Local-RAG Google-Search** | Replaces paid Google Search API with a local diskANN index over 32 M pages | `examples/sglang_multiturn/search_browser/rag_diskann/start.sh` |
| **Browse service** | Fetches + summarizes URLs (with Postgres caching) | `examples/sglang_multiturn/search_browser/tool_backend/start_browse.sh` |
| **LLM Judge** | Async chat-completions endpoint for reward scoring | any OpenAI-compatible server — we used SGLang FP8 in `sgl_serve/qwen3_4B_FP8_router.sh` |

Then **copy `tool_backend/.env.example` → `tool_backend/.env`** and fill in:
- `LLM_JUDGE_API_BASE`, `LLM_JUDGE_MODEL`
- `SCRAPEDO_API_KEY` (or your scrape backend of choice)
- `PG_HOST/PORT/USER/PASSWORD/DATABASE` (Postgres for browse cache)

Tool endpoints (`search_service_url`, `browse_service_url`) are configured in
`examples/sglang_multiturn/config/tool_config/google_search_browse_tool_config.yaml`
— point them at your running services.

---

## Data preparation

The training data is published as a 🤗 dataset:
**[`simplex-ai-inc/LiteResearcher-Data`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data)**.
One command gets you all three parquet files (28K prompts, 19 MB total):

```bash
hf download simplex-ai-inc/LiteResearcher-Data --repo-type dataset \
            --local-dir ./literesearcher_data
```

| File | Used by | Rows |
|---|---|---|
| `literesearcher_data/stage1/train.parquet`     | Stage 1 train | 10,398 — pure local-RAG warmup |
| `literesearcher_data/stage2/train.parquet`     | Stage 2 train | 16,199 — 25-bucket difficulty + diversity curriculum |
| `literesearcher_data/validation/wiki.parquet`  | Stage 1+2 val | 1,694 — shared Wikipedia QA monitoring set |

Then point the launcher env-vars at the downloaded files (see Stage 1/2 sections
below). The dataset card on Hugging Face documents the full schema and curriculum
design.

### Rebuilding from raw sources (optional)

If you want to regenerate the parquets locally from raw QA dumps:

```bash
# Stage-1 RAG data (multi-hop + single-hop, mask URLs in tool responses)
python data/deepresearch_rl/merge_rag_with_mask_url.py

# Stage-2 mix data
python data/deepresearch_rl/stage2/merge_data.py     # combine subsets
python data/deepresearch_rl/stage2/merge_wiki.py     # wiki subset
python data/deepresearch_rl/stage2/merge_science.py  # bio/chem/math subsets
```

See [`data/deepresearch_rl/README_data_processing.md`](data/deepresearch_rl/README_data_processing.md)
for raw-source layout and schema.

---

## Training

### Pre-step: SFT cold-start

Stage-1 RL initializes from a Qwen3-4B that has been SFT'd on 68.2 K
agentic-tool-use trajectories distilled from Tongyi-DeepResearch-30B. We trained
ours with **LLaMA-Factory**:

```bash
# Hyper-params we used (LLaMA-Factory):
#   base model        : Qwen/Qwen3-4B
#   training examples : 68.2 K  (distilled deep-research trajectories)
#   max length        : 64 K
#   global batch size : 128
#   epochs            : 3  (we used the ckpt-533 snapshot)
#   lr / scheduler    : 1e-5 cosine, 3 % warmup
```

A SFT cold-start is a hard requirement — pure RL from a vanilla Qwen3-4B does
not learn to use the tools reliably enough to bootstrap rewards. If you don't
have the distillation data on hand, swap in any tool-using SFT of comparable
quality and adjust expectations accordingly.

### Stage 1 — single node, 8 × H20

Override any of `PROJECT_DIR`, `CONDA_ENV`, `TRAIN_DATA`, `VAL_DATA`, `MODEL_PATH`,
`CHECKPOINT_PATH`, `ENV_FILE`, `RESUME_MODE` by exporting before the call:

```bash
export PROJECT_DIR=/path/to/repo
export MODEL_PATH=/path/to/qwen3-4b-sft-cold-start    # ← your SFT checkpoint
bash examples/sglang_multiturn/search_browser/stage1_rag_only.sh
```

Key config knobs (override at the CLI via Hydra):
- `data.train_batch_size=256`, `actor_rollout_ref.rollout.n=8`   → 256 prompts × 8 samples = 2048 rollouts / step
- `data.max_response_length=32768`, `actor_rollout_ref.rollout.multi_turn.max_assistant_turns=40`
- `actor_rollout_ref.actor.use_kl_loss=False`, `actor_rollout_ref.actor.entropy_coeff=0`
- `actor_rollout_ref.actor.clip_ratio_high=0.4`, `clip_ratio_low=0.2`
- TIS: `algorithm.rollout_is=true`, `rollout_is_threshold=2.0`, `rollout_is_level=token`, `rollout_is_mode=truncate`

Plateau happens around `global_step_220` — that's the Stage-2 starting point.

### Stage 2 — two nodes, 16 × H20

```bash
# On head node first:
bash examples/sglang_multiturn/search_browser/ray_setup_head.sh
# Use the printed command to join the worker node.

# Then launch:
export MODEL_PATH=/path/to/stage1-ckpt/global_step_220
bash examples/sglang_multiturn/search_browser/stage_2_mix_rag_on_policy_48k.sh
```

Stage-2 differences vs Stage-1:
- `data.train_batch_size=128` (smaller bs, longer ctx)
- `data.max_response_length=49152`, `multi_turn.max_assistant_turns=60`
- `actor_rollout_ref.rollout.temperature=1.0` (was 0.7)
- `trainer.nnodes=2`

---

## Monitoring

```bash
# rollout trajectories (one jsonl per global_step)
tail -f rollout_trajectory/<PROJECT>_<EXPERIMENT>/<TIMESTAMP>/*.jsonl

# training log (Hydra + Ray + verl)
tail -f logs_packing/<PROJECT>_<EXPERIMENT>_<TIMESTAMP>.log

# SwanLab dashboard (if SWANLAB_API_KEY set)
# WandB dashboard  (if WANDB_API_KEY  set)
```

Key metrics:
- `reward/mean` — average per-step reward
- `mismatch/rollout_is_mean` (should be ≈1.0), `…_eff_sample_size` (>0.5), `…_veto_fraction` (<0.1)
- `timing/gen` ≈ 60–70 % of step time (rollout is the bottleneck)

---

## Reproducing the paper numbers

| Stage | Plateau-avg accuracy on the **internal RL val set** | Wall-clock |
|---|---|---|
| SFT cold-start (Qwen3-4B) | 55.6 | — |
| + Stage 1 RL              | **64.7** (plateau @ step ~220) | ~7 days × 8 H20 |
| + Stage 2 RL              | **68.3** (plateau)             | ~5 days × 16 H20 |

Out-of-distribution benchmark numbers (GAIA / Xbench-DS / Frames / BrowseComp) require
running the corresponding eval harness in our separate `DeepResearch/bench_results/`
codebase (not part of this release).

For a write-up of the **behavioral mechanisms** driving the Stage-1→Stage-2 lift
(M1 tool-use ↑, M2 retry ↓ + recover ↑, M3 longer `<think>` blocks, M4 calibrated
hedging), see `literesearcher_behavior_analysis/` on this same repo.

---

## Citation

```bibtex
@article{lee2026literesearcher,
  title  = {LiteResearcher: Scalable Agentic RL for Deep Research Agents},
  author = {Lee, Wanli and others},
  year   = {2026},
  eprint = {2604.17931},
  archivePrefix = {arXiv},
}
```

## License

Inherits the upstream verl Apache-2.0 license. See `LICENSE` for the full text.

---

*Issues with the release? File against the
[upstream verl repo](https://github.com/volcengine/verl), tag with `[LiteResearcher]`,
or open one on this fork.*
