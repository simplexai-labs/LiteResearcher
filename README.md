<div align="center">

<img src="logo_cropped.png" width="400">

### A Low-Cost, Scalable Agentic RL Training Framework for Deep Research Agent

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.17931)
[![Webpage](https://img.shields.io/badge/Project-Webpage-0a0a0a?logo=githubpages&logoColor=white)](https://simplex-ai-inc.github.io/LiteResearcher/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-ffcc00?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data)
[![Corpus](https://img.shields.io/badge/Corpus%2032M-HuggingFace-ffcc00?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Corpus)
[![Model](https://img.shields.io/badge/Model-HuggingFace-ffcc00?logo=huggingface&logoColor=black)](https://huggingface.co/simplex-ai-inc/LiteResearcher-4B)
[![Trajectories](https://img.shields.io/badge/Trajectories-Live%20Viewer-4f8cff)](https://simplex-ai-inc.github.io/LiteResearcher/cases/)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)

**If you like our project, please give us a star ⭐ on GitHub for the latest update.**

</div>

**LiteResearcher-4B is a 4B deep research agent trained with zero marginal RL API cost, outperforming 30B open-source deep research agents and matching frontier systems such as Claude-4.5-Sonnet and GPT-5.**

| Key number | Result | Why it matters |
|------------|--------|----------------|
| **Open-source SOTA** | **71.3% GAIA** / **78.0% Xbench-DS** | Beats 30B open-source agents on the headline deep-research benchmarks and surpasses Claude-4.5-Sonnet on GAIA and GPT-5-high on Xbench-DS. |
| **+15.7 GAIA points from RL** | SFT **55.6%** → RL **71.3%** | Shows the gain comes from scalable RL; AgentCPM-Explore reports only **+3.8** points when training with live web interaction. |
| **73.2M local tool calls** | **$0 marginal API cost** | The same RL tool-call volume would cost **$59K-$243K** with live web search/browse APIs. |

**LiteResearcher** makes Agentic RL scalable by replacing live-web interaction during RL with a stable local search/browse environment that mirrors real-world search dynamics while eliminating per-call API cost.

<div align="center">
<img src="docs/static/compare.png" width="100%">
<p><em>Left: Xbench-DeepSearch accuracy vs. model size — our 4B model reaches 78.0%, matching/surpassing 100×+ larger systems. Right: Average rollout time and cost per turn — LiteResearcher is the fastest and cheapest.</em></p>
</div>

## Results

<div align="center">
<img src="figures/table_main_results.png" width="100%">
</div>

## Method Overview

<div align="center">
<img src="docs/static/overview.png" width="90%">
</div>

Three pillars enable low-cost, scalable Agentic RL:

1. **Co-construct Training Data & Corpus** — Scale up information sources with a simple-but-effective synthesis pipeline, then co-evolve training QA pairs and the local webpage corpus.
2. **Stable Local Tool Environment** — Build local search engine (Milvus + BGE-M3) and local browse tool (PostgreSQL) from ~32M real webpages, enabling the RL stage to run fully locally with no API consumption, 10–46× speedup, and zero marginal tool cost.
3. **Difficulty-Aware Curriculum RL** — Multi-stage curriculum with on-policy GRPO, filtering tasks by pass@8 difficulty to sustain monotonic improvement.

## Trajectory Cases

We release 15 hand-audited rollout trajectories from LiteResearcher-4B across 8 deep-research benchmarks (GAIA, Xbench-DS, Frame, HLE, Seal-0, WebwalkerQA, BrowseComp, BrowseComp-zh). Each case is judged `correct`, leak-free, and reviewed by 4 independent Opus-4.7 (1M context) subagents to verify the answer is derived from cited evidence (no fabrication, no hedged guess).

**🔎 Live viewer:** https://simplex-ai-inc.github.io/LiteResearcher/cases/

Each trajectory renders 40–170 steps showing the model's `think` → `search` → `visit` → `answer` chain, with tool queries, visited URLs, and tool responses inline. Source data lives under [`docs/cases/`](docs/cases/).

## Repository Structure

```
├── Inference/              # Inference & evaluation (released)
├── Training/               # RL training — GRPO + curriculum (released)
├── DataGen/                # Data synthesis (coming soon)
├── Environment/            # Local search/browse environment (released)
└── docs/                   # Project page
```

## Quick Start — Evaluation

```bash
cd Inference
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set MODEL, SERPER_KEY_ID (browser uses Jina Reader by default; set SCRAPEDO_API_KEY only if using BROWSER_PROVIDER=scrapedo)

# Start model server (SGLang/vLLM)
bash scripts/start_sglang.sh

# Run evaluation
bash scripts/run_all.sh
```

See [`Inference/README.md`](Inference/README.md) for detailed configuration and usage.

## Quick Start — Training

The full two-stage RL training pipeline (GRPO + TIS + difficulty-aware curriculum)
is in [`Training/`](Training/), and the training data is hosted on
[🤗 `LiteResearcher-Data`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data).

```bash
cd Training
pip install -e .[sglang]                   # install verl-based training stack
cp examples/sglang_multiturn/search_browser/tool_backend/.env.example \
   examples/sglang_multiturn/search_browser/tool_backend/.env
# Edit .env: set PG_*, SUMMARY_API_*, LLM_JUDGE_API_*, optional SCRAPEDO_API_KEY

# One-shot data download (28K prompts, 19 MB)
hf download simplex-ai-inc/LiteResearcher-Data --repo-type dataset \
            --local-dir ./literesearcher_data

# Stage 1 — single node 8×H20, RAG-only warmup, 32K ctx
export TRAIN_DATA=./literesearcher_data/stage1/train.parquet
export VAL_DATA="$TRAIN_DATA"     # no separate val bundled; verl needs a non-empty val_files
export MODEL_PATH=/path/to/qwen3-4b-sft-cold-start
bash examples/sglang_multiturn/search_browser/stage1_rag_only.sh

# Stage 2 — 2 nodes × 8 H20, mix curriculum, 48K ctx, resume from Stage-1 step ~220
export TRAIN_DATA=./literesearcher_data/stage2/train.parquet
export VAL_DATA="$TRAIN_DATA"
export MODEL_PATH=/path/to/stage1-ckpt/global_step_220
bash examples/sglang_multiturn/search_browser/stage_2_mix_rag_on_policy_48k.sh
```

See [`Training/README.md`](Training/README.md) for the full reproduction recipe
(including the SFT cold-start prerequisite, environment variables, and config
knobs) and the
[dataset card](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data)
for the data schema and curriculum design.

## Release Plan

- [x] Evaluation code
- [x] Project page
- [x] Model weights ([`LiteResearcher-4B`](https://huggingface.co/simplex-ai-inc/LiteResearcher-4B))
- [x] Local search/browse environment setup ([`Environment/`](Environment/))
- [x] Search corpus — 32M records ([`LiteResearcher-Corpus`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Corpus))
- [x] Training code — GRPO + curriculum RL ([`Training/`](Training/))
- [x] Training data — Stage-1 & Stage-2 prompts ([`LiteResearcher-Data`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data))
- [ ] Data synthesis pipeline

## Citation

```bibtex
@article{li2026literesearcher,
  title={LiteResearcher: A Scalable Agentic RL Training Framework for Deep Research Agent},
  author={Li, Wanli and Qu, Bince and Pan, Bo and Zhang, Jianyu and Liu, Zheng and Zhang, Pan and Chen, Wei and Zhang, Bo},
  journal={arXiv preprint arXiv:2604.17931},
  year={2026}
}
```

## License

Apache 2.0
