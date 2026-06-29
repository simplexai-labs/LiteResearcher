<div align="center">

# LiteResearcher

### A Low-Cost, Scalable Agentic RL Training Framework for Deep Research Agent

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.17931)
[![Project](https://img.shields.io/badge/Project-Webpage-0a0a0a?logo=githubpages&logoColor=white)](https://simplexai-labs.github.io/LiteResearcher/)
[![Models](https://img.shields.io/badge/Models-HuggingFace-ffcc00?logo=huggingface&logoColor=black)](https://huggingface.co/simplex-ai-inc)
[![Datasets](https://img.shields.io/badge/Datasets-HuggingFace-ffcc00?logo=huggingface&logoColor=black)](https://huggingface.co/simplex-ai-inc)
[![Live Demo](https://img.shields.io/badge/Trajectories-Live%20Viewer-4f8cff)](https://simplexai-labs.github.io/LiteResearcher/cases/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**If you like our project, please give us a star ⭐ on GitHub for the latest update.**

</div>

## News

**2026-06**
- 🚀 Training code released — GRPO + difficulty-aware curriculum ([`training/`](training/))
- 📊 Training data released — Stage-1 & Stage-2 prompts ([`LiteResearcher-Data`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data))
- 🧊 SFT cold-start checkpoint released ([`LiteResearcher-4B-SFT`](https://huggingface.co/simplex-ai-inc/LiteResearcher-4B-SFT))
- 🛠️ Data synthesis pipeline released ([`datagen/`](datagen/))
- 🌐 Local search/browse environment released ([`environment/`](environment/))
- 📚 32M-record search corpus released ([`LiteResearcher-Corpus`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Corpus))

**2026-04**
- 🎯 RL model weights released ([`LiteResearcher-4B`](https://huggingface.co/simplex-ai-inc/LiteResearcher-4B))
- 📈 Evaluation code & project page released

**LiteResearcher-4B is a 4B deep research agent that matches frontier systems at a fraction of the size — trained with $0 marginal API cost by replacing live-web interaction with a stable local search/browse environment that mirrors real-world search dynamics.**

**Highlights**

- **Open-source SOTA** — **71.3% GAIA** / **78.0% Xbench-DS**, beating 30B open-source agents and surpassing Claude-4.5-Sonnet on GAIA and GPT-5-high on Xbench-DS.
- **+15.7 GAIA points from RL** — SFT **55.6%** → RL **71.3%**, vs. only **+3.8** for AgentCPM-Explore when training with live web interaction.
- **$0 marginal API cost** — **73.2M** local tool calls during RL; the same volume would cost **$59K–$243K** via live search/browse APIs.

<div align="center">
<img src="docs/static/compare.png" width="100%">
<p><em>Left: Xbench-DeepSearch accuracy vs. model size — our 4B model reaches 78.0%, matching/surpassing 100×+ larger systems. Right: Average rollout time and cost per turn — LiteResearcher is the fastest and cheapest.</em></p>
</div>

## Results

**Main results.** Comparison across commercial models and open-source deep research agents on eight benchmarks. **Best open-source results are in bold.**

| Model | GAIA | BrowseComp | BrowseComp-ZH | HLE | Frames | WebWalker | Seal-0 | Xbench-DS |
|:------|----:|----:|----:|----:|----:|----:|----:|----:|
| **Commercial Models** | | | | | | | | |
| Claude-4-Sonnet | 68.3 | 12.2 | 29.1 | 20.3 | 80.7 | 61.7 | – | 64.6 |
| Claude-4.5-Sonnet | 71.2 | 19.6 | 40.8 | 24.5 | 85.0 | – | 53.4 | 66.0 |
| DeepSeek-V3.2 | 63.5 | 67.6 | 65.0 | 40.8 | 80.2 | – | 38.5 | 71.0 |
| DeepSeek-V3.1 | 63.1 | 30.0 | 49.2 | 29.8 | 83.7 | 61.2 | – | 71.0 |
| Minimax-M2 | 75.7 | 44.0 | 48.5 | 31.8 | – | – | – | 72.0 |
| OpenAI-GPT-5-high | 76.4 | 54.9 | 65.0 | 35.2 | – | – | 51.4 | 77.8 |
| GLM-4.6 | 71.9 | 45.1 | 49.5 | 30.4 | – | – | – | 70.0 |
| Kimi-Researcher | – | – | – | 26.9 | 78.8 | – | 36.0 | 69.0 |
| Kimi-K2-0905 | 60.2 | 7.4 | 22.2 | 21.7 | 58.1 | – | 25.2 | 61.0 |
| **Open-Source Models (>8B)** | | | | | | | | |
| Tongyi DeepResearch-30B | 70.9 | 43.4 | 46.7 | 32.9 | 90.6 | 72.2 | – | 75.0 |
| ASearcher-QWQ-v2 | 58.7 | – | – | – | 74.5 | – | – | 51.1 |
| WebSailor-30B | 53.2 | – | – | – | – | – | – | 53.3 |
| WebDancer-QwQ | 51.5 | 3.8 | 18.0 | – | – | 47.9 | – | 38.3 |
| DeepMiner-32B | 58.7 | 33.5 | 40.1 | – | – | – | – | 62.0 |
| AFM-RL-32B | 55.3 | 11.1 | – | 18.0 | – | 63.0 | – | – |
| SFR-DeepResearch | 66.0 | – | – | 28.7 | 82.8 | – | – | – |
| **Small Models (≤8B)** | | | | | | | | |
| Mirothinker-8B | 66.4 | **31.1** | **40.2** | 21.5 | 80.6 | 60.6 | 40.4 | 60.6 |
| WebExplorer-8B | 50.0 | 15.7 | 32.0 | 17.3 | 75.7 | 62.7 | – | 53.7 |
| AgentCPM-Explore-4B | 63.9 | 24.1 | 29.1 | 19.1 | 82.7 | 68.1 | 40.5 | 70.0 |
| **LiteResearcher-4B** | **71.3** | 27.5* | 32.5* | **22.0** | **83.1** | **72.7** | **41.8** | **78.0** |

<sub>All scores are accuracy (%); **–** = not reported. Among **≤8B models**, the best score on each benchmark is in **bold** — LiteResearcher-4B leads on 6 of 8 (Mirothinker-8B leads on BrowseComp and BrowseComp-ZH). Results without `*` use a 128k context window; `*` results use a 64k window with a memory mechanism that, on reaching the context limit, calls a summarization model to compress each prior tool-interaction step into one sentence.</sub>

## Method Overview

<div align="center">
<img src="docs/static/overview.png" width="90%">
</div>

Three pillars enable low-cost, scalable Agentic RL:

1. **Co-construct Training Data & Corpus** — Scale up information sources with a simple-but-effective synthesis pipeline, then co-evolve training QA pairs and the local webpage corpus.
2. **Stable Local Tool Environment** — Build local search engine (Milvus + BGE-M3) and local browse tool (PostgreSQL) from ~32M real webpages, enabling the RL stage to run fully locally with no API consumption, 10–46× speedup, and zero marginal tool cost.
3. **Difficulty-Aware Curriculum RL** — Multi-stage curriculum with on-policy GRPO, filtering tasks by pass@8 difficulty to sustain monotonic improvement.

## Trajectory Cases

We release 15 hand-audited rollout trajectories from LiteResearcher-4B across 8 deep-research benchmarks (GAIA, Xbench-DS, Frames, HLE, Seal-0, WebWalker, BrowseComp, BrowseComp-ZH). Each case is judged `correct`, leak-free, and reviewed by 4 independent Opus-4.7 (1M context) subagents.

**🔎 Live viewer:** https://simplexai-labs.github.io/LiteResearcher/cases/

Each trajectory renders 40–170 steps showing the model's `think` → `search` → `visit` → `answer` chain, with tool queries, visited URLs, and tool responses inline. Source data lives under [`docs/cases/`](docs/cases/).

## Repository Structure

```
├── inference/              # Inference & evaluation (released)
├── training/               # RL training — GRPO + curriculum (released)
├── datagen/                # Data synthesis (released)
├── environment/            # Local search/browse environment (released)
└── docs/                   # Project page
```

## Quick Start — Evaluation

```bash
cd inference
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set MODEL, SERPER_KEY_ID (browser uses Jina Reader by default; set SCRAPEDO_API_KEY only if using BROWSER_PROVIDER=scrapedo)

# Start model server (SGLang/vLLM)
bash scripts/start_sglang.sh

# Run evaluation
bash scripts/run_all.sh
```

See [`inference/README.md`](inference/README.md) for detailed configuration and usage.

## Quick Start — Training

The full two-stage RL training pipeline (GRPO + TIS + difficulty-aware curriculum)
is in [`training/`](training/), and the training data is hosted on
[🤗 `LiteResearcher-Data`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data).

### Prerequisites

- **GPU** — Stage 1: 8×H20 (1 node); Stage 2: 16×H20 (2 nodes).
- **Local tool backend** — RL runs against the local search/browse environment,
  not live web. Bring up the search service (Milvus + Redis) and the browse
  service (PostgreSQL) **before** training. See [`environment/`](environment/)
  for the search backend and
  [`examples/sglang_multiturn/search_browser/tool_backend/`](training/examples/sglang_multiturn/search_browser/tool_backend/)
  for the browse backend.

### 1. Install

```bash
cd training
pip install -e .[sglang]                   # verl-based training stack
```

### 2. Configure the tool backend

```bash
cp examples/sglang_multiturn/search_browser/tool_backend/.env.example \
   examples/sglang_multiturn/search_browser/tool_backend/.env
# Edit .env: PG_* (browse DB), SUMMARY_API_*, LLM_JUDGE_API_*, optional SCRAPEDO_API_KEY

# Start the browse service (reads the .env above)
bash examples/sglang_multiturn/search_browser/tool_backend/start_browse.sh
```

### 3. Download the training data

```bash
hf download simplex-ai-inc/LiteResearcher-Data --repo-type dataset \
            --local-dir ./literesearcher_data    # 28K prompts, 19 MB
```

### 4. Stage 1 — RAG-only warmup (8×H20, 32K ctx)

```bash
export TRAIN_DATA=./literesearcher_data/stage1/train.parquet
export VAL_DATA="$TRAIN_DATA"     # no separate val bundled; verl needs a non-empty val_files
export MODEL_PATH=$(hf download simplex-ai-inc/LiteResearcher-4B-SFT \
                                --local-dir ./literesearcher_sft)
bash examples/sglang_multiturn/search_browser/stage1_rag_only.sh
```

### 5. Stage 2 — mixed curriculum (16×H20, 48K ctx)

Resume from a Stage-1 checkpoint (around step 220).

```bash
export TRAIN_DATA=./literesearcher_data/stage2/train.parquet
export VAL_DATA="$TRAIN_DATA"
export MODEL_PATH=/path/to/stage1-ckpt/global_step_220
bash examples/sglang_multiturn/search_browser/stage_2_mix_rag_on_policy_48k.sh
```

See [`training/README.md`](training/README.md) for the full reproduction recipe
(including the SFT cold-start prerequisite, environment variables, and config
knobs) and the
[dataset card](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data)
for the data schema and curriculum design.

## Release Plan

- [x] Evaluation code
- [x] Project page
- [x] Model weights — RL ([`LiteResearcher-4B`](https://huggingface.co/simplex-ai-inc/LiteResearcher-4B))
- [x] Model weights — SFT cold-start ([`LiteResearcher-4B-SFT`](https://huggingface.co/simplex-ai-inc/LiteResearcher-4B-SFT), built on [`Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)) **🆕**
- [x] Local search/browse environment setup ([`environment/`](environment/))
- [x] Search corpus — 32M records ([`LiteResearcher-Corpus`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Corpus))
- [x] Training code — GRPO + curriculum RL ([`training/`](training/))
- [x] Training data — Stage-1 & Stage-2 prompts ([`LiteResearcher-Data`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Data))
- [x] Data synthesis pipeline ([`datagen/`](datagen/))

## Acknowledgements

LiteResearcher's training stack is built on [verl](https://github.com/volcengine/verl),
ByteDance's RL training library, which we fork and extend with the multi-turn
search/browse agent loop, difficulty-aware curriculum, and local-tool reward
pipeline. We also build on [SGLang](https://github.com/sgl-project/sglang) for
rollout serving, [Qwen3](https://github.com/QwenLM/Qwen3) as the base model, and
[Milvus](https://milvus.io/) + [BGE-M3](https://huggingface.co/BAAI/bge-m3) for
the local search environment. We thank these projects and their communities.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for
development setup, pull-request guidelines, and our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Powered By

LiteResearcher is the engine behind **[lev8](https://lev8.com)**, Simplex AI's
parallel agentic search platform — frontier-grade deep research, fast and cheap
enough to run hundreds of agents per query. **Explore → [lev8.com](https://lev8.com)**

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

Released under the [Apache License 2.0](LICENSE).

