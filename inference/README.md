# LiteResearcher

An open-source deep research agent for evaluating LLMs on complex question answering. Uses a ReAct (Reasoning + Acting) loop with web search and browsing tools.

## Architecture

```
┌───────────────────────────────────────────────┐
│              run_inference.py                  │
│         (parallel evaluation runner)           │
│                                               │
│          ┌────────────────────┐               │
│          │     ReActAgent     │               │
│          │  (agent.py)        │               │
│          └─────────┬──────────┘               │
│                    │                          │
│          ┌─────────▼──────────┐               │
│          │    LLM Server      │               │
│          │  (SGLang / vLLM)   │               │
│          └─────────┬──────────┘               │
│                    │                          │
│         ┌──────────┼──────────┐               │
│         │                     │               │
│   ┌─────▼─────┐        ┌─────▼─────┐         │
│   │  search    │        │  visit    │         │
│   │  server    │        │  server   │         │
│   └─────┬─────┘        └─────┬─────┘         │
│         │                     │               │
│    Serper API           Jina Reader /         │
│                        ScrapeDo + LLM Summary │
└───────────────────────────────────────────────┘
```

## Project Structure

```
├── src/
│   ├── agent.py           # ReAct agent (reasoning + tool calling)
│   ├── prompts.py         # System and judge prompts
│   ├── run_inference.py   # Parallel evaluation runner
│   ├── search_server.py   # Search service (Google Serper)
│   └── browser_server.py  # Browser service (fetch + LLM summarize)
├── scripts/
│   ├── run_all.sh         # One-click: start servers + run eval
│   ├── start_servers.sh   # Start search & browser servers
│   ├── run_inference.sh   # Run evaluation only
│   └── start_sglang.sh   # Start SGLang model server
├── data/
│   └── example.jsonl      # Example dataset
├── .env.example
└── requirements.txt
```

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: set MODEL, SERPER_KEY_ID (browser uses Jina Reader by default; set SCRAPEDO_API_KEY only if BROWSER_PROVIDER=scrapedo)

# 3. Start model server
bash scripts/start_sglang.sh

# 4. Run evaluation
bash scripts/run_all.sh
```

Or step by step:
```bash
bash scripts/start_servers.sh        # Terminal 1: search + browser servers
bash scripts/run_inference.sh        # Terminal 2: run evaluation
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | — | Model path or name (required) |
| `SGLANG_API_BASE` | `http://127.0.0.1:6001/v1` | LLM server endpoint |
| `SERPER_KEY_ID` | — | [Serper](https://serper.dev) API key for search |
| `BROWSER_PROVIDER` | `jina` | Page-fetch backend: `jina` or `scrapedo` |
| `JINA_API_KEY` | — | [Jina Reader](https://jina.ai/reader) API key (optional, raises rate limits) |
| `SCRAPEDO_API_KEY` | — | [ScrapeDo](https://scrape.do) API key (required only if `BROWSER_PROVIDER=scrapedo`) |
| `SUMMARY_MODEL_NAME` | — | Model for webpage summarization |
| `TEMPERATURE` | `0.6` | Sampling temperature |
| `MAX_LLM_CALL_PER_RUN` | `100` | Max reasoning turns per question |
| `MAIN_MAX_MODEL_LEN` | `90000` | Max context length (tokens) |
| `MAX_TIMEOUT_SECONDS` | `9000` | Per-question timeout |
| `MAX_WORKERS` | `20` | Parallel inference workers |
| `ROLL_OUT_COUNT` | `1` | Rollouts per question (pass@k) |

## Dataset Format

JSONL with one question per line:

```json
{"question": "What year was the first Nobel Prize in Physics awarded?", "answer": "1901"}
```

## Output Format

```json
{
  "metadata": {
    "model": "...",
    "judge_summary": {"accuracy": 0.85, "correct": 17, "total": 20}
  },
  "records": [
    {
      "question": "...",
      "prediction": "...",
      "answer": "...",
      "judge": {"correct": true, "verdict": "CORRECT"},
      "total_time": 45.2,
      "turn_times": [{"turn": 1, "llm_time": 3.2, "tool_time": 5.1, "action": "tool_call"}]
    }
  ]
}
```

## License

Apache 2.0
