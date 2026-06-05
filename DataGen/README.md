# Direct Information-Seeking Datagen

Utilities for generating and filtering direct information-seeking question-answer
pairs from cached web pages, Wikipedia snapshots, Wikipedia revision histories,
and BBC news data.

The generated examples are intended to be self-contained: each question should
carry enough context to be answered without seeing the source document.

## Features

- Convert cached HTML to markdown-like text.
- Generate factual Q&A pairs with OpenRouter or any OpenAI-compatible chat API.
- Run a threaded producer/consumer pipeline against an HTML record service,
  local Wikipedia Arrow files, or BBC parquet files.
- Filter Q&A pairs with an LLM quality judge.
- Optionally verify whether answers are still present in web search or current
  Wikipedia pages.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/directqa/` | Reusable Python package code, including the mmap-backed HTML cache. |
| `scripts/generation/` | Q&A generation entrypoints for HTMLStorage, HTTP, Wiki Arrow, and BBC data. |
| `scripts/filtering/` | LLM quality filtering and optional search/current-Wikipedia verification. |
| `scripts/wiki/` | Wikipedia history sampling, revision extraction, generation, and diagnostics. |
| `scripts/diagnostics/` | Small inspection and conversion utilities. |
| `scripts/ops/` | Operational shell helpers for long-running jobs. |
| `docs/` | Workflow-specific documentation. |
| `examples/` | Tiny example inputs. |
| `run_threaded_qa.sh` | Shell wrapper for the threaded generator. |

Generated datasets, logs, checkpoints, storage shards, and local installers are
ignored by `.gitignore` and should not be committed.

## Installation

```bash
git clone <your-repo-url>
cd direct-information-seeking-datagen-full
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example environment file and fill only the services you use:

```bash
cp .env.example .env
```

This project does not load `.env` automatically. Export values in your shell or
pass them as CLI arguments.

## Quick Checks

Compile all Python files:

```bash
python3 -m py_compile $(find src scripts -name "*.py" -type f)
```

Inspect a cached HTMLStorage record:

```bash
python3 scripts/diagnostics/read_html_storage.py \
  --storage-dir /path/to/storage_1234 \
  --url "https://example.com/page" \
  --output read_html.md
```

## Generate Q&A from HTMLStorage

Create a checkpoint JSON containing URLs. Supported top-level shapes are:

- a list of objects with `url` or `link`
- an object with one of `urls`, `links`, `data`, or `results`

Then run:

```bash
export OPENROUTER_API_KEY=...

python3 scripts/generation/generate_qa_pairs.py \
  --checkpoint-json checkpoint.json \
  --storage-dir /path/to/storage_1234 \
  --output-dir qa_outputs \
  --num-urls 10
```

Useful options:

- `--model-name`: defaults to `OPENROUTER_MODEL` or `anthropic/claude-3.5-sonnet`
- `--api-url`: defaults to OpenRouter's chat completions endpoint
- `--max-markdown-chars`: truncates long pages before LLM calls

## Threaded Batch Generation

HTTP record service mode:

```bash
./run_threaded_qa.sh \
  --html-host 127.0.0.1 \
  --html-port 8522 \
  --start-index 0 \
  --end-index 1000 \
  --output-dir qa_outputs \
  --llm-api-key "$LLM_API_KEY" \
  --llm-model "$LLM_MODEL" \
  --llm-workers 8 \
  --extra "--skip-existing"
```

Wiki Arrow mode:

```bash
./run_threaded_qa.sh --wiki \
  --wiki-dir wiki_en \
  --start-index 0 \
  --end-index 10000 \
  --output-dir qa_outputs_wiki \
  --llm-api-key "$LLM_API_KEY" \
  --llm-model "$LLM_MODEL"
```

BBC parquet mode:

```bash
./run_threaded_qa.sh --bbc \
  --bbc-dir /path/to/bbc_news_alltime \
  --bbc-limit 10000 \
  --output-dir qa_outputs_bbc \
  --llm-api-key "$LLM_API_KEY" \
  --llm-model "$LLM_MODEL"
```

## Filtering

LLM quality filter:

```bash
python3 scripts/filtering/filter_qa_pairs.py \
  --input_dir qa_outputs \
  --output_dir output_filtered \
  --workers 8 \
  --llm-api-key "$LLM_API_KEY" \
  --llm-model "$LLM_MODEL"
```

Search-based verification with Serper:

```bash
export SERPER_API_KEY=...
python3 scripts/filtering/serper_filter.py \
  --input output_filtered/positive.jsonl \
  --output-dir serper_filter \
  --producers 4 \
  --consumers 8
```

Current-Wikipedia verification:

```bash
export SCRAPEDO_API_KEY=...
python3 scripts/filtering/filter_current_wiki.py \
  --input serper_filter/positive.jsonl \
  --qa-dir qa_outputs_wiki_history \
  --output-dir output_filtered_current_wiki \
  --workers 8
```

## Output Format

Each generated JSON file contains fields like:

```json
{
  "url": "https://example.com/page",
  "url_hash": "78dd80224050b1058156889a517034a2",
  "markdown_length": 9096,
  "html_length": 37659,
  "qa_pairs": [
    {
      "question": "What was Example Corp.'s revenue in fiscal year 2025?",
      "answer": "$10.2 billion"
    }
  ],
  "total_pairs": 1,
  "model_name": "anthropic/claude-3.5-sonnet",
  "generation_timestamp": "2026-06-04T12:00:00"
}
```

## Data and Secrets

Do not commit:

- API keys or `.env`
- raw HTML storage directories
- `.jsonl`, Arrow, parquet, or generated output directories
- logs, checkpoints, and local installers

Before publishing, choose and add a license file that matches your intended
reuse policy.
