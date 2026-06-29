# Wiki Q&A Generation

Use `scripts/generation/threaded_qa_batch_generator.py` or the
`run_threaded_qa.sh` wrapper to generate Q&A pairs from local Wikipedia Arrow
files.

## Requirements

- Wikipedia Arrow files in a directory such as `wiki_en/`
- An OpenAI-compatible LLM server
- `LLM_API_KEY` and `LLM_MODEL` exported, or passed as CLI arguments

## Example

```bash
export LLM_API_KEY=...
export LLM_MODEL=...

./run_threaded_qa.sh --wiki \
  --wiki-dir wiki_en \
  --start-index 0 \
  --end-index 100000 \
  --output-dir qa_outputs_wiki \
  --llm-workers 16 \
  --skip-existing
```

Direct Python invocation:

```bash
python3 scripts/generation/threaded_qa_batch_generator.py \
  --wiki \
  --wiki-dir wiki_en \
  --start-index 0 \
  --end-index 100000 \
  --output-dir qa_outputs_wiki \
  --llm-api-key "$LLM_API_KEY" \
  --llm-model-name "$LLM_MODEL" \
  --llm-workers 16 \
  --skip-existing
```

Use `--resume qa_outputs_wiki` to skip URLs already present in an existing
output directory.
