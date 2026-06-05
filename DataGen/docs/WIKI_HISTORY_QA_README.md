# Wikipedia History Q&A Generation

This workflow creates Q&A pairs from historical Wikipedia revisions and adds a
time constraint to each question.

## Workflow

1. Use `scripts/wiki/sample_wiki_urls.py` to sample Wikipedia titles and create
   history URLs.
2. Use `scripts/wiki/extract_wiki_revisions.py` to fetch each history page and
   extract revision metadata.
3. Use `scripts/wiki/threaded_wiki_history_qa_generator.py` or `run_threaded_qa.sh
   --wiki-history` to fetch historical pages and generate Q&A pairs.

## Environment

```bash
export SCRAPEDO_API_KEY=...
export LLM_API_KEY=...
export LLM_MODEL=...
```

## Build Revision Inputs

```bash
python3 scripts/wiki/sample_wiki_urls.py \
  --wiki-dir wiki_en \
  --output-file sampled_wiki_history_urls.txt \
  --sample-size 10000 \
  --skip-test

python3 scripts/wiki/extract_wiki_revisions.py \
  --input-file sampled_wiki_history_urls.txt \
  --output-dir wiki_revisions \
  --workers 16 \
  --scrapedo-api-key "$SCRAPEDO_API_KEY"
```

## Generate Q&A

```bash
./run_threaded_qa.sh \
  --wiki-history \
  --wiki-revisions-dir wiki_revisions \
  --output-dir qa_outputs_wiki_history \
  --llm-workers 16 \
  --llm-api-key "$LLM_API_KEY" \
  --llm-model "$LLM_MODEL" \
  --start-index 0 \
  --end-index 1000
```

For a tiny smoke test:

```bash
./scripts/ops/test_wiki_history_qa.sh
```

## Output

Each generated JSON file is keyed by revision `oldid` and includes the revision
URL, timestamp constraint, markdown/html lengths, Q&A pairs, model name, and
generation timestamp.
