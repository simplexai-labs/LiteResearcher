# LiteResearcher — Local Search/Browse Environment

A lightweight, large-scale retrieval engine for deep-research / RAG agents, built on
[Milvus](https://milvus.io/) + [BGE-M3](https://huggingface.co/BAAI/bge-m3) hybrid search.

This is the **local tool environment** for [LiteResearcher](https://github.com/simplexai-labs/LiteResearcher):
a stable, fully-local search + browse stack built from ~32M real webpages that replaces
live-web APIs during RL training (10–46× speedup, zero marginal tool cost).

> 📦 **Corpus:** the ~32M-record search corpus (`url + title + doc`) is released at
> [`simplex-ai-inc/LiteResearcher-Corpus`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-Corpus).
> Download it and build the index with the steps below.

It does two things:

1. **Build the index** — embed a JSONL/JSON.GZ corpus with BGE-M3 (dense + sparse) and
   load it into Milvus, with optional DISKANN index for 10M+ documents.
2. **Serve search** — a FastAPI backend that returns Google-style results
   (`{link, title, snippet, score}`), suitable as the web-search tool behind a research agent.

The retrieval backend and the embedding model are decoupled through a **Redis queue**, so you
can scale the GPU embedding workers independently of the search service.

---

## Architecture

```
                ┌────────────────────────────────────────────────────────┐
   BUILD        │  data.py  ──▶  rag_core/  ──▶  BGE-M3  ──▶  Milvus       │
   (offline)    │                              (dense+sparse)  (+ optional │
                │                                               PostgreSQL)│
                └────────────────────────────────────────────────────────┘

                ┌────────────────────────────────────────────────────────┐
   SERVE        │  client ──▶ local_rag_server.py (:8018)                 │
   (online)     │                │   ▲                                    │
                │      rpush embed_queue │ blpop res:{id}                  │
                │                ▼   │                                     │
                │        Redis ──────┴──── embedding_worker.py (BGE-M3/GPU)│
                │                │                                         │
                │                └──▶ Milvus hybrid search ──▶ results     │
                └────────────────────────────────────────────────────────┘
```

## Layout

```
LiteResearcher/
├── config.py            # Index-build config (data paths, model, collection, DISKANN/SQL toggles)
├── data.py              # Index-build entrypoint: python data.py
├── rag_core/            # Index-build library (embedding / collection / data loading / optional PostgreSQL)
├── tools/
│   └── convert_serper.py    # serper/Google results → import format
├── server/              # Retrieval backend
│   ├── local_rag_server.py  # Search service (:8018), hybrid search
│   ├── embedding_worker.py  # BGE-M3 embedding worker (Redis queue consumer)
│   ├── sql_fulltext.py      # Optional: fetch full document by URL from PostgreSQL
│   ├── diskann_config.py    # Backend config (Milvus URI, collection, model, GPU, SQL)
│   ├── start.sh / stop.sh
├── milvus_config/       # Milvus single-node docker deployment (incl. DISKANN config)
└── requirements.txt
```

---

## Quick start

### 0. Install

```bash
pip install -r requirements.txt
# You also need local BGE-M3 model weights and a reachable Redis instance.
```

### 1. Start Milvus

```bash
cd milvus_config
# Edit .env and set DOCKER_VOLUME_DIRECTORY to your real data directory.
docker-compose up -d
# Milvus listens on localhost:19530
```

> DISKANN parameters (MaxDegree / SearchListSize, etc.) live under `common.DiskIndex`
> in `milvus_config/milvus.yaml`, not in the Python code.

### 2. Build the index

#### 2a. (Optional) Download the released corpus

Our released ~32M retrieval corpus is on HuggingFace; each record has `url` / `title` / `doc`:

```bash
huggingface-cli download simplex-ai-inc/LiteResearcher-Corpus \
  serper_test_text.jsonl.zst --repo-type dataset --local-dir ./corpus
zstd -d ./corpus/serper_test_text.jsonl.zst    # decompress to jsonl
```

> Each line is already `{"url": ..., "title": ..., "doc": ...}`. To import it directly,
> use `tools/convert_serper.py` to align it into the nested format below (`--text-field doc`).

The importer expects one JSON object per line (`.jsonl` or `.json.gz`), in the
**Dolma-style nested format**:

```json
{"id": "doc-1", "text": "body text…", "metadata": {"title": "Title", "url": "https://..."}}
```

| JSONL field       | → Milvus field | Notes |
|-------------------|----------------|-------|
| `metadata.title`  | `title`        | Defaults to empty string |
| `metadata.url`    | `url`          | Falls back to `id` if missing |
| `text`            | `doc` + vector source | **Required**; the line is skipped if empty |
| `id`              | internal doc_id | Defaults to `doc_{line_number}` |

> ⚠️ `title` / `url` are nested under `metadata`, not top-level; `text` is top-level.

#### 2b. (Optional) Convert serper / Google search results

If your raw data is in the flat serper format (top-level `link` / `title` / `content` / `snippet`),
align the fields first with the conversion script:

```bash
python tools/convert_serper.py raw_serper.jsonl corpus.jsonl --dedup --min-chars 50
# link → metadata.url, title → metadata.title, content → text (falls back to snippet if empty)
```

#### 2c. Configure and import

Edit `config.py`:

```python
DATA_FOLDER_PATH = "/path/to/your/corpus"   # batch mode
DATA_FILE_PATTERN = "*.jsonl"
BGE_MODEL_PATH = "/path/to/bge-m3"
DATA_COLLECTION_NAME = "litesearch"
ENABLE_DISKANN = False     # recommended True for 10M+ documents
```

Then import:

```bash
python data.py
```


### 3. Serve search

`server/` is configured independently. Edit `server/diskann_config.py` (or use environment variables):

```python
API_MILVUS_URI = "http://localhost:19530"
SEARCH_COLLECTION_NAME = "litesearch"   # must match the importer
API_BGE_MODEL_PATH = "/path/to/bge-m3"
API_DEVICE = ["cuda:0"]
```

Start (worker first, then the search service — the script orchestrates this):

```bash
cd server
REDIS_HOST=127.0.0.1 EMBED_WORKERS=1 bash start.sh
# Stop: bash stop.sh
```

### 4. Query

```bash
curl -X POST http://localhost:8018/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning algorithms", "limit": 10, "search_type": "hybrid"}'
```

```json
{
  "results": [
    {"link": "https://...", "title": "...", "snippet": "...", "score": 0.83}
  ],
  "total": 10,
  "search_type": "hybrid"
}
```

---

## API

`local_rag_server.py` (port 8018):

| Endpoint            | Description |
|---------------------|-------------|
| `POST /search`      | Single query, hybrid / dense / sparse |
| `POST /batch_search`| Batch query (multiple queries at once, shared embedding) |
| `POST /check_url`   | Check whether a single URL is in the collection |
| `POST /batch_check_url` | Batch URL existence check |
| `POST /web_parser`  | Fetch the full original document by URL (requires PostgreSQL full-text storage, see below) |
| `GET  /health`      | Health check and runtime stats |
| `GET  /stats`       | Throughput / latency stats |

`/search` request body:

```json
{
  "query": "...",
  "limit": 10,
  "search_type": "hybrid",   // hybrid | dense | sparse
  "sparse_weight": 0.7,
  "dense_weight": 1.0
}
```

---

## Full-text fetch (PostgreSQL, optional)

Vector search only returns a truncated snippet. If the research agent needs to read the
**full original document**, enable PostgreSQL full-text storage to close the loop:

```
store full text at import  ──▶  PostgreSQL  ──▶  /web_parser fetches full body by URL
```

**1) Write full text to PostgreSQL at import time** — in `config.py`:

```python
ENABLE_SQL_STORAGE = True
SQL_HOST = "localhost"; SQL_PORT = 5432
SQL_DATABASE = "postgres"; SQL_USER = "postgres"; SQL_PASSWORD = "..."
SQL_SCHEMA = "litesearch_sql"; SQL_TABLE = "documents"
```

Then run `python data.py` as usual — it writes both Milvus (vectors) and PostgreSQL (full text).

**2) Enable full-text fetch on the backend** — in `server/diskann_config.py` (or environment
variables); the SQL connection parameters must match the importer:

```python
ENABLE_SQL_FULLTEXT = True   # or export ENABLE_SQL_FULLTEXT=true
# SQL_HOST / SQL_PORT / SQL_DATABASE / SQL_USER / SQL_PASSWORD / SQL_SCHEMA / SQL_TABLE
```

**3) Fetch full text by URL:**

```bash
curl -X POST http://localhost:8018/web_parser \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.python.org/"}'
# → {"found": true, "url": "...", "title": "...", "text": "full body text..."}
```

> SQL is off by default (`ENABLE_SQL_STORAGE=False` / `ENABLE_SQL_FULLTEXT=False`), which does
> not affect pure vector search. If SQL is disabled or PostgreSQL is unreachable at startup, the
> backend degrades gracefully: `/web_parser` returns 503 while all other endpoints work normally.

---

## Configuration cheatsheet

### Index build (`config.py`)
- `ENABLE_BATCH_MODE` — batch folder vs. single file
- `ENABLE_DISKANN` — DISKANN for tens-of-millions scale (FP32 disk index); otherwise in-memory HNSW
- `ENABLE_SQL_STORAGE` — whether to store full text in PostgreSQL (for full-document fetch)
- `DATA_IMPORT_STRATEGY` — `append` / `overwrite` / `ask`
- `MULTIPROCESS_WORKERS` — number of processes for field extraction

### Serve (`server/diskann_config.py`, supports environment variables)
- `MILVUS_URI`, `SEARCH_COLLECTION`, `BGE_MODEL_PATH`, `WORKER_GPU`
- `DISKANN_SEARCH_LIST` — search accuracy/speed trade-off (50–300)
- Multi-GPU: set a different `WORKER_GPU=cuda:N` for each `embedding_worker.py` process

### Redis (search backend ↔ worker)
- `REDIS_HOST` / `REDIS_PORT` / `INPUT_QUEUE` (defaults: `127.0.0.1:6379` / `embed_queue`)

---

## Notes

- **Hybrid search** fuses BGE-M3's dense (semantic) + sparse (keyword) vectors, weighted by
  `WeightedRanker(sparse_weight, dense_weight)`.
- **DISKANN requires FP32**; non-DISKANN mode uses in-memory HNSW.
- Embedding and retrieval are decoupled via Redis: scale GPU workers independently without blocking.
