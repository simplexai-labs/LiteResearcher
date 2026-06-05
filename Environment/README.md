# LiteResearcher — Local Search/Browse Environment

A lightweight, large-scale retrieval engine for deep-research / RAG agents, built on
[Milvus](https://milvus.io/) + [BGE-M3](https://huggingface.co/BAAI/bge-m3) hybrid search.

This is the **local tool environment** for [LiteResearcher](https://github.com/simplex-ai-inc/LiteResearcher):
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
├── config.py            # 数据导入端配置（数据路径、模型、集合名、DISKANN/SQL 开关）
├── data.py              # 数据导入入口：python data.py
├── rag_core/            # 索引构建库（向量化 / 建集合 / 数据加载 / 可选 PostgreSQL）
├── tools/
│   └── convert_serper.py    # serper/Google 结果 → 导入格式
├── server/              # 检索后端
│   ├── local_rag_server.py  # 检索服务 (:8018)，混合搜索
│   ├── embedding_worker.py  # BGE-M3 向量化 worker（Redis 队列消费者）
│   ├── sql_fulltext.py      # 可选：从 PostgreSQL 按 URL 取整篇全文
│   ├── diskann_config.py    # 后端配置（Milvus 地址、集合、模型、GPU、SQL）
│   ├── start.sh / stop.sh
├── milvus_config/       # Milvus 单机 docker 部署（含 DISKANN 配置）
└── requirements.txt
```

---

## Quick start

### 0. Install

```bash
pip install -r requirements.txt
# 另需本地的 BGE-M3 模型权重，以及一个可达的 Redis 实例
```

### 1. Start Milvus

```bash
cd milvus_config
# 编辑 .env，把 DOCKER_VOLUME_DIRECTORY 改为真实数据目录
docker-compose up -d
# Milvus 监听 localhost:19530
```

> DISKANN 参数（MaxDegree / SearchListSize 等）在 `milvus_config/milvus.yaml`
> 的 `common.DiskIndex` 下设置，不在 Python 代码里。

### 2. Build the index

#### 2a.（可选）下载发布的语料

我们发布的 ~32M 检索语料在 HuggingFace，每条含 `url` / `title` / `doc`：

```bash
huggingface-cli download simplex-ai-inc/LiteResearcher-Corpus \
  serper_test_text.jsonl.zst --repo-type dataset --local-dir ./corpus
zstd -d ./corpus/serper_test_text.jsonl.zst    # 解压成 jsonl
```

> 该文件每行已是 `{"url": ..., "title": ..., "doc": ...}`。若直接用于导入，可用
> `tools/convert_serper.py` 对齐成下方的嵌套格式（`--text-field doc`）。

导入端期望每行一个 JSON 对象（`.jsonl` 或 `.json.gz`），字段为 **Dolma 风格的嵌套格式**：

```json
{"id": "doc-1", "text": "正文内容……", "metadata": {"title": "标题", "url": "https://..."}}
```

| JSONL 字段        | → Milvus 字段 | 说明 |
|-------------------|---------------|------|
| `metadata.title`  | `title`       | 缺省为空串 |
| `metadata.url`    | `url`         | 缺省回退到 `id` |
| `text`            | `doc` + 向量源 | **必填**，为空则整行跳过 |
| `id`              | 内部 doc_id   | 缺省为 `doc_{行号}` |

> ⚠️ `title` / `url` 嵌在 `metadata` 里，不是顶层字段；`text` 在顶层。

#### 2b.（可选）转换 serper / Google 搜索结果

如果你的原始数据是 serper 扁平格式（顶层 `link` / `title` / `content` / `snippet`），
先用转换脚本对齐字段：

```bash
python tools/convert_serper.py raw_serper.jsonl corpus.jsonl --dedup --min-chars 50
# link → metadata.url, title → metadata.title, content → text（空则回退 snippet）
```

#### 2c. 配置并导入

编辑 `config.py`：

```python
DATA_FOLDER_PATH = "/path/to/your/corpus"   # 批量模式
DATA_FILE_PATTERN = "*.jsonl"
BGE_MODEL_PATH = "/path/to/bge-m3"
DATA_COLLECTION_NAME = "litesearch"
ENABLE_DISKANN = False     # 10M+ 文档建议 True
```

然后导入：

```bash
python data.py
```


### 3. Serve search

`server/` 是独立配置的。编辑 `server/diskann_config.py`（或用环境变量）：

```python
API_MILVUS_URI = "http://localhost:19530"
SEARCH_COLLECTION_NAME = "litesearch"   # 与导入端一致
API_BGE_MODEL_PATH = "/path/to/bge-m3"
API_DEVICE = ["cuda:0"]
```

启动（先 worker 后检索服务，脚本已编排）：

```bash
cd server
REDIS_HOST=127.0.0.1 EMBED_WORKERS=1 bash start.sh
# 停止：bash stop.sh
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

| Endpoint            | 说明 |
|---------------------|------|
| `POST /search`      | 单条查询，hybrid / dense / sparse |
| `POST /batch_search`| 批量查询（一次多个 query，共享 embedding） |
| `POST /check_url`   | 检查单个 URL 是否在集合中 |
| `POST /batch_check_url` | 批量 URL 存在性检查 |
| `POST /web_parser`  | 按 URL 取整篇原文（需 PostgreSQL 全文存储，见下文） |
| `GET  /health`      | 健康检查与运行统计 |
| `GET  /stats`       | 吞吐 / 延迟统计 |

`/search` 请求体：

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

## 全文取数（PostgreSQL，可选）

向量检索只返回截断的 snippet。如果研究 agent 需要读**整篇原文**，可启用 PostgreSQL
全文存储，形成闭环：

```
导入时存全文  ──▶  PostgreSQL  ──▶  /web_parser 按 URL 取整篇正文
```

**1) 导入时把全文写入 PostgreSQL** —— 在 `config.py`：

```python
ENABLE_SQL_STORAGE = True
SQL_HOST = "localhost"; SQL_PORT = 5432
SQL_DATABASE = "postgres"; SQL_USER = "postgres"; SQL_PASSWORD = "..."
SQL_SCHEMA = "litesearch_sql"; SQL_TABLE = "documents"
```

然后正常 `python data.py`，会同时写 Milvus（向量）和 PostgreSQL（全文）。

**2) 检索后端开启全文取数** —— 在 `server/diskann_config.py`（或环境变量），
SQL 连接参数需与导入端一致：

```python
ENABLE_SQL_FULLTEXT = True   # 或 export ENABLE_SQL_FULLTEXT=true
# SQL_HOST / SQL_PORT / SQL_DATABASE / SQL_USER / SQL_PASSWORD / SQL_SCHEMA / SQL_TABLE
```

**3) 按 URL 取全文：**

```bash
curl -X POST http://localhost:8018/web_parser \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.python.org/"}'
# → {"found": true, "url": "...", "title": "...", "text": "整篇正文..."}
```

> SQL 默认关闭（`ENABLE_SQL_STORAGE=False` / `ENABLE_SQL_FULLTEXT=False`），不影响纯
> 向量检索。后端启动时若 SQL 未启用或 PostgreSQL 不可达，会自动退化，`/web_parser`
> 返回 503，其余接口正常。

---

## Configuration cheatsheet

### Index 构建 (`config.py`)
- `ENABLE_BATCH_MODE` — 批量文件夹 vs 单文件
- `ENABLE_DISKANN` — 千万级数据用 DISKANN（FP32 磁盘索引）；否则内存 HNSW
- `ENABLE_SQL_STORAGE` — 是否把全文存入 PostgreSQL（用于取整篇文档）
- `DATA_IMPORT_STRATEGY` — `append` / `overwrite` / `ask`
- `MULTIPROCESS_WORKERS` — 字段抽取多进程数

### Serve (`server/diskann_config.py`，支持环境变量)
- `MILVUS_URI`, `SEARCH_COLLECTION`, `BGE_MODEL_PATH`, `WORKER_GPU`
- `DISKANN_SEARCH_LIST` — 搜索精度/速度权衡（50–300）
- 多卡部署：为每个 `embedding_worker.py` 进程设置不同的 `WORKER_GPU=cuda:N`

### Redis（检索后端 ↔ worker）
- `REDIS_HOST` / `REDIS_PORT` / `INPUT_QUEUE`（默认 `127.0.0.1:6379` / `embed_queue`）

---

## Notes

- **Hybrid search** 融合 BGE-M3 的 dense（语义）+ sparse（关键词）向量，由
  `WeightedRanker(sparse_weight, dense_weight)` 加权。
- **DISKANN 要求 FP32**；非 DISKANN 模式使用内存 HNSW。
- Embedding 与检索通过 Redis 解耦：可独立扩容 GPU worker，互不阻塞。
