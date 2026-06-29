#!/usr/bin/env python3
"""
🎯 LiteResearcher 检索后端配置
被 local_rag_server.py（检索服务）和 embedding_worker.py（向量化 worker）共享。
独立于数据导入端的 config.py。

支持环境变量覆盖，便于部署：
    MILVUS_URI, SEARCH_COLLECTION, BGE_MODEL_PATH, WORKER_GPU
"""

import os

# ============================= 服务端口配置 =============================

DISKANN_RAG_PORT = 8018          # 检索服务端口
DISKANN_EMBEDDING_PORT = 8028    # 兼容旧引用（当前架构走 Redis 队列，不再使用 HTTP embedding 端口）

# ============================= BGE 模型配置 =============================

API_BGE_MODEL_PATH = os.environ.get("BGE_MODEL_PATH", "/path/to/bge-m3")  # 本地 BGE-M3 模型路径
API_BGE_MAX_LENGTH = 128         # Embedding 最大输入长度
API_BATCH_SIZE = 32              # 批次大小

# 向量化 worker 使用的 GPU（每个 worker 进程取一个）
# 多卡部署时，可为每个 worker 进程设置 WORKER_GPU=cuda:N
API_DEVICE = ["cuda:0"]

# DISKANN 必须使用 FP32 精度
API_USE_FP16 = False

# ============================= Milvus 连接配置 =============================

# 本地: http://localhost:19530
API_MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")

# 检索的集合名（需与导入端 config.py 的 SEARCH_COLLECTION_NAME 一致）
SEARCH_COLLECTION_NAME = os.environ.get("SEARCH_COLLECTION", "litesearch")

# ============================= DISKANN 搜索配置 =============================

DISKANN_SEARCH_LIST = 100        # 搜索列表大小（50-300，越大越精确但越慢）
DISKANN_DEFAULT_LIMIT = 10       # 默认返回结果数

DEFAULT_SPARSE_WEIGHT = 0.7      # 默认稀疏向量权重
DEFAULT_DENSE_WEIGHT = 1.0       # 默认密集向量权重

# ============================= 服务器配置 =============================

API_HOST = "0.0.0.0"             # 监听地址
UVICORN_WORKERS = 8              # 检索服务 worker 数量
EMBEDDING_WORKERS = 1            # 建议每张 GPU 启动一个 embedding_worker.py 进程

# ============================= 向量化批处理配置 =============================

MAX_BATCH_SIZE = 128             # BGE-M3 单次最大 batch
MAX_WAIT_TIME = 0.3              # 攒批最大等待时间（秒）
MIN_BATCH_SIZE = 8              # 触发处理的最小 batch

# ============================= PostgreSQL 全文取数配置（可选）=============================
# 仅当导入时启用了 ENABLE_SQL_STORAGE（把全文写入 PostgreSQL）后才需要。
# 检索后端用它支撑 /web_parser、/get_document 等"按 URL 取整篇原文"接口。
# 列与导入端 config.py 的 SQL_SCHEMA/SQL_TABLE 必须一致（表含 url/title/text）。
ENABLE_SQL_FULLTEXT = os.environ.get("ENABLE_SQL_FULLTEXT", "false").lower() == "true"
SQL_HOST = os.environ.get("SQL_HOST", "localhost")
SQL_PORT = int(os.environ.get("SQL_PORT", 5432))
SQL_DATABASE = os.environ.get("SQL_DATABASE", "postgres")
SQL_USER = os.environ.get("SQL_USER", "postgres")
SQL_PASSWORD = os.environ.get("SQL_PASSWORD", "your_postgres_password")
SQL_SCHEMA = os.environ.get("SQL_SCHEMA", "litesearch_sql")
SQL_TABLE = os.environ.get("SQL_TABLE", "documents")
SQL_POOL_MIN = 1                 # 连接池最小连接数
SQL_POOL_MAX = 8                 # 连接池最大连接数
