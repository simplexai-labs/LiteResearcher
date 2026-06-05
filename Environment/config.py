"""
LiteResearcher - 数据库构建配置
所有配置集中在此文件，修改后运行 `python data.py` 即可导入数据。

语料格式（JSONL / JSON.GZ，每行一个 JSON 对象）：
    {"id": "...", "text": "正文", "metadata": {"title": "标题", "url": "链接"}}
即 serper 风格的 url / title / doc 语料。
"""

# ========================================
# 🗂️ 数据源配置
# ========================================
ENABLE_BATCH_MODE = True  # True=批量处理文件夹, False=单文件处理

# 批量模式：处理文件夹下匹配 pattern 的所有文件
DATA_FOLDER_PATH = "/path/to/your/corpus"
DATA_FILE_PATTERN = "*.jsonl"

# 单文件模式
DATA_FILE_PATH = "/path/to/your/corpus/part-0000.jsonl"

# ========================================
# 🤖 模型和GPU配置
# ========================================
BGE_MODEL_PATH = "/path/to/bge-m3"   # 本地 BGE-M3 模型路径
BGE_MAX_LENGTH = 1024
BATCH_SIZE = 2000

# 向量化使用的 GPU（可多卡并行）
DEVICE = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
USE_FP16 = True   # 向量化计算精度（DISKANN 存储仍为 FP32）

# ========================================
# 📄 文档处理配置
# ========================================
DOC_EMBEDDING_MAX_TOKENS = 500   # embedding 输入最大 tokens
DOC_STORAGE_MAX_CHARS = 200      # Milvus doc 字段最大字符数（向后兼容）
DOC_STORAGE_MAX_TOKENS = 50      # Milvus doc 字段最大 token 数
DOC_USE_TOKEN_LIMIT = True       # True=按 token 限制, False=按字符限制

# 多进程字段抽取（突破 GIL，适合 CPU 密集型任务）
ENABLE_MULTIPROCESS_FIELD_EXTRACTION = True
MULTIPROCESS_WORKERS = 16        # 通常为 CPU 核心数的 1/4 ~ 1/2

# ========================================
# 🗄️ Milvus 向量数据库配置
# ========================================
MILVUS_URI = "http://localhost:19530"

DATA_COLLECTION_NAME = "litesearch"    # 写入的集合名
SEARCH_COLLECTION_NAME = "litesearch"  # 检索的集合名
DB_MAX_URL_LENGTH = 1024               # url 字段最大长度

# 导入策略：ask（询问）/ append（追加）/ overwrite（覆盖）
DATA_IMPORT_STRATEGY = "append"

# ========================================
# 🚀 DISKANN 索引配置
# ========================================
# 千万级以上数据强烈推荐启用。具体参数（MaxDegree、SearchListSize 等）
# 需在 milvus_config/milvus.yaml -> common.DiskIndex 中设置。
# 启用后存储精度强制为 FP32；关闭则使用内存 HNSW 索引（百万级）。
ENABLE_DISKANN = False

# ========================================
# 🔍 搜索功能配置
# ========================================
DEFAULT_SEARCH_LIMIT = 10
DEFAULT_SPARSE_WEIGHT = 0.7
DEFAULT_DENSE_WEIGHT = 1.0

ENABLE_PERFORMANCE_MONITORING = True
SLOW_QUERY_THRESHOLD = 1.0
ENSURE_EXACT_LIMIT = True
MAX_SEARCH_MULTIPLIER = 3
DISPLAY_MAX_TOKENS = 200

# ========================================
# 📊 PostgreSQL 存储配置（可选）
# ========================================
# 启用后会把完整文档正文存入 PostgreSQL，供 web_parser 等取全文使用。
ENABLE_SQL_STORAGE = False
SQL_HOST = "localhost"
SQL_PORT = 5432
SQL_DATABASE = "postgres"
SQL_USER = "postgres"
SQL_PASSWORD = "your_postgres_password"

SQL_SCHEMA = "litesearch_sql"
SQL_TABLE = "documents"
SQL_URL_MAX_LENGTH = 1024

SQL_BATCH_SIZE = 30000
SQL_ENABLE_FULL_TEXT_SEARCH = False
SQL_ANALYSIS_MULTITHREAD_THRESHOLD = 10000
SQL_ENABLE_DATA_QUALITY_CHECK = True
SQL_USE_COPY_INSERT = False

SQL_ENABLE_MULTITHREAD_DEDUP = True
SQL_DEDUP_THREAD_COUNT = 32
