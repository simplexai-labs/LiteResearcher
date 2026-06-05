"""
RAG Core - Milvus RAG 数据构建核心模块
负责：BGE-M3 向量化、Milvus 集合管理、数据加载与插入、可选 PostgreSQL 存储。
数据格式：JSONL / JSON.GZ，字段为 url / title / doc（即 serper 风格语料）。
"""

__version__ = "1.0.0"

# 按依赖关系排序，避免循环导入
from . import config_loader
from . import utils
from . import bge_model
from . import database
from . import data_loader
from . import data_inserter
from . import config_backup
from . import sql_storage
from . import progress_manager
from . import data_processor
from . import data_sql_processor

# 导出常用函数
from .bge_model import setup_bge_model, warmup_model
from .database import setup_milvus_client, get_or_create_collection, create_data_collection
from .data_loader import load_data_corpus, load_data_corpus_with_monitoring, read_jsonl_gz_monitoring_only
from .data_inserter import insert_data_to_milvus

from .utils import setup_signal_handlers, cleanup_gpu_memory, enhanced_memory_cleanup, smart_memory_cleanup
from .config_backup import backup_config_for_import, list_config_backups
from .sql_storage import (
    insert_data_to_sql, insert_documents_to_postgres, insert_documents_to_postgres_with_analysis,
    search_documents_by_url, search_documents_fulltext,
    initialize_sql_connection_pool, close_sql_connection_pool, search_documents_by_url_fast,
)
from .progress_manager import (
    load_progress, save_progress, show_progress, validate_config,
    create_and_load_data_batch, create_and_load_data_batch_sql_only,
    show_sql_progress, get_existing_vectorized_files, start_rag_txt_monitoring,
)
from .data_processor import create_and_load_single_data
from .data_sql_processor import create_and_load_single_data_sql_only, show_sql_only_info

__all__ = [
    # 模型管理
    'setup_bge_model', 'warmup_model',
    # 数据库操作
    'setup_milvus_client', 'get_or_create_collection', 'create_data_collection',
    # 数据加载
    'load_data_corpus', 'load_data_corpus_with_monitoring', 'read_jsonl_gz_monitoring_only',
    'insert_data_to_milvus',
    # 工具函数
    'setup_signal_handlers', 'cleanup_gpu_memory', 'enhanced_memory_cleanup', 'smart_memory_cleanup',
    # 配置备份
    'backup_config_for_import', 'list_config_backups',
    # SQL 存储
    'insert_data_to_sql', 'insert_documents_to_postgres', 'insert_documents_to_postgres_with_analysis',
    'search_documents_by_url', 'search_documents_fulltext',
    'initialize_sql_connection_pool', 'close_sql_connection_pool', 'search_documents_by_url_fast',
    # 进度管理与数据处理
    'load_progress', 'save_progress', 'show_progress', 'validate_config',
    'create_and_load_data_batch', 'create_and_load_single_data',
    'show_sql_progress', 'get_existing_vectorized_files', 'start_rag_txt_monitoring',
    'create_and_load_data_batch_sql_only', 'create_and_load_single_data_sql_only', 'show_sql_only_info',
    # 模块
    'config_loader', 'utils', 'bge_model', 'database', 'data_loader',
    'data_inserter', 'config_backup', 'sql_storage', 'progress_manager', 'data_processor',
]
