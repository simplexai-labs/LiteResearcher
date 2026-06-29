"""
配置加载器 - 从外部config.py加载配置
"""

import sys
import os

# 添加父目录到路径，以便导入外部的config.py
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    # 导入外部config.py中的所有配置
    from config import *
except ImportError as e:
    raise ImportError(f"无法加载外部config.py配置文件: {e}")

# 验证必要的配置项
_required_configs = [
    'ENABLE_BATCH_MODE', 'DATA_FILE_PATH', 'DATA_FOLDER_PATH', 'DATA_FILE_PATTERN', 'BGE_MODEL_PATH', 'DATA_COLLECTION_NAME', 'SEARCH_COLLECTION_NAME',
    'BATCH_SIZE', 'DEVICE', 'USE_FP16',
    'ENABLE_MULTIPROCESS_FIELD_EXTRACTION', 'MULTIPROCESS_WORKERS', 'DATA_IMPORT_STRATEGY',
    'DEFAULT_SEARCH_LIMIT', 'DEFAULT_SPARSE_WEIGHT', 'DEFAULT_DENSE_WEIGHT',
    'ENABLE_SQL_STORAGE', 'SQL_HOST', 'SQL_PORT', 'SQL_DATABASE', 'SQL_USER', 'SQL_PASSWORD',
    'SQL_SCHEMA', 'SQL_TABLE', 'SQL_BATCH_SIZE',
    'DOC_EMBEDDING_MAX_TOKENS', 'DOC_STORAGE_MAX_CHARS', 'DOC_STORAGE_MAX_TOKENS', 'DOC_USE_TOKEN_LIMIT',
    # DISKANN索引配置（参数在milvus.yaml中设置）
    'ENABLE_DISKANN'
]

for config_name in _required_configs:
    if config_name not in globals():
        raise ValueError(f"配置文件中缺少必要的配置项: {config_name}")

# 重新导出所有配置，供其他模块使用
__all__ = [name for name in globals() if not name.startswith('_')]
