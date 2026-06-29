"""
核心数据处理逻辑
单文件数据加载的主要流程（JSONL / JSON.GZ，字段 url/title/doc）
"""

import os
from .config_loader import (
    DATA_COLLECTION_NAME, DOC_STORAGE_MAX_CHARS, DOC_STORAGE_MAX_TOKENS,
    DOC_USE_TOKEN_LIMIT, DOC_EMBEDDING_MAX_TOKENS,
)
from .progress_manager import load_progress, save_progress
# 直接导入模块避免循环导入
from .bge_model import setup_bge_model, warmup_model
from .database import setup_milvus_client, create_data_collection
from .data_loader import load_data_corpus_with_monitoring
from .data_inserter import insert_data_to_milvus
from .utils import setup_signal_handlers, cleanup_gpu_memory
from .sql_storage import insert_data_to_sql
from .database import handle_existing_collection


def create_and_load_single_data(file_path, import_strategy=None, enable_sql_storage=None):
    """
    创建集合并加载单个数据文件

    Args:
        file_path: 数据文件路径（.json / .jsonl / .json.gz）
        import_strategy: 导入策略
        enable_sql_storage: 是否启用SQL存储，None时使用配置文件设置

    Returns:
        bool: 处理是否成功
    """
    print(f"\n🚀 开始处理文件: {os.path.basename(file_path)}")

    # 记录处理开始
    progress = load_progress(DATA_COLLECTION_NAME)
    current_file = os.path.abspath(file_path)

    if current_file in progress["processed_files"]:
        print(f"⚠️  文件 {file_path} 已经处理过，跳过...")
        return True

    try:
        # 设置信号处理器
        setup_signal_handlers()

        # 1. 连接Milvus数据库
        print("🔗 连接Milvus数据库...")
        setup_milvus_client()

        # 2. 检查集合存在状态，确定导入策略
        final_strategy = handle_existing_collection(DATA_COLLECTION_NAME, import_strategy)

        if final_strategy == "append":
            print(f"📝 将追加数据到现有集合 '{DATA_COLLECTION_NAME}'")
        elif final_strategy == "overwrite":
            print(f"🗑️  将覆盖现有集合 '{DATA_COLLECTION_NAME}'")
        elif final_strategy == "create_new":
            print(f"🆕 将创建新集合 '{DATA_COLLECTION_NAME}'")

        # 3. 初始化BGE模型
        print("🤖 初始化BGE-M3模型...")
        embedding_function, dense_dim = setup_bge_model()

        print("🔥 预热模型...")
        warmup_success = warmup_model(embedding_function)
        if not warmup_success:
            print("⚠️ 模型预热失败，但继续数据加载...")

        # 4. 加载数据
        print(f"📖 加载数据文件: {file_path}")
        print("🔍 使用带监控的数据加载方式...")
        data = load_data_corpus_with_monitoring(file_path, return_data=True)
        print(f"✅ 带监控加载完成: {len(data):,} 条记录")

        if not data:
            print("❌ 没有有效数据可加载")
            return False
        print(f"📐 Dense向量维度: {dense_dim}")

        # 5. 插入完整原始数据到PostgreSQL（可选）
        sql_success = insert_data_to_sql(data, enable_sql_storage)
        if not sql_success:
            print("⚠️  PostgreSQL插入失败，但继续Milvus数据加载...")

        # 6. 创建/获取Milvus集合
        print("🏗️  使用完整文档存储架构创建/获取集合（title+doc+向量）...")
        collection, is_new_collection = create_data_collection(dense_dim, DATA_COLLECTION_NAME, final_strategy)

        # 7. 插入数据到Milvus
        print("💾 开始向量化和数据插入到Milvus...")
        success = insert_data_to_milvus(collection, data, embedding_function)

        if success:
            collection.flush()
            print("✅ 完整文档数据加载完成！")
            print(f"   📊 Milvus记录数: {collection.num_entities:,}")
            if DOC_USE_TOKEN_LIMIT:
                print(f"   📄 完整文档模式: 已启用（title + doc前{DOC_STORAGE_MAX_TOKENS} tokens）")
            else:
                print(f"   📄 完整文档模式: 已启用（title + doc前{DOC_STORAGE_MAX_CHARS}字符）")
            print(f"   🔤 Embedding长度: 最大{DOC_EMBEDDING_MAX_TOKENS} tokens")

            # 记录成功处理的文件
            progress["processed_files"].append(current_file)
            save_progress(DATA_COLLECTION_NAME, progress["processed_files"])
            return True
        else:
            print("❌ 完整文档数据加载失败")
            return False

    except Exception as e:
        print(f"❌ 数据加载过程中发生错误: {e}")
        return False
    finally:
        print("🔄 清理资源...")

        if 'data' in locals():
            del data
        if 'embedding_function' in locals():
            del embedding_function
        if 'collection' in locals():
            del collection

        import gc
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                print(f"  回收了 {collected} 个对象")

        cleanup_gpu_memory()

        try:
            import ctypes
            import sys
            if sys.platform.startswith('linux'):
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    if hasattr(libc, 'malloc_trim'):
                        libc.malloc_trim(0)
                except Exception:
                    pass
        except ImportError:
            pass

        print("✅ 资源清理完成")
