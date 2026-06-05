"""
纯SQL数据处理器 - 只处理PostgreSQL插入，不涉及Milvus操作
从data_processor.py改编，移除所有向量化和Milvus相关功能
"""

import os
from .config_loader import DATA_COLLECTION_NAME, ENABLE_SQL_STORAGE, SQL_SCHEMA
from .progress_manager import load_sql_progress, save_sql_progress
from .data_loader import load_data_corpus_with_monitoring
from .utils import setup_signal_handlers, cleanup_gpu_memory
from .sql_storage import insert_data_to_sql


def create_and_load_single_data_sql_only(file_path, enable_sql_storage=None):
    """
    纯SQL版本：只加载数据并插入到PostgreSQL，不进行向量化和Milvus操作
    
    Args:
        file_path: 数据文件路径
        enable_sql_storage: 是否启用SQL存储，None时使用配置文件设置
    
    Returns:
        bool: 处理是否成功
    """
    print(f"\n🚀 开始处理文件（纯SQL版本）: {os.path.basename(file_path)}")
    
    # 检查是否是tail压缩模式
    filename = os.path.basename(file_path)
    is_tail_compression = filename.startswith("cc_en_tail-") and filename.endswith(".json.gz")
    current_file = os.path.abspath(file_path)
    
    # 加载SQL进度（无论是否tail压缩模式都需要）
    sql_progress = load_sql_progress(SQL_SCHEMA)
    
    if is_tail_compression:
        print(f"🔥 Tail压缩模式：强制处理文件，跳过重复检查")
    else:
        # 只有非tail压缩模式才检查重复
        if current_file in sql_progress["processed_files"]:
            print(f"⚠️  文件 {file_path} 已经SQL处理过，跳过...")
            return True
    
    try:
        # 设置信号处理器
        setup_signal_handlers()
        
        # 检查SQL存储是否启用
        should_enable_sql = enable_sql_storage if enable_sql_storage is not None else ENABLE_SQL_STORAGE
        if not should_enable_sql:
            print("❌ SQL存储未启用，纯SQL版本需要启用SQL存储才能工作")
            return False
        
        print("💾 纯SQL模式：只进行数据加载和PostgreSQL插入")
        print("📝 跳过：Milvus连接、向量化、embedding等操作")
        
        # 1. 加载数据
        print(f"📖 加载数据文件: {file_path}")
        print("🔍 使用带监控的数据加载方式...")
        data = load_data_corpus_with_monitoring(file_path, return_data=True)
        print(f"✅ 带监控加载完成: {len(data):,} 条记录")
        
        if not data:
            print("❌ 没有有效数据可加载")
            return False
        
        # 2. 插入完整原始数据到PostgreSQL
        print("💾 开始插入数据到PostgreSQL...")
        sql_success = insert_data_to_sql(data, enable_sql_storage)
        
        if sql_success:
            print("✅ 纯SQL数据加载完成！")
            print(f"   📊 处理文档数: {len(data):,}")
            print(f"   💾 数据已存储到PostgreSQL")
            print(f"   📄 模式: 纯SQL模式（无向量化）")
            
            # 记录成功处理的文件到SQL专用进度
            sql_progress["processed_files"].append(current_file)
            save_sql_progress(SQL_SCHEMA, sql_progress["processed_files"])
            return True
        else:
            print("❌ PostgreSQL数据插入失败")
            return False
            
    except Exception as e:
        print(f"❌ 纯SQL数据加载过程中发生错误: {e}")
        return False
    finally:
        print("🔄 清理资源...")
        
        # 显式删除data变量（如果存在）
        if 'data' in locals():
            print("🗑️  删除data变量...")
            del data
        
        # 强制垃圾回收
        import gc
        print("🗑️  执行垃圾回收...")
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                print(f"  回收了 {collected} 个对象")
        
        # 由于没有GPU操作，跳过GPU内存清理
        print("🚀 纯SQL模式：跳过GPU内存清理")
        
        # 尝试强制释放内存给操作系统
        try:
            import ctypes
            import sys
            if sys.platform.startswith('linux'):
                print("♻️  尝试强制释放内存给操作系统...")
                try:
                    libc = ctypes.CDLL("libc.so.6")
                    if hasattr(libc, 'malloc_trim'):
                        result = libc.malloc_trim(0)
                        if result:
                            print("✅ malloc_trim 成功")
                        else:
                            print("⚠️  malloc_trim 返回0")
                except Exception as e:
                    print(f"⚠️  malloc_trim 失败: {e}")
        except ImportError:
            pass
        
        print("✅ 资源清理完成")


def show_sql_only_info():
    """显示纯SQL模式的功能说明"""
    print("\n" + "="*60)
    print("🚀 Milvus RAG 纯SQL数据处理器")
    print("="*60)
    print("📋 功能说明:")
    print("   ✅ 加载数据 JSON数据文件")
    print("   ✅ 插入数据到PostgreSQL数据库")
    print("   ✅ 创建URL索引和全文搜索索引")
    print("   ✅ 支持批量文件处理")
    print("   ✅ 进度跟踪和断点续传")
    print("")
    print("❌ 不包含功能:")
    print("   ❌ Milvus向量数据库操作")
    print("   ❌ BGE模型加载和向量化")
    print("   ❌ Embedding生成")
    print("   ❌ 向量搜索功能")
    print("")
    print("💡 适用场景:")
    print("   🎯 仅需要文本数据存储和检索")
    print("   🎯 测试数据加载性能")
    print("   🎯 数据预处理和清洗")
    print("   🎯 多机器并行数据处理")
    print("="*60)
