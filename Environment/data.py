#!/usr/bin/env python3
"""
🚀 LiteResearcher 数据加载工具 - 配置驱动
所有配置读取自 config.py，无命令行参数。

用法：
    1. 编辑 config.py（数据路径、模型路径、集合名、是否启用 DISKANN/SQL）
    2. python data.py
"""

import sys
from config import (
    ENABLE_BATCH_MODE, DATA_FILE_PATH, DATA_FOLDER_PATH, DATA_FILE_PATTERN,
    DATA_COLLECTION_NAME, ENABLE_SQL_STORAGE, SQL_SCHEMA, DATA_IMPORT_STRATEGY,
    ENABLE_DISKANN,
)

from rag_core import (
    show_progress, validate_config,
    create_and_load_single_data, create_and_load_data_batch,
    backup_config_for_import,
)


def main():
    print("=" * 60)
    print("🚀 LiteResearcher 数据加载工具")
    print("=" * 60)
    print(f"📊 存储架构: 完整文档（url + title + doc + 向量）")
    print(f"🗄️  Milvus集合: {DATA_COLLECTION_NAME}")

    if ENABLE_BATCH_MODE:
        print(f"📁 处理模式: 批量文件夹处理")
        print(f"📁 数据文件夹: {DATA_FOLDER_PATH}")
        print(f"🔍 文件模式: {DATA_FILE_PATTERN}")
    else:
        print(f"📁 处理模式: 单文件处理")
        print(f"📁 数据文件: {DATA_FILE_PATH}")

    print(f"📊 SQL存储: {'启用' if ENABLE_SQL_STORAGE else '禁用'}")
    if ENABLE_SQL_STORAGE:
        print(f"📋 SQL Schema: {SQL_SCHEMA}")

    print(f"⚙️  导入策略: {DATA_IMPORT_STRATEGY}")

    if ENABLE_DISKANN:
        print(f"🚀 DISKANN索引: 启用（千万级数据优化，参数见 milvus.yaml -> common.DiskIndex）")
    else:
        print(f"🚀 DISKANN索引: 禁用（使用内存 HNSW 索引）")

    print("=" * 60)

    # 显示当前进度
    show_progress(DATA_COLLECTION_NAME)

    # 配置验证
    if not validate_config(ENABLE_BATCH_MODE, DATA_FOLDER_PATH if ENABLE_BATCH_MODE else None):
        print("❌ 配置验证失败")
        sys.exit(1)

    # 配置备份
    print("\n📋 创建配置备份...")
    if not backup_config_for_import(DATA_COLLECTION_NAME):
        print("⚠️ 配置备份失败，但继续执行数据加载...")

    # 核心数据处理
    print(f"\n🎯 开始数据处理...")

    if ENABLE_BATCH_MODE:
        print(f"🔄 批量处理模式...")
        success = create_and_load_data_batch(
            DATA_FOLDER_PATH,
            DATA_FILE_PATTERN,
            DATA_IMPORT_STRATEGY,
            ENABLE_SQL_STORAGE,
        )
        action_desc = "批量文件处理"
    else:
        print(f"📄 单文件处理模式...")
        success = create_and_load_single_data(
            DATA_FILE_PATH,
            DATA_IMPORT_STRATEGY,
            ENABLE_SQL_STORAGE,
        )
        action_desc = "单文件数据加载"

    if success:
        print(f"\n🎉 {action_desc}完成！")
        print(f"✅ 集合 '{DATA_COLLECTION_NAME}' 已创建并加载数据")
        print(f"🌐 现在可以启动检索后端：cd server && bash start.sh")
        sys.exit(0)
    else:
        print(f"\n❌ {action_desc}失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
