"""数据库连接和集合管理功能"""

from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

from .config_loader import (
    MILVUS_URI, DATA_COLLECTION_NAME, SEARCH_COLLECTION_NAME, USE_FP16,
    DB_MAX_URL_LENGTH, DATA_IMPORT_STRATEGY, DOC_STORAGE_MAX_CHARS, DOC_STORAGE_MAX_TOKENS, DOC_USE_TOKEN_LIMIT,
    ENABLE_DISKANN
)


def setup_milvus_client() -> None:
    """
    连接到Milvus数据库
    
    Returns:
        None
    """
    connections.connect(uri=MILVUS_URI)
    print("✅ 成功连接到Milvus")


def handle_existing_collection(collection_name: str, strategy: str = None) -> str:
    """
    处理现有集合的策略选择
    
    Args:
        collection_name: 集合名称
        strategy: 处理策略 ("ask", "append", "overwrite", None)
        
    Returns:
        最终决定的策略 ("append", "overwrite", "create_new")
    """
    if not utility.has_collection(collection_name):
        return "create_new"
    
    # 获取现有集合信息
    existing_collection = Collection(collection_name)
    entity_count = existing_collection.num_entities
    
    # 检查集合加载状态
    try:
        loading_progress = existing_collection.utility.loading_progress()
        is_loaded = loading_progress["loading_progress"] == "100%"
    except Exception:
        # 如果无法检查加载状态，假设未加载
        is_loaded = False
    
    print(f"🔍 发现现有集合: {collection_name}")
    print(f"   📊 现有记录数: {entity_count:,}")
    print(f"   📅 集合状态: {'已加载' if is_loaded else '未加载'}")
    
    if strategy is None:
        strategy = DATA_IMPORT_STRATEGY
    
    if strategy == "ask":
        print(f"\n❓ 集合 '{collection_name}' 已存在，请选择处理方式:")
        print("   1. 追加数据 (append) - 在现有数据基础上添加新数据")
        print("   2. 覆盖数据 (overwrite) - 删除现有集合并重新创建")
        print("   3. 取消操作 (cancel)")
        
        while True:
            choice = input("请输入选择 (1/2/3): ").strip()
            if choice == "1":
                return "append"
            elif choice == "2":
                return "overwrite"
            elif choice == "3":
                print("❌ 用户取消操作")
                exit(0)
            else:
                print("❌ 无效选择，请输入 1、2 或 3")
    
    elif strategy == "append":
        print(f"📝 策略: 追加数据到现有集合")
        return "append"
    
    elif strategy == "overwrite":
        print(f"🗑️  策略: 覆盖现有集合")
        return "overwrite"
    
    else:
        raise ValueError(f"无效的导入策略: {strategy}")


def create_data_collection(dense_dim: int, collection_name: str = None, confirmed_strategy: str = None):
    """
    创建或获取数据数据集合
    
    Args:
        dense_dim (int): 密集向量维度
        collection_name (str): 集合名称，默认使用DATA_COLLECTION_NAME
        confirmed_strategy (str): 已确认的导入策略 ("append", "overwrite", "create_new")
        
    Returns:
        tuple: (Collection对象, 是否为新创建的集合)
    """
    if collection_name is None:
        collection_name = DATA_COLLECTION_NAME
    
    # 如果没有提供确认的策略，使用默认处理
    if confirmed_strategy is None:
        strategy = handle_existing_collection(collection_name, None)
    else:
        strategy = confirmed_strategy
    
    if strategy == "append":
        # 追加模式：直接返回现有集合
        print(f"📝 追加模式: 使用现有集合 {collection_name}")
        existing_collection = Collection(collection_name)
        existing_collection.load()
        return existing_collection, False
    
    elif strategy == "overwrite":
        # 覆盖模式：删除现有集合
        print(f"🗑️  覆盖模式: 删除现有集合 {collection_name}")
        Collection(collection_name).drop()
    
    # 创建新集合（strategy == "create_new" 或 "overwrite"）
    print("🏗️  创建数据集合...")
    print(f"📊 集合名称: {collection_name}")
    print(f"🔢 Dense向量维度: {dense_dim}")
    print(f"🎮 计算精度: {'FP16' if USE_FP16 else 'FP32'}")
    
    # 根据不同模式选择存储精度和向量类型
    if ENABLE_DISKANN:
        print(f"🚀 DISKANN模式: 使用FP32存储（DISKANN要求）")
        dense_vector_dtype = DataType.FLOAT_VECTOR  # DISKANN只支持FP32
    else:
        print(f"💾 标准模式: 使用FP32存储和GPU_IVF_PQ索引")
        dense_vector_dtype = DataType.FLOAT_VECTOR
    
    print("📋 定义完整文档数据模式（URL+title+doc+向量）...")
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=DB_MAX_URL_LENGTH+50, description="document URL"),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1124, description="document title"),
        FieldSchema(name="doc", dtype=DataType.VARCHAR, max_length=DOC_STORAGE_MAX_CHARS+200 if not DOC_USE_TOKEN_LIMIT else DOC_STORAGE_MAX_TOKENS*4+200, description="document content"),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=dense_vector_dtype, dim=dense_dim),
    ]
    schema = CollectionSchema(fields)
    print(f"   🔗 URL最大长度: {DB_MAX_URL_LENGTH}")
    print(f"   📝 Title最大长度: 1024字符")
    if DOC_USE_TOKEN_LIMIT:
        print(f"   📄 Doc最大长度: {DOC_STORAGE_MAX_TOKENS} tokens")
    else:
        print(f"   📄 Doc最大长度: {DOC_STORAGE_MAX_CHARS}字符")
    print(f"   🎮 Dense向量类型: {dense_vector_dtype}")
    print(f"   💾 存储字段: url, title, doc, sparse_vector, dense_vector")
    
    # 创建集合
    print("🔨 创建新集合...")
    col = Collection(collection_name, schema, consistency_level="Eventually")
    
    # 创建索引
    print("📊 创建向量索引...")
    print("   🔍 创建稀疏向量索引...")
    
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", 
                    "metric_type": "IP"
                    }
    col.create_index("sparse_vector", sparse_index)
    
    # 根据配置选择合适的dense索引类型
    if ENABLE_DISKANN:
        print("   🚀 创建DISKANN索引（千万级数据优化）...")
        print("   💾 DISKANN专为大规模数据和磁盘存储优化...")
        
        # 使用DISKANN索引，参数由milvus.yaml配置文件控制
        dense_index = {
            "index_type": "DISKANN",
            "metric_type": "IP"
            # 注意：DISKANN参数由milvus.yaml文件中的DiskIndex部分控制，不在这里设置
        }
        print(f"   📊 DISKANN配置: 由milvus.yaml文件控制")
        print(f"   📋 配置文件路径: milvus.yaml -> common.DiskIndex")
        print(f"   🎯 优化目标: 千万级数据，磁盘存储，高吞吐量")
        print(f"   ⚙️  参数说明: 所有DISKANN参数在Milvus配置文件中设置")
    else:
        print("   🌟 创建HNSW索引（内存索引，百万级数据）...")
        print("   📈 HNSW提供优秀的搜索精度和低延迟...")

        # 使用HNSW索引（CPU/GPU 通用，适合百万级数据）
        dense_index = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {
                "M": 64,               # 每个节点最大连接数（高精度配置）
                "efConstruction": 200  # 构建时候选邻居数（平衡精度和构建时间）
            }
        }
        print(f"   📊 HNSW配置: M=64, efConstruction=200 (高精度配置)")

    col.create_index("dense_vector", dense_index)
    
    print("📚 加载集合到内存...")
    col.load()
    
    print(f"✅ {collection_name}集合创建完成！")
    print(f"   📊 状态: 已加载")
    print(f"   🔧 一致性级别: Eventually")
    print(f"   🎮 计算精度: {'FP16' if USE_FP16 else 'FP32'}")
    
    if ENABLE_DISKANN:
        print(f"   💾 存储精度: FP32 (DISKANN模式)")
        print(f"   🗜️  索引类型: DISKANN (千万级数据优化)")
        print(f"   🚀 DISKANN优化: 启用 (磁盘存储 + 高吞吐量)")
        print(f"   🎯 数据规模: 千万级以上数据")
        print(f"   💿 存储方式: 磁盘索引 + 内存PQ缓存")
    else:
        print(f"   💾 存储精度: FP32")
        print(f"   🗜️  索引类型: HNSW (内存索引)")
    
    if DOC_USE_TOKEN_LIMIT:
        print(f"   📋 完整文档模式: 启用（title + doc前{DOC_STORAGE_MAX_TOKENS} tokens）")
    else:
        print(f"   📋 完整文档模式: 启用（title + doc前{DOC_STORAGE_MAX_CHARS}字符）")
    
    return col, True


def get_or_create_collection(dense_dim: int = None, collection_name: str = None):
    """
    获取现有集合或创建新集合
    
    Args:
        dense_dim (int): 密集向量维度（创建新集合时需要）
        collection_name (str): 集合名称，默认使用SEARCH_COLLECTION_NAME
        
    Returns:
        Collection: Milvus集合对象
    """
    if collection_name is None:
        collection_name = SEARCH_COLLECTION_NAME
        
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.load()
        print(f"✅ 加载现有集合: {collection_name}，共 {collection.num_entities} 条记录")
        return collection
    else:
        if dense_dim is None:
            raise ValueError("创建新集合需要提供dense_dim参数")
        print(f"⚠️  集合 {collection_name} 不存在，创建新集合")
        collection, _ = create_data_collection(dense_dim, collection_name)
        return collection


def get_collection_stats(collection):
    """获取集合统计信息"""
    if not collection:
        return None
    
    try:
        return {
            "name": collection.name,
            "num_entities": collection.num_entities,
            "is_loaded": collection.utility.loading_progress()["loading_progress"] == "100%",
            "schema": {
                "fields": len(collection.schema.fields),
                "primary_field": collection.schema.primary_field.name if collection.schema.primary_field else None
            }
        }
    except Exception as e:
        print(f"⚠️  获取集合统计信息失败: {e}")
        return None 