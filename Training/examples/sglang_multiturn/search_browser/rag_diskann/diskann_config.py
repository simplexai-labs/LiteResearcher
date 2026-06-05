#!/usr/bin/env python3
"""
🎯 DISKANN RAG服务专用配置文件
用于DISKANN RAG服务和Embedding服务的本地配置
完全独立于父级api_config.py
"""

import os

# ============================= 服务端口配置 =============================

DISKANN_RAG_PORT = 8018          # DISKANN RAG服务端口
DISKANN_EMBEDDING_PORT = 8028    # DISKANN Embedding服务端口

# ============================= BGE模型配置（DISKANN专用） =============================

# DISKANN专用BGE模型配置
API_BGE_MODEL_PATH = "/mnt/nas_nfs/home/wanli/Embeding_Models/bge-m3"  # BGE模型路径
API_BGE_MAX_LENGTH = 128         # Embedding最大输入长度
API_BATCH_SIZE = 32              # 批次大小

# GPU设备配置 - DISKANN使用六张GPU并行
API_DEVICE = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5"]  # DISKANN使用的GPU设备（六卡并行）

# DISKANN必须使用FP32精度
API_USE_FP16 = False             # DISKANN不支持FP16，必须使用FP32

# ============================= Milvus连接配置 =============================

# Milvus连接地址（根据实际部署修改）
# 本地: http://localhost:19530
# 远程: http://10.160.199.229:19530
API_MILVUS_URI = "http://10.160.199.232:19530"

# 搜索集合配置
SEARCH_COLLECTION_NAME = "serper_test"  # 默认搜索集合名称

# ============================= DISKANN搜索配置 =============================

# DISKANN搜索参数
DISKANN_SEARCH_LIST = 100        # DISKANN搜索列表大小（默认100，可根据精度需求调整50-300）
DISKANN_DEFAULT_LIMIT = 10       # 默认返回结果数

# 搜索性能配置
DEFAULT_SPARSE_WEIGHT = 0.7      # 默认稀疏向量权重
DEFAULT_DENSE_WEIGHT = 1.0       # 默认密集向量权重

# ============================= 服务器配置 =============================

# 网络配置
API_HOST = "0.0.0.0"             # 服务器监听地址

# 并发配置
UVICORN_WORKERS = 8               # RAG服务器worker数量
EMBEDDING_WORKERS = 6             # Embedding服务器worker数量（每卡 1 worker，批处理更高效）

# ============================= Embedding服务配置 =============================

# 批处理配置 - 高并发优化
MAX_BATCH_SIZE = 128              # 最大batch size（4090可以处理64个文本）
MAX_WAIT_TIME = 0.3             # 最大等待时间（20ms，给请求积累时间）
MIN_BATCH_SIZE = 8              # 最小batch size（凑够32个再处理，充分利用GPU）

# ============================= 并发限制配置 =============================

# 并发控制（信号量）- 防止系统过载
TOTAL_CONCURRENCY_LIMIT = 400       # 总请求并发限制
EMBEDDING_CONCURRENCY_LIMIT = 400   # Embedding 服务并发限制
MILVUS_CONCURRENCY_LIMIT = 400      # Milvus 查询并发限制

# ============================= 性能配置 =============================

# 日志配置
LOG_LEVEL = "info"               # 日志级别

# 模型预热
ENABLE_MODEL_WARMUP = True       # 是否在启动时预热模型

# 内存管理
ENABLE_MEMORY_OPTIMIZATION = True   # 是否启用内存优化

# ============================= 配置验证 =============================

def validate_diskann_config():
    """验证DISKANN配置的有效性"""
    errors = []

    # 检查必要的路径
    if not os.path.exists(API_BGE_MODEL_PATH):
        errors.append(f"BGE模型路径不存在: {API_BGE_MODEL_PATH}")

    # 检查设备配置
    if not API_DEVICE or not isinstance(API_DEVICE, list):
        errors.append("API_DEVICE必须是设备列表")

    # 检查端口配置
    if not (1024 <= DISKANN_RAG_PORT <= 65535):
        errors.append(f"DISKANN RAG端口无效: {DISKANN_RAG_PORT}")

    if not (1024 <= DISKANN_EMBEDDING_PORT <= 65535):
        errors.append(f"DISKANN Embedding端口无效: {DISKANN_EMBEDDING_PORT}")

    # 检查批次大小
    if API_BATCH_SIZE < 1 or API_BATCH_SIZE > 256:
        errors.append(f"批次大小无效: {API_BATCH_SIZE}")

    # DISKANN不支持FP16
    if API_USE_FP16:
        errors.append("DISKANN不支持FP16，API_USE_FP16必须为False")

    if errors:
        print("❌ DISKANN配置验证失败:")
        for error in errors:
            print(f"   • {error}")
        return False

    print("✅ DISKANN配置验证通过")
    return True


def print_diskann_config():
    """打印DISKANN配置信息"""
    print("\n" + "="*60)
    print("🎯 DISKANN RAG服务配置")
    print("="*60)
    print(f"📡 DISKANN RAG服务: http://{API_HOST}:{DISKANN_RAG_PORT}")
    print(f"🤖 DISKANN Embedding服务: http://{API_HOST}:{DISKANN_EMBEDDING_PORT}")
    print(f"🤖 BGE模型: {API_BGE_MODEL_PATH}")
    print(f"📏 最大长度: {API_BGE_MAX_LENGTH}")
    print(f"🎮 设备: {API_DEVICE}")
    print(f"🔢 精度: FP32 (DISKANN要求)")
    print(f"📦 批次大小: {API_BATCH_SIZE}")
    print(f"📊 搜索集合: {SEARCH_COLLECTION_NAME}")
    print(f"🔍 DISKANN搜索列表大小: {DISKANN_SEARCH_LIST}")
    print(f"⚡ 特性: 内嵌模型 + DISKANN索引 + FP32向量")
    print("="*60)


if __name__ == "__main__":
    print("🧪 测试DISKANN配置...")
    print_diskann_config()
    validate_diskann_config()
