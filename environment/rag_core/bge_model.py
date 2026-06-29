"""BGE-M3 embedding模型功能"""

import time
import torch
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from .config_loader import (
    BGE_MODEL_PATH, BGE_MAX_LENGTH, BATCH_SIZE, 
    DEVICE, USE_FP16, ENABLE_PERFORMANCE_MONITORING
)
from .utils import get_sparse_length


def setup_bge_model() -> tuple:
    """
    初始化BGE-M3模型，使用原生多GPU支持
    
    Returns:
        tuple: (embedding函数, dense向量维度)
    """
    print("🤖 初始化BGE-M3模型...")
    print(f"📂 模型路径: {BGE_MODEL_PATH}")
    print(f"📏 embedding最大输入长度: {BGE_MAX_LENGTH}")
    print(f"🎮 使用设备: {DEVICE}")
    print(f"🔢 半精度模式: {'启用' if USE_FP16 else '禁用'}")
    print("⏳ 正在加载模型，请稍候...")
    
    start_time = time.time()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print(f"🎯 GPU配置:")
    print(f"   可用GPU数量: {gpu_count}")
    print(f"   使用设备: {DEVICE}")
    print(f"   Batch大小: {BATCH_SIZE}")
    print(f"   半精度: {USE_FP16}")
    
    # 🎯 简化：直接使用BGE-M3的原生多设备支持
    ef = BGEM3EmbeddingFunction(
        model_name=BGE_MODEL_PATH,
        batch_size=BATCH_SIZE,
        device=DEVICE,  
        use_fp16=USE_FP16,
        max_length=BGE_MAX_LENGTH,
        normalize_embeddings=True
    )
    
    load_time = time.time() - start_time
    dense_dim = ef.dim["dense"]
    
    print(f"✅ BGE-M3模型加载完成！")
    print(f"   Dense维度: {dense_dim}")
    print(f"   最大长度: {BGE_MAX_LENGTH}")
    print(f"   使用设备: {DEVICE}")
    print(f"   Batch大小: {BATCH_SIZE}")
    print(f"   半精度: {USE_FP16}")
    print(f"   加载耗时: {load_time:.2f}秒")
    
    return ef, dense_dim


def warmup_model(ef) -> bool:
    """
    预热模型，确保模型处于ready状态
    
    Args:
        ef: embedding函数对象
        
    Returns:
        bool: 预热是否成功
    """
    print("🔥 预热模型...")
    
    # 准备预热文本
    warmup_texts = [
        "This is a warmup text for the BGE model.",
        "模型预热测试文本",
        "Test warmup embedding generation"
    ]
    
    try:
        # 执行预热embedding生成
        start_time = time.time()
        warmup_embeddings = ef.encode_documents(warmup_texts)
        warmup_time = time.time() - start_time
        
        # 🔧 修复：安全地验证预热结果
        dense_count = len(warmup_embeddings["dense"]) if warmup_embeddings["dense"] is not None else 0
        
        # 🔧 使用安全的sparse向量长度获取方法
        sparse_count = get_sparse_length(warmup_embeddings["sparse"])
        
        print(f"✅ 模型预热完成！")
        print(f"   预热文本数: {len(warmup_texts)}")
        print(f"   Dense向量数: {dense_count}")
        print(f"   Sparse向量数: {sparse_count}")
        print(f"   预热耗时: {warmup_time:.2f}秒")
        print(f"🚀 模型已ready，可接受用户请求！")
        
        # 验证结果的合理性
        if dense_count != len(warmup_texts):
            print(f"⚠️  警告: Dense向量数量({dense_count})与文本数量({len(warmup_texts)})不匹配")
        
        if sparse_count > 0 and sparse_count != len(warmup_texts):
            print(f"⚠️  警告: Sparse向量数量({sparse_count})与文本数量({len(warmup_texts)})不匹配")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型预热失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        import traceback
        print(f"   详细错误: {traceback.format_exc()}")
        return False


def optimize_bge_for_single_query(ef):
    """为单查询优化BGE模型配置"""
    if hasattr(ef, 'batch_size'):
        original_batch_size = ef.batch_size
        ef.batch_size = 1  # 单查询优化
        return original_batch_size
    return None


def restore_bge_batch_size(ef, original_batch_size):
    """恢复BGE模型的原始batch size"""
    if original_batch_size is not None and hasattr(ef, 'batch_size'):
        ef.batch_size = original_batch_size 