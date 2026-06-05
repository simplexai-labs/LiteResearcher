"""系统工具函数"""

import torch
import gc
import signal
import sys
import warnings
import logging
import tiktoken
import atexit
import os
import psutil
import threading
from transformers import AutoTokenizer
from typing import Set

from .config_loader import BGE_MAX_LENGTH, BGE_MODEL_PATH

warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

# BGE tokenizer全局实例（延迟加载）
_bge_tokenizer = None

# 线程本地存储 - 每个线程独立的tokenizer
_thread_local_data = threading.local()

def get_thread_local_tiktoken_encoding():
    """获取线程本地的tiktoken编码器，避免锁竞争"""
    if not hasattr(_thread_local_data, 'tiktoken_encoding'):
        # 每个线程创建自己的编码器实例
        _thread_local_data.tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
    return _thread_local_data.tiktoken_encoding

def get_bge_tokenizer():
    """获取BGE tokenizer实例（延迟加载）"""
    global _bge_tokenizer
    if _bge_tokenizer is None:
        try:
            _bge_tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_PATH)
            print(f"✅ BGE tokenizer加载完成: {BGE_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  BGE tokenizer加载失败，使用tiktoken: {e}")
            _bge_tokenizer = "fallback"
    return _bge_tokenizer


def cleanup_gpu_memory() -> None:
    """
    清理GPU内存（保持向后兼容）
    
    Returns:
        None
    """
    if torch.cuda.is_available():
        print("🧹 清理GPU内存...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        print("✅ GPU内存清理完成")


def enhanced_memory_cleanup(enable_comprehensive: bool = True) -> dict:
    """
    增强内存清理函数 - 可选择启用全面清理
    
    Args:
        enable_comprehensive: 是否启用全面清理（包括系统内存）
    
    Returns:
        dict: 清理统计信息
    """
    if not enable_comprehensive:
        # 如果不启用全面清理，回退到原有逻辑
        cleanup_gpu_memory()
        return {"mode": "basic", "message": "使用基础GPU清理"}
    
    try:
        # 动态导入增强清理模块
        from .enhanced_memory_cleaner import enhanced_cleanup
        print("🚀 使用增强内存清理...")
        return enhanced_cleanup()
    except ImportError as e:
        print(f"⚠️  增强清理模块不可用，使用基础清理: {e}")
        cleanup_gpu_memory()
        return {"mode": "fallback", "message": "回退到基础清理"}


def smart_memory_cleanup() -> dict:
    """
    智能内存清理 - 根据当前内存使用情况选择清理策略
    
    Returns:
        dict: 清理统计信息
    """
    try:
        # 尝试获取当前内存使用情况
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        print(f"📊 当前内存使用: {memory_mb:.1f} MB ({memory_percent:.1f}%)")
        
        # 根据内存使用情况决定清理策略
        if memory_mb > 8000 or memory_percent > 80:  # 内存使用超过8GB或80%
            print("⚠️  检测到高内存使用，启用增强清理...")
            return enhanced_memory_cleanup(enable_comprehensive=True)
        else:
            print("ℹ️  内存使用正常，使用基础清理...")
            return enhanced_memory_cleanup(enable_comprehensive=False)
            
    except ImportError:
        print("⚠️  psutil不可用，使用基础清理...")
        return enhanced_memory_cleanup(enable_comprehensive=False)
    except Exception as e:
        print(f"⚠️  内存检测失败，使用基础清理: {e}")
        return enhanced_memory_cleanup(enable_comprehensive=False)


def get_sparse_length(sparse_data) -> int:
    """
    安全地获取sparse向量的长度
    
    Args:
        sparse_data: 稀疏向量数据
        
    Returns:
        int: 向量长度
    """
    try:
        if hasattr(sparse_data, 'shape'):
            return sparse_data.shape[0]
        elif hasattr(sparse_data, 'getnnz'):
            return sparse_data.getnnz()
        elif hasattr(sparse_data, '__len__'):
            return len(sparse_data)
        else:
            return 0
    except Exception:
        return 0


def safe_sparse_check(sparse_data):
    """安全地检查sparse向量是否为空"""
    try:
        if sparse_data is None:
            return True
        length = get_sparse_length(sparse_data)
        return length == 0
    except Exception:
        return True


# 全局进程管理器
_process_manager = None

class ProcessManager:
    """进程管理器，用于跟踪和清理子进程"""
    
    def __init__(self):
        self.child_processes: Set[int] = set()
        self.executor_instances = []
        self.is_shutting_down = False
        
    def register_child_process(self, pid: int):
        """注册子进程PID"""
        if not self.is_shutting_down:
            self.child_processes.add(pid)
            
    def register_executor(self, executor):
        """注册executor实例"""
        if not self.is_shutting_down:
            self.executor_instances.append(executor)
            
    def cleanup_all_processes(self):
        """清理所有子进程和executor"""
        if self.is_shutting_down:
            return
            
        self.is_shutting_down = True
        print("\n🧹 开始清理所有子进程...")
        
        # 1. 关闭所有executor
        for executor in self.executor_instances:
            try:
                print(f"🔄 关闭executor: {type(executor).__name__}")
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                print(f"⚠️  关闭executor失败: {e}")
        
        # 2. 强制终止残留的子进程
        terminated_count = 0
        for pid in list(self.child_processes):
            try:
                if psutil.pid_exists(pid):
                    proc = psutil.Process(pid)
                    if proc.is_running():
                        print(f"🔄 终止子进程 PID: {pid}")
                        proc.terminate()
                        terminated_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                # 进程已经不存在或无权限，忽略
                pass
            except Exception as e:
                print(f"⚠️  终止进程 {pid} 失败: {e}")
        
        # 3. 等待一段时间让进程优雅退出
        if terminated_count > 0:
            import time
            print(f"⏳ 等待 {terminated_count} 个进程优雅退出...")
            time.sleep(2)
            
            # 4. 强制杀死仍然存在的进程
            killed_count = 0
            for pid in list(self.child_processes):
                try:
                    if psutil.pid_exists(pid):
                        proc = psutil.Process(pid)
                        if proc.is_running():
                            print(f"💥 强制杀死进程 PID: {pid}")
                            proc.kill()
                            killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                except Exception as e:
                    print(f"⚠️  强制杀死进程 {pid} 失败: {e}")
            
            if killed_count > 0:
                print(f"💥 强制杀死了 {killed_count} 个残留进程")
        
        self.child_processes.clear()
        self.executor_instances.clear()
        print("✅ 子进程清理完成")


def get_process_manager():
    """获取全局进程管理器实例"""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager


def signal_handler(signum, frame):
    """处理中断信号"""
    print(f"\n🛑 接收到中断信号 {signum} (Ctrl+C)")
    
    # 清理所有子进程
    process_manager = get_process_manager()
    process_manager.cleanup_all_processes()
    
    # 清理GPU内存
    cleanup_gpu_memory()
    
    print("🚪 主进程即将退出...")
    sys.exit(0)


def emergency_cleanup():
    """紧急清理函数（程序正常退出时调用）"""
    try:
        process_manager = get_process_manager()
        process_manager.cleanup_all_processes()
        cleanup_gpu_memory()
    except Exception as e:
        print(f"⚠️  紧急清理过程中出错: {e}")


def setup_signal_handlers():
    """设置信号处理器和退出清理"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 注册程序退出时的清理函数
    atexit.register(emergency_cleanup)
    
    print("✅ 信号处理器和退出清理已设置")


def get_token_count(text: str, use_bge_tokenizer: bool = False) -> int:
    """
    计算文本的token数量 - 无锁版本
    
    Args:
        text: 输入文本
        use_bge_tokenizer: 是否使用BGE tokenizer（默认False，统一使用tiktoken保持一致性）
    
    Returns:
        token数量
    """
    if not text:
        return 0
    
    # 使用线程本地的tiktoken编码器，避免锁竞争
    encoding = get_thread_local_tiktoken_encoding()
    return len(encoding.encode(text))


def truncate_text_to_tokens(text: str, max_tokens: int, use_bge_tokenizer: bool = False) -> str:
    """
    将文本截断到指定的token数量 - 无锁版本
    
    Args:
        text: 输入文本
        max_tokens: 最大token数量
        use_bge_tokenizer: 是否使用BGE tokenizer（默认False，统一使用tiktoken保持一致性）
    
    Returns:
        截断后的文本
    """
    if not text:
        return ""
    
    # 使用线程本地的tiktoken编码器，避免锁竞争
    encoding = get_thread_local_tiktoken_encoding()
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # 截断到max_tokens
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)
    
    return truncated_text


def combine_title_and_text_for_embedding(title: str, text: str, max_tokens: int = BGE_MAX_LENGTH) -> str:
    """
    组合title和text的前一段作为embedding输入，确保总长度不超过max_tokens
    
    Args:
        title: 文档标题
        text: 文档内容
        max_tokens: 最大token数量，默认为BGE_MAX_LENGTH
    
    Returns:
        组合后的文本，用于embedding
    """
    if not title and not text:
        return ""
    
    # 处理空值
    title = title.strip() if title else ""
    text = text.strip() if text else ""
    
    # 如果只有title或只有text
    if not title:
        # 只有text，直接截断到max_tokens
        if get_token_count(text) <= max_tokens:
            return text
        else:
            return truncate_text_to_tokens(text, max_tokens)
    
    if not text:
        # 只有title，直接返回（通常title不会超过max_tokens）
        if get_token_count(title) <= max_tokens:
            return title
        else:
            return truncate_text_to_tokens(title, max_tokens)
    
    # 计算title的token数
    title_tokens = get_token_count(title)
    
    # 如果title本身就超过了max_tokens，只返回截断的title
    if title_tokens >= max_tokens:
        return truncate_text_to_tokens(title, max_tokens)
    
    # 为text内容预留的token数
    # 预留2个token用于分隔符（如": "或"\n"）
    remaining_tokens = max_tokens - title_tokens - 2
    
    if remaining_tokens <= 0:
        # 如果没有空间了，只返回title
        return title
    
    # 截断text到剩余的token数
    truncated_text = truncate_text_to_tokens(text, remaining_tokens)
    
    # 组合title和text
    if truncated_text:
        combined = f"{title}: {truncated_text}"
        
        # 最终验证长度（安全检查）
        if get_token_count(combined) > max_tokens:
            # 如果还是超过了，再次截断
            combined = truncate_text_to_tokens(combined, max_tokens)
        
        return combined
    else:
        return title


def safe_truncate(text, max_length):
    """安全截断文本，确保长度严格不超过max_length"""
    if not text:
        return ""
    
    text_str = str(text)
    
    # 如果长度本身就符合要求，直接返回
    if len(text_str) <= max_length:
        return text_str
    
    # 如果需要截断，确保截断后长度严格不超过max_length
    if max_length <= 3:
        # 如果max_length太小，只能返回部分字符
        return text_str[:max_length]
    else:
        # 正常情况：截断后加...，总长度正好等于max_length
        truncated = text_str[:max_length-3] + "..."
        
        # 二次检查确保长度正确
        if len(truncated) > max_length:
            # 如果还是超长，再次截断
            return text_str[:max_length-3][:max_length-3] + "..."
        
        return truncated





def get_display_max_tokens() -> int:
    """
    获取显示时的最大token数
    
    Returns:
        显示最大token数
    """
    try:
        from .config_loader import DISPLAY_MAX_TOKENS
    except ImportError:
        from config import DISPLAY_MAX_TOKENS
    
    return DISPLAY_MAX_TOKENS 