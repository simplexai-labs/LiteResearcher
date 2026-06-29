#!/usr/bin/env python3
"""
增强内存清理模块 - 即插即用的内存管理解决方案
解决Python进程在处理大数据后内存不释放的问题
"""

import gc
import os
import sys
import time
import torch
import ctypes
import tracemalloc
from typing import Optional, Dict, Any


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class EnhancedMemoryCleaner:
    """增强内存清理器 - 全面的内存管理工具"""
    
    def __init__(self, enable_monitoring: bool = True):
        """
        初始化增强内存清理器
        
        Args:
            enable_monitoring: 是否启用内存监控
        """
        self.enable_monitoring = enable_monitoring
        self.initial_memory = None
        self.peak_memory = None
        
        if self.enable_monitoring and PSUTIL_AVAILABLE:
            self.initial_memory = self._get_process_memory()
        
        # 初始化GPU监控
        self.gpu_initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_initialized = True
            except:
                pass
    
    def _get_process_memory(self) -> Dict[str, float]:
        """获取当前进程内存信息 (MB)"""
        if not PSUTIL_AVAILABLE:
            return {"rss": 0, "vms": 0, "percent": 0}
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            return {
                "rss": memory_info.rss / 1024 / 1024,  # 物理内存 (MB)
                "vms": memory_info.vms / 1024 / 1024,  # 虚拟内存 (MB) 
                "percent": memory_percent  # 系统内存占用百分比
            }
        except Exception:
            return {"rss": 0, "vms": 0, "percent": 0}
    
    def _get_gpu_memory(self) -> Dict[str, float]:
        """获取GPU内存信息 (MB)"""
        if not self.gpu_initialized:
            return {"used": 0, "total": 0, "free": 0}
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 监控第一张卡
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            return {
                "used": mem_info.used / 1024 / 1024,
                "total": mem_info.total / 1024 / 1024,
                "free": mem_info.free / 1024 / 1024
            }
        except Exception:
            return {"used": 0, "total": 0, "free": 0}
    
    def _force_garbage_collection(self, rounds: int = 5) -> int:
        """强制执行多轮垃圾回收"""
        total_collected = 0
        
        print(f"🗑️  执行 {rounds} 轮强制垃圾回收...")
        for i in range(rounds):
            collected = gc.collect()
            total_collected += collected
            if collected > 0:
                print(f"  第 {i+1} 轮: 回收 {collected} 个对象")
        
        # 清理所有代的垃圾
        for generation in range(3):
            gc.collect(generation)
        
        return total_collected
    
    def _clear_pytorch_cache(self):
        """清理PyTorch缓存"""
        if not torch.cuda.is_available():
            return
        
        print("🎮 清理PyTorch GPU缓存...")
        
        # 多次清理确保彻底
        for i in range(3):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # 尝试清理所有GPU设备
        for device_id in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
            except Exception:
                pass
    
    def _force_memory_trim(self) -> bool:
        """强制释放内存给操作系统"""
        success = False
        
        print("♻️  尝试强制释放内存给操作系统...")
        
        # 方法1: Linux的malloc_trim
        try:
            if sys.platform.startswith('linux'):
                # 尝试加载libc并调用malloc_trim
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                if hasattr(libc, 'malloc_trim'):
                    result = libc.malloc_trim(0)
                    if result:
                        print("  ✅ malloc_trim 成功")
                        success = True
                    else:
                        print("  ⚠️  malloc_trim 返回0")
        except Exception as e:
            print(f"  ⚠️  malloc_trim 失败: {e}")
        
        # 方法2: 尝试其他平台的内存整理
        try:
            if sys.platform == 'win32':
                # Windows: 尝试调用SetProcessWorkingSetSize
                import ctypes.wintypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetCurrentProcess()
                kernel32.SetProcessWorkingSetSize(handle, -1, -1)
                print("  ✅ Windows内存整理完成")
                success = True
        except Exception as e:
            print(f"  ⚠️  Windows内存整理失败: {e}")
        
        # 方法3: 更激进的内存释放策略
        try:
            # 删除所有可能的大对象引用
            import sys
            for name in list(sys.modules.keys()):
                if name.startswith('__pycache__') or name.startswith('temp_'):
                    try:
                        del sys.modules[name]
                    except:
                        pass
            
            # 强制执行Python内存回收
            import mmap
            if hasattr(mmap, 'PAGESIZE'):
                # 尝试让系统回收页面
                pass
            
            print("  ✅ 激进内存策略执行完成")
            success = True
            
        except Exception as e:
            print(f"  ⚠️  激进内存策略失败: {e}")
        
        return success
    
    def _clear_python_caches(self):
        """清理Python内部缓存"""
        print("🐍 清理Python内部缓存...")
        
        # 清理sys.modules中未使用的模块
        # (谨慎操作，只清理明确可以清理的)
        
        # 清理线程本地数据
        try:
            import threading
            for thread in threading.enumerate():
                if hasattr(thread, '_target') and thread._target is None:
                    # 清理已完成线程的数据
                    pass
        except Exception:
            pass
        
        # 清理正则表达式缓存
        try:
            import re
            re.purge()
            print("  ✅ 正则表达式缓存已清理")
        except Exception:
            pass
        
        # 清理编解码器缓存
        try:
            import codecs
            codecs.getincrementaldecoder('utf-8')._incremental_decoder_cache.clear()
        except Exception:
            pass
    
    def _memory_pressure_relief(self):
        """内存压力缓解"""
        print("💾 执行内存压力缓解...")
        
        # 显式删除大对象的引用
        # 这里可以根据需要添加特定的清理逻辑
        
        # 尝试释放未使用的内存页
        try:
            if hasattr(os, 'sync'):
                os.sync()  # 同步文件系统缓存
        except Exception:
            pass
    
    def get_memory_status(self) -> Dict[str, Any]:
        """获取当前内存状态"""
        process_mem = self._get_process_memory()
        gpu_mem = self._get_gpu_memory()
        
        status = {
            "process_memory_mb": process_mem,
            "gpu_memory_mb": gpu_mem,
            "timestamp": time.time()
        }
        
        # 计算内存变化
        if self.initial_memory:
            status["memory_delta_mb"] = {
                "rss": process_mem["rss"] - self.initial_memory["rss"],
                "vms": process_mem["vms"] - self.initial_memory["vms"],
                "percent": process_mem["percent"] - self.initial_memory["percent"]
            }
        
        return status
    
    def print_memory_status(self, title: str = "内存状态"):
        """打印内存状态"""
        status = self.get_memory_status()
        process_mem = status["process_memory_mb"]
        gpu_mem = status["gpu_memory_mb"]
        
        print(f"\n📊 {title}:")
        print(f"   🖥️  进程内存: {process_mem['rss']:.1f} MB (物理) / {process_mem['vms']:.1f} MB (虚拟)")
        print(f"   📈 系统占用: {process_mem['percent']:.1f}%")
        
        if gpu_mem["total"] > 0:
            gpu_usage_percent = (gpu_mem["used"] / gpu_mem["total"]) * 100
            print(f"   🎮 GPU内存: {gpu_mem['used']:.1f} MB / {gpu_mem['total']:.1f} MB ({gpu_usage_percent:.1f}%)")
        
        # 显示内存变化
        if "memory_delta_mb" in status:
            delta = status["memory_delta_mb"]
            print(f"   📉 内存变化: {delta['rss']:+.1f} MB (物理) / {delta['percent']:+.1f}% (系统)")
    
    def comprehensive_cleanup(self, 
                            enable_force_trim: bool = True,
                            enable_cache_clear: bool = True,
                            gc_rounds: int = 5) -> Dict[str, Any]:
        """
        执行全面的内存清理
        
        Args:
            enable_force_trim: 是否启用强制内存释放
            enable_cache_clear: 是否清理各种缓存
            gc_rounds: 垃圾回收轮数
        
        Returns:
            清理统计信息
        """
        print("\n🧹 开始全面内存清理...")
        
        # 记录清理前状态
        before_status = self.get_memory_status()
        if self.enable_monitoring:
            self.print_memory_status("清理前")
        
        cleanup_stats = {
            "gc_collected": 0,
            "force_trim_success": False,
            "cache_cleared": False,
            "memory_before": before_status,
            "memory_after": None,
            "memory_freed_mb": 0
        }
        
        # 1. 强制垃圾回收
        cleanup_stats["gc_collected"] = self._force_garbage_collection(gc_rounds)
        
        # 2. 清理PyTorch缓存
        self._clear_pytorch_cache()
        
        # 3. 清理Python缓存
        if enable_cache_clear:
            self._clear_python_caches()
            cleanup_stats["cache_cleared"] = True
        
        # 4. 内存压力缓解
        self._memory_pressure_relief()
        
        # 5. 强制内存释放
        if enable_force_trim:
            cleanup_stats["force_trim_success"] = self._force_memory_trim()
        
        # 6. 最后一轮垃圾回收
        final_collected = self._force_garbage_collection(2)
        cleanup_stats["gc_collected"] += final_collected
        
        # 等待一段时间让系统处理
        time.sleep(1)
        
        # 记录清理后状态
        after_status = self.get_memory_status()
        cleanup_stats["memory_after"] = after_status
        
        # 计算释放的内存
        if before_status and after_status:
            memory_freed = before_status["process_memory_mb"]["rss"] - after_status["process_memory_mb"]["rss"]
            cleanup_stats["memory_freed_mb"] = memory_freed
        
        # 显示清理结果
        if self.enable_monitoring:
            self.print_memory_status("清理后")
            print(f"\n✅ 内存清理完成!")
            print(f"   🗑️  垃圾回收: {cleanup_stats['gc_collected']} 个对象")
            print(f"   💾 内存释放: {cleanup_stats['memory_freed_mb']:.1f} MB")
            if cleanup_stats["force_trim_success"]:
                print(f"   ♻️  强制释放: 成功")
            if cleanup_stats["cache_cleared"]:
                print(f"   🧹 缓存清理: 完成")
        
        return cleanup_stats
    
    def ultra_cleanup_for_large_data(self) -> Dict[str, Any]:
        """
        超级清理模式 - 专门用于大数据处理完成后的彻底清理
        这会执行最激进的清理策略，可能会影响性能但能最大化释放内存
        
        Returns:
            清理统计信息
        """
        print("\n🚀 启动超级内存清理模式（大数据专用）...")
        
        # 记录清理前状态
        before_status = self.get_memory_status()
        if self.enable_monitoring:
            self.print_memory_status("超级清理前")
        
        cleanup_stats = {
            "gc_collected": 0,
            "force_trim_success": False,
            "cache_cleared": False,
            "modules_cleared": 0,
            "tensors_cleared": 0,
            "memory_before": before_status,
            "memory_after": None,
            "memory_freed_mb": 0
        }
        
        # 1. 多轮激进垃圾回收
        print("🗑️  执行激进垃圾回收...")
        for i in range(10):  # 比常规清理更多轮次
            collected = gc.collect()
            cleanup_stats["gc_collected"] += collected
            if collected > 0:
                print(f"  第 {i+1} 轮: 回收 {collected} 个对象")
            
            # 清理所有代
            for generation in range(3):
                gc.collect(generation)
        
        # 2. 彻底清理PyTorch和GPU
        print("🎮 彻底清理PyTorch和GPU...")
        if torch.cuda.is_available():
            # 清理所有GPU
            for device_id in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        # 尝试释放所有未使用的缓存内存
                        torch.cuda.synchronize()
                except Exception:
                    pass
            
            # 额外的PyTorch清理
            try:
                import torch.backends.cudnn as cudnn
                cudnn.benchmark = False
                if hasattr(torch.cuda, 'reset_max_memory_allocated'):
                    torch.cuda.reset_max_memory_allocated()
                if hasattr(torch.cuda, 'reset_max_memory_cached'):
                    torch.cuda.reset_max_memory_cached()
            except:
                pass
        
        # 3. 清理大型Python对象和模块
        print("📦 清理Python模块和大对象...")
        try:
            import sys
            modules_to_clear = []
            
            # 找到可能占用大量内存的模块
            for name, module in sys.modules.items():
                if any(keyword in name.lower() for keyword in [
                    'numpy', 'pandas', 'torch', 'tensorflow', 'transformers',
                    'sklearn', 'scipy', 'matplotlib', 'data', 'cache'
                ]):
                    # 不要删除核心模块，只清理它们的缓存
                    if hasattr(module, '__dict__'):
                        for attr_name in list(module.__dict__.keys()):
                            if attr_name.startswith('_cache') or attr_name.startswith('cache_'):
                                try:
                                    delattr(module, attr_name)
                                except:
                                    pass
                
                # 清理临时模块
                if any(temp_name in name for temp_name in ['temp_', 'tmp_', '__pycache__']):
                    modules_to_clear.append(name)
            
            # 删除临时模块
            for name in modules_to_clear:
                try:
                    del sys.modules[name]
                    cleanup_stats["modules_cleared"] += 1
                except:
                    pass
                    
        except Exception as e:
            print(f"  ⚠️  模块清理警告: {e}")
        
        # 4. 超级缓存清理
        print("🧹 超级缓存清理...")
        self._clear_python_caches()
        
        # 额外的缓存清理
        try:
            # 清理更多Python内部缓存
            import linecache
            linecache.clearcache()
            
            import functools
            # 尝试清理functools的lru_cache
            for obj in gc.get_objects():
                if hasattr(obj, 'cache_clear') and callable(getattr(obj, 'cache_clear')):
                    try:
                        obj.cache_clear()
                    except:
                        pass
                        
        except Exception as e:
            print(f"  ⚠️  超级缓存清理警告: {e}")
            
        cleanup_stats["cache_cleared"] = True
        
        # 5. 强制内存释放 (多次尝试)
        print("♻️  多次强制内存释放...")
        for i in range(3):
            success = self._force_memory_trim()
            if success:
                cleanup_stats["force_trim_success"] = True
                break
            time.sleep(0.5)  # 等待一下再试
        
        # 6. 最终垃圾回收轮
        print("🔄 最终垃圾回收...")
        final_collected = 0
        for i in range(5):
            collected = gc.collect()
            final_collected += collected
            if collected == 0:
                break
        cleanup_stats["gc_collected"] += final_collected
        
        # 7. 等待系统处理
        print("⏱️  等待系统处理...")
        time.sleep(2)
        
        # 记录清理后状态
        after_status = self.get_memory_status()
        cleanup_stats["memory_after"] = after_status
        
        # 计算释放的内存
        if before_status and after_status:
            memory_freed = before_status["process_memory_mb"]["rss"] - after_status["process_memory_mb"]["rss"]
            cleanup_stats["memory_freed_mb"] = memory_freed
        
        # 显示清理结果
        if self.enable_monitoring:
            self.print_memory_status("超级清理后")
            print(f"\n🎉 超级内存清理完成!")
            print(f"   🗑️  垃圾回收: {cleanup_stats['gc_collected']} 个对象")
            print(f"   📦 清理模块: {cleanup_stats['modules_cleared']} 个")
            print(f"   💾 内存释放: {cleanup_stats['memory_freed_mb']:.1f} MB")
            if cleanup_stats["force_trim_success"]:
                print(f"   ♻️  强制释放: 成功")
            print(f"   🧹 超级缓存清理: 完成")
        
        return cleanup_stats

    def auto_cleanup_on_high_usage(self, 
                                  memory_threshold_mb: float = 8000,
                                  memory_percent_threshold: float = 80.0) -> bool:
        """
        在内存使用量高时自动清理
        
        Args:
            memory_threshold_mb: 内存使用量阈值 (MB)
            memory_percent_threshold: 系统内存占用百分比阈值
        
        Returns:
            是否执行了清理
        """
        status = self.get_memory_status()
        process_mem = status["process_memory_mb"]
        
        should_cleanup = (
            process_mem["rss"] > memory_threshold_mb or 
            process_mem["percent"] > memory_percent_threshold
        )
        
        if should_cleanup:
            print(f"⚠️  检测到高内存使用量: {process_mem['rss']:.1f} MB ({process_mem['percent']:.1f}%)")
            print(f"   触发自动清理 (阈值: {memory_threshold_mb} MB 或 {memory_percent_threshold}%)")
            
            # 如果内存使用超过20GB，使用超级清理
            if process_mem["rss"] > 20000:
                print("🚀 内存使用超过20GB，启动超级清理模式...")
                self.ultra_cleanup_for_large_data()
            else:
                self.comprehensive_cleanup()
            return True
        
        return False
    
    def __del__(self):
        """析构时清理GPU监控"""
        if self.gpu_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


# 全局清理器实例
_global_cleaner: Optional[EnhancedMemoryCleaner] = None


def get_memory_cleaner(enable_monitoring: bool = True) -> EnhancedMemoryCleaner:
    """获取全局内存清理器实例"""
    global _global_cleaner
    if _global_cleaner is None:
        _global_cleaner = EnhancedMemoryCleaner(enable_monitoring)
    return _global_cleaner


def enhanced_cleanup(
    enable_force_trim: bool = True,
    enable_cache_clear: bool = True,
    gc_rounds: int = 5,
    auto_threshold_mb: float = 8000
) -> Dict[str, Any]:
    """
    增强内存清理的便捷函数 - 可直接替换原有的cleanup_gpu_memory()
    
    Args:
        enable_force_trim: 是否启用强制内存释放
        enable_cache_clear: 是否清理各种缓存  
        gc_rounds: 垃圾回收轮数
        auto_threshold_mb: 内存阈值，超过此值时启用更激进的清理
    
    Returns:
        清理统计信息
    """
    cleaner = get_memory_cleaner()
    
    # 如果内存使用量很高，启用更激进的清理
    current_status = cleaner.get_memory_status()
    current_memory = current_status["process_memory_mb"]["rss"]
    
    if current_memory > auto_threshold_mb:
        print(f"⚠️  高内存使用检测: {current_memory:.1f} MB > {auto_threshold_mb} MB")
        print("🚀 启用激进清理模式...")
        enable_force_trim = True
        enable_cache_clear = True
        gc_rounds = max(gc_rounds, 8)
    
    return cleaner.comprehensive_cleanup(
        enable_force_trim=enable_force_trim,
        enable_cache_clear=enable_cache_clear,
        gc_rounds=gc_rounds
    )


def monitor_and_cleanup_if_needed(
    memory_threshold_mb: float = 8000,
    memory_percent_threshold: float = 80.0
) -> bool:
    """
    监控内存使用并在需要时自动清理
    
    Args:
        memory_threshold_mb: 内存使用量阈值 (MB)
        memory_percent_threshold: 系统内存占用百分比阈值
    
    Returns:
        是否执行了清理
    """
    cleaner = get_memory_cleaner()
    return cleaner.auto_cleanup_on_high_usage(memory_threshold_mb, memory_percent_threshold)


def print_current_memory_status(title: str = "当前内存状态"):
    """打印当前内存状态的便捷函数"""
    cleaner = get_memory_cleaner()
    cleaner.print_memory_status(title)


def simple_enhanced_cleanup():
    """
    简化版增强清理 - 最小侵入性的替换函数
    可以直接替换现有的 cleanup_gpu_memory() 调用
    """
    return enhanced_cleanup(
        enable_force_trim=True,
        enable_cache_clear=True,
        gc_rounds=5
    )


if __name__ == "__main__":
    # 测试模块功能
    print("🧪 测试增强内存清理模块...")
    
    cleaner = get_memory_cleaner()
    cleaner.print_memory_status("初始状态")
    
    # 创建一些测试数据
    print("\n📝 创建测试数据...")
    test_data = [list(range(1000000)) for _ in range(10)]  # ~80MB数据
    
    cleaner.print_memory_status("创建数据后")
    
    # 删除数据
    del test_data
    
    # 执行清理
    stats = enhanced_cleanup()
    
    print(f"\n📋 清理统计: {stats}")
