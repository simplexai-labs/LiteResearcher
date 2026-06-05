"""数据加载功能"""

import json
import chardet
import gzip
import os
import time
import gc
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import math

# 用于内存和显存监控
try:
    import psutil  # 用于获取进程内存 (RAM)
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil 未安装，无法监控内存使用情况")

try:
    import pynvml  # 用于获取NVIDIA显存 (VRAM)
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠️ pynvml 未安装，无法监控显存使用情况")


def get_memory_info():
    """获取当前Python进程的内存和GPU 0的显存信息。"""
    # 获取进程内存 (RAM)
    ram_str = "N/A"
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            ram_used_mb = process.memory_info().rss / (1024 * 1024)
            ram_str = f"{ram_used_mb:.2f} MB"
        except Exception:
            ram_str = "无法获取"

    # 获取显存 (VRAM)
    gpu_str = "N/A"
    if PYNVML_AVAILABLE:
        try:
            # 假设监控第一张卡 (GPU 0)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_used_mb = mem_info.used / (1024 * 1024)
            gpu_total_mb = mem_info.total / (1024 * 1024)
            gpu_str = f"{gpu_used_mb:.0f}/{gpu_total_mb:.0f} MB"
        except Exception:
            # 如果没有NVIDIA GPU或驱动，则会失败
            gpu_str = "No NVIDIA GPU"
        
    return ram_str, gpu_str


def is_gzip_file(file_path: str) -> bool:
    """
    检测文件是否为gzip压缩文件
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        bool: 是否为gzip文件
    """
    # 检查文件扩展名
    if file_path.endswith('.gz'):
        return True
    
    # 检查文件头部magic number
    try:
        with open(file_path, 'rb') as f:
            header = f.read(2)
            return header == b'\x1f\x8b'  # gzip magic number
    except Exception:
        return False


def detect_encoding_gzip(file_path: str, sample_size: int = 10240) -> str:
    """
    检测gzip压缩文件的编码
    
    Args:
        file_path (str): 文件路径
        sample_size (int): 采样大小
        
    Returns:
        str: 检测到的编码格式
    """
    try:
        with gzip.open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            print(f"🔍 检测到gzip文件编码: {encoding} (置信度: {confidence:.2f})")
            return encoding
    except Exception as e:
        print(f"⚠️  gzip文件编码检测失败: {e}")
        return 'utf-8'


def detect_encoding(file_path: str, sample_size: int = 10240) -> str:
    """
    检测文件编码（支持普通文件和gzip文件）
    
    Args:
        file_path (str): 文件路径
        sample_size (int): 采样大小
        
    Returns:
        str: 检测到的编码格式
    """
    if is_gzip_file(file_path):
        return detect_encoding_gzip(file_path, sample_size)
    
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            print(f"🔍 检测到文件编码: {encoding} (置信度: {confidence:.2f})")
            return encoding
    except Exception as e:
        print(f"⚠️  编码检测失败: {e}")
        return 'utf-8'


def clean_text(text: str) -> str:
    """
    清理数据文本中的有问题字符
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除NULL字符（0x00）- 这是PostgreSQL不支持的主要问题
    cleaned_text = text.replace('\x00', '')
    
    # 移除其他控制字符，但保留常见的空白字符
    import re
    # 保留换行符(\n)、制表符(\t)、回车符(\r)，移除其他控制字符
    cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned_text)
    
    return cleaned_text


def open_with_encoding(file_path: str) -> tuple:
    """
    尝试多种编码方式打开文件（支持普通文件和gzip文件）
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        tuple: (文件对象, 使用的编码, 是否为gzip文件)
    """
    is_gzip = is_gzip_file(file_path)
    detected_encoding = detect_encoding(file_path)
    encodings = [detected_encoding, 'utf-8', 'latin-1', 'cp1252']
    
    print(f"📁 文件类型: {'gzip压缩文件' if is_gzip else '普通文件'}")
    
    for encoding in encodings:
        try:
            print(f"🔄 尝试使用编码: {encoding}")
            
            if is_gzip:
                # 使用gzip.open打开压缩文件
                f = gzip.open(file_path, 'rt', encoding=encoding, errors='replace')
            else:
                # 使用普通open打开文件
                f = open(file_path, 'r', encoding=encoding, errors='replace')
            
            # 测试读取几行
            pos = f.tell() if not is_gzip else None  # gzip文件不支持tell/seek
            test_lines = 0
            for _ in range(3):
                line = f.readline()
                if not line:
                    break
                test_lines += 1
            
            if not is_gzip and pos is not None:
                f.seek(pos)
            elif is_gzip:
                # gzip文件需要重新打开
                f.close()
                f = gzip.open(file_path, 'rt', encoding=encoding, errors='replace')
            
            print(f"✅ 成功使用编码: {encoding} (测试读取了 {test_lines} 行)")
            return f, encoding, is_gzip
            
        except Exception as e:
            print(f"❌ 编码 {encoding} 失败: {e}")
            if 'f' in locals():
                try:
                    f.close()
                except:
                    pass
            continue
    
    print("⚠️  使用UTF-8编码并忽略错误字符")
    if is_gzip:
        f = gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
    else:
        f = open(file_path, 'r', encoding='utf-8', errors='ignore')
    return f, 'utf-8', is_gzip


def count_file_lines(file_path: str) -> int:
    """
    计算文件总行数（支持普通文件和gzip文件）
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        int: 文件行数
    """
    try:
        f, encoding_used, is_gzip = open_with_encoding(file_path)
        file_type = "gzip压缩文件" if is_gzip else "普通文件"
        print(f"📊 计算{file_type}总行数...")
        
        total_lines = 0
        for _ in tqdm(f, desc=f"计算行数({file_type})", unit="行"):
            total_lines += 1
        f.close()
        print(f"📊 {file_type}总行数: {total_lines:,}")
        return total_lines
    except Exception as e:
        print(f"⚠️  无法计算总行数: {e}")
        return None


def process_json_lines_batch(lines_batch: List[Tuple[int, str]]) -> Tuple[List[dict], int]:
    """
    处理一批JSON行的多线程函数
    
    Args:
        lines_batch: (行号, 行内容) 的列表
        
    Returns:
        (有效文档列表, 错误数量)
    """
    data = []
    error_count = 0
    
    for line_num, line in lines_batch:
        try:
            line = line.strip()
            if not line:
                continue
                
            json_obj = json.loads(line)
            doc_id = json_obj.get("id", f"doc_{line_num}")
            metadata = json_obj.get("metadata", {})
            title = metadata.get("title", "")
            url = metadata.get("url", doc_id)
            text = json_obj.get("text", "")
            
            if text.strip():
                # 清理所有字段中的有问题字符
                clean_doc_id = clean_text(str(doc_id))
                clean_title = clean_text(str(title))
                clean_url = clean_text(str(url))
                cleaned_text = clean_text(str(text))

                data.append({
                    "id": clean_doc_id,
                    "title": clean_title,
                    "url": clean_url,
                    "text": cleaned_text
                })
        except json.JSONDecodeError:
            error_count += 1
        except Exception:
            error_count += 1
    
    return data, error_count


def load_data_corpus_multithread(file_path: str, num_workers: int = 4) -> list:
    """
    多线程加载数据数据集（支持.json和.json.gz文件）
    
    Args:
        file_path (str): 数据文件路径
        num_workers (int): 工作线程数
        
    Returns:
        list: 文档数据列表
    """
    print(f"📚 多线程加载数据 JSON文件: {file_path} (工作线程: {num_workers})")
    
    total_lines = count_file_lines(file_path)
    batch_size = max(1000, total_lines // (num_workers * 4))  # 每个批次至少1000行
    
    data = []
    total_error_count = 0
    
    try:
        f, encoding_used, is_gzip = open_with_encoding(file_path)
        file_type = "gzip压缩文件" if is_gzip else "普通文件"
        print(f"🔄 开始多线程解析数据 JSON数据 ({file_type}, 编码: {encoding_used}, 批次大小: {batch_size:,})...")
        
        # 读取所有行并分批
        lines_with_numbers = [(i+1, line) for i, line in enumerate(f)]
        f.close()
        
        # 分批处理
        num_batches = math.ceil(len(lines_with_numbers) / batch_size)
        
        # 创建进度条
        progress_bar = tqdm(
            total=num_batches,
            desc="🚀 多线程处理批次",
            unit="batch",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            dynamic_ncols=True
        )
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 注册executor到进程管理器
            try:
                from .utils import get_process_manager
                process_manager = get_process_manager()
                process_manager.register_executor(executor)
                print(f"📝 已注册ThreadPoolExecutor到进程管理器")
            except ImportError:
                # 如果无法导入，继续执行
                process_manager = None
            
            # 提交所有批次任务
            futures = []
            for i in range(0, len(lines_with_numbers), batch_size):
                batch = lines_with_numbers[i:i + batch_size]
                future = executor.submit(process_json_lines_batch, batch)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    # 检查是否正在关闭
                    if process_manager and process_manager.is_shutting_down:
                        print("🛑 检测到关闭信号，停止处理剩余批次")
                        break
                        
                    batch_data, batch_errors = future.result(timeout=30)
                    data.extend(batch_data)
                    total_error_count += batch_errors
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        '有效记录': len(data),
                        '错误': total_error_count,
                        '线程': num_workers
                    })
                    progress_bar.update(1)
                    
                except Exception as e:
                    if not (process_manager and process_manager.is_shutting_down):
                        tqdm.write(f"⚠️  批次处理失败: {e}")
                    total_error_count += batch_size  # 估算错误数
        
        progress_bar.close()
        
    except KeyboardInterrupt:
        print(f"\n🛑 用户中断了多线程数据加载")
        progress_bar.close()
        # 进程管理器会自动清理线程
        raise
    except Exception as e:
        print(f"❌ 多线程文件读取失败: {e}")
        progress_bar.close()
        raise e
    
    print(f"✅ 多线程加载完成: {len(data):,} 条记录")
    if total_error_count > 0:
        print(f"⚠️  跳过了 {total_error_count:,} 行有问题的数据")
        print(f"📊 数据完整性: {len(data)/(len(data)+total_error_count)*100:.2f}%")
    
    return data


def load_data_corpus(file_path: str, use_multithread: bool = True, num_workers: int = 8) -> list:
    """
    加载数据数据集，支持多线程加速（支持.json和.json.gz文件）
    
    Args:
        file_path (str): 数据文件路径
        use_multithread (bool): 是否使用多线程
        num_workers (int): 工作线程数
        
    Returns:
        list: 文档数据列表
    """
    if use_multithread:
        return load_data_corpus_multithread(file_path, num_workers)
    
    print(f"📚 单线程加载数据 JSON文件: {file_path}")
    
    data = []
    error_count = 0
    total_lines = count_file_lines(file_path)
    
    try:
        f, encoding_used, is_gzip = open_with_encoding(file_path)
        file_type = "gzip压缩文件" if is_gzip else "普通文件"
        print(f"🔄 开始解析数据 JSON数据 ({file_type}, 编码: {encoding_used})...")
        
        # 创建详细的数据加载进度条
        load_progress = tqdm(
            f, 
            total=total_lines,
            desc="🔄 加载数据数据", 
            unit="行",
            unit_scale=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            dynamic_ncols=True
        )
        
        for line_num, line in enumerate(load_progress, 1):
            try:
                line = line.strip()
                if not line:
                    continue
                    
                json_obj = json.loads(line)
                doc_id = json_obj.get("id", f"doc_{line_num}")
                metadata = json_obj.get("metadata", {})
                title = metadata.get("title", "")
                url = metadata.get("url", doc_id)
                text = json_obj.get("text", "")
                
                if text.strip():
                    # 清理所有字段中的有问题字符
                    clean_doc_id = clean_text(str(doc_id))
                    clean_title = clean_text(str(title))
                    clean_url = clean_text(str(url))
                    cleaned_text = clean_text(str(text))

                    data.append({
                        "id": clean_doc_id,
                        "title": clean_title,
                        "url": clean_url,
                        "text": cleaned_text
                    })
                    
                    # 更新进度条状态
                    load_progress.set_postfix({
                        '有效记录': len(data),
                        '错误': error_count,
                        '当前': clean_doc_id[:15] if clean_doc_id else f'line_{line_num}',
                        '文本长度': f'{len(clean_text):,}'
                    })
                
            except json.JSONDecodeError as e:
                error_count += 1
                # 更新进度条状态 - JSON错误
                load_progress.set_postfix({
                    '有效记录': len(data),
                    '❌JSON错误': error_count,
                    '行': line_num,
                    '错误': str(e)[:20]
                })
                if error_count <= 10:
                    tqdm.write(f"⚠️  行 {line_num} JSON解析错误: {str(e)[:100]}...")
                elif error_count == 11:
                    tqdm.write("⚠️  更多JSON解析错误将不再显示...")
                continue
                
            except Exception as e:
                error_count += 1
                # 更新进度条状态 - 其他错误
                load_progress.set_postfix({
                    '有效记录': len(data),
                    '❌其他错误': error_count,
                    '行': line_num,
                    '错误': str(e)[:20]
                })
                if error_count <= 10:
                    tqdm.write(f"⚠️  行 {line_num} 其他错误: {str(e)[:100]}...")
                continue
        
        # 关闭进度条
        load_progress.close()
        f.close()
        
    except Exception as e:
        print(f"❌ 文件读取失败: {e}")
        raise e
    
    print(f"✅ 成功加载 {len(data):,} 条记录")
    if error_count > 0:
        print(f"⚠️  跳过了 {error_count:,} 行有问题的数据")
        if len(data) > 0:
            print(f"📊 数据完整性: {len(data)/(len(data)+error_count)*100:.2f}%")
    
    return data


def load_data_corpus_with_monitoring(file_path: str, return_data: bool = True) -> list:
    """
    以流式方式高效读取数据数据集，带有进度条、计时、内存/显存监控和清理
    基于 data_load_time.py 的实现，但适配了 数据 数据格式
    
    Args:
        file_path (str): 数据文件路径（支持 .json 和 .json.gz）
        return_data (bool): 是否返回数据，False时只做监控测试
        
    Returns:
        list: 文档数据列表（如果 return_data=True）
    """
    print(f"📚 带监控加载数据数据集: {file_path}")
    
    # 初始化NVIDIA监控
    gpu_monitoring_active = False
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            gpu_monitoring_active = True
            print("✅ NVIDIA GPU监控已启用")
        except Exception as e:
            print(f"⚠️ NVIDIA GPU监控启动失败: {e}")
    else:
        print("⚠️ 未安装pynvml，无法监控显存")
    
    try:
        # 步骤 1: 预扫描获取总行数
        print(f"🔍 开始预扫描文件以统计总行数: '{file_path}'")
        is_gzip = is_gzip_file(file_path)
        
        if is_gzip:
            file_opener = lambda: gzip.open(file_path, 'rt', encoding='utf-8')
            file_type = "gzip压缩文件"
        else:
            file_opener = lambda: open(file_path, 'r', encoding='utf-8')
            file_type = "普通文件"
        
        print(f"📁 文件类型: {file_type}")
        
        with file_opener() as f:
            total_lines = sum(1 for _ in tqdm(f, desc="🔢 正在计数", unit=" 行"))
            
    except FileNotFoundError:
        print(f"❌ 文件 '{file_path}' 不存在")
        if gpu_monitoring_active: 
            pynvml.nvmlShutdown()
        return []
    except Exception as e:
        print(f"❌ 文件访问失败: {e}")
        if gpu_monitoring_active: 
            pynvml.nvmlShutdown()
        return []

    # 步骤 2: 开始计时并准备读取
    start_time = time.time()
    data_list = [] if return_data else None
    valid_count = 0
    error_count = 0
    
    print(f"📊 文件共 {total_lines:,} 行，开始读取与解析...")
    ram_initial, gpu_initial = get_memory_info()
    print(f"📈 初始内存占用 - RAM: {ram_initial} | VRAM: {gpu_initial}")
    
    # 步骤 3: 读取文件内容
    try:
        with file_opener() as f:
            # 使用tqdm显示进度
            with tqdm(total=total_lines, desc="📖 正在读取", unit=" 行") as pbar:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                            
                        json_obj = json.loads(line)
                        
                        # 提取数据格式数据
                        doc_id = json_obj.get("id", f"doc_{line_num}")
                        metadata = json_obj.get("metadata", {})
                        title = metadata.get("title", "")
                        url = metadata.get("url", doc_id)
                        text = json_obj.get("text", "")
                        
                        if text.strip():
                            valid_count += 1
                            
                            if return_data:
                                # 清理所有字段中的有问题字符
                                clean_doc_id = clean_text(str(doc_id))
                                clean_title = clean_text(str(title))
                                clean_url = clean_text(str(url))
                                cleaned_text = clean_text(str(text))

                                data_list.append({
                                    "id": clean_doc_id,
                                    "title": clean_title,
                                    "url": clean_url,
                                    "text": cleaned_text
                                })
                            
                    except json.JSONDecodeError:
                        error_count += 1
                    except Exception:
                        error_count += 1
                    
                    # 每 1000 次迭代更新一次内存/显存信息，以减少性能开销
                    if pbar.n % 1000 == 0:
                        ram, gpu = get_memory_info()
                        pbar.set_postfix({
                            'RAM': ram, 
                            'VRAM': gpu, 
                            '有效': valid_count,
                            '错误': error_count
                        })
                    pbar.update(1)

        elapsed_time = time.time() - start_time
        ram_final, gpu_final = get_memory_info()  # 获取最终的内存占用
        
        print(f"\n✅ 读取完毕")
        print(f"📊 统计信息:")
        print(f"   - 总行数: {total_lines:,}")
        print(f"   - 有效记录: {valid_count:,}")
        print(f"   - 错误记录: {error_count:,}")
        print(f"   - 数据完整性: {valid_count/(valid_count+error_count)*100:.2f}%")
        print(f"⏱️  耗时: {elapsed_time:.2f} 秒")
        print(f"🚀 处理速度: {valid_count/elapsed_time:.0f} 记录/秒")
        print(f"📈 内存变化 - RAM: {ram_initial} → {ram_final}")
        print(f"📈 显存变化 - VRAM: {gpu_initial} → {gpu_final}")

        # 步骤 4: 按要求清理内存（如果不返回数据）
        if not return_data and data_list is not None:
            print("🧹 正在释放内存...")
            del data_list
            gc.collect()
            
            # 获取清理后的内存占用
            ram_cleaned, _ = get_memory_info()
            print(f"🧹 内存已释放。当前进程内存占用 (RAM): {ram_cleaned}")

        return data_list if return_data else []
        
    except KeyboardInterrupt:
        print(f"\n🛑 用户中断了数据加载")
        raise
    except Exception as e:
        print(f"❌ 数据读取过程中发生错误: {e}")
        raise e
    finally:
        # 关闭NVIDIA监控
        if gpu_monitoring_active:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def read_jsonl_gz_monitoring_only(file_path: str):
    """
    仅用于监控测试的函数，不返回数据
    完全复制 data_load_time.py 的行为
    
    Args:
        file_path (str): 文件路径
    """
    print(f"🔍 纯监控模式 - 不返回数据，仅测试性能: {file_path}")
    load_data_corpus_with_monitoring(file_path, return_data=False)