"""PostgreSQL数据存储功能 - 用于URL检索"""

import psycopg2
import psycopg2.extras
from psycopg2 import pool
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
import csv

from .config_loader import (
    ENABLE_SQL_STORAGE, SQL_HOST, SQL_PORT, SQL_DATABASE, SQL_USER, SQL_PASSWORD,
    SQL_SCHEMA, SQL_TABLE, SQL_URL_MAX_LENGTH,
    SQL_ENABLE_FULL_TEXT_SEARCH, SQL_BATCH_SIZE, SQL_USE_COPY_INSERT,
    SQL_ENABLE_MULTITHREAD_DEDUP, SQL_DEDUP_THREAD_COUNT
)
from .utils import safe_truncate


def _clean_text_for_postgres(text: str) -> str:
    """
    清理文本中的PostgreSQL不兼容字符
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除NULL字符（0x00）
    cleaned_text = text.replace('\x00', '')
    
    # 移除其他可能有问题的控制字符
    # 保留常见的换行符、制表符等
    import re
    # 移除除了常见空白字符外的所有控制字符
    cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned_text)
    
    return cleaned_text


def _detect_problematic_characters(text: str) -> dict:
    """
    检测文本中可能有问题的字符
    
    Args:
        text: 要检测的文本
        
    Returns:
        包含问题字符统计的字典
    """
    if not text:
        return {}
    
    problems = {}
    
    # 检测NULL字符
    null_count = text.count('\x00')
    if null_count > 0:
        problems['null_chars'] = null_count
    
    # 检测其他控制字符
    import re
    control_chars = re.findall(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', text)
    if control_chars:
        problems['control_chars'] = len(control_chars)
        problems['control_char_types'] = len(set(control_chars))
    
    # 检测非ASCII字符
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii > 0:
        problems['non_ascii_chars'] = non_ascii
    
    return problems


def _analyze_documents_batch(documents_batch: List[Dict[str, Any]]) -> Tuple[int, int, List[int], List[int]]:
    """
    多线程分析一批文档的统计信息
    
    Args:
        documents_batch: 文档批次
        
    Returns:
        (有效文档数, 总文本长度, URL长度列表, 标题长度列表)
    """
    valid_docs = 0
    total_text_length = 0
    url_lengths = []
    title_lengths = []
    
    for doc in documents_batch:
        if doc.get('url') and doc.get('text'):
            valid_docs += 1
            text_len = len(doc.get('text', ''))
            total_text_length += text_len
            url_lengths.append(len(doc.get('url', '')))
            title_lengths.append(len(doc.get('title', '')))
    
    return valid_docs, total_text_length, url_lengths, title_lengths


def _analyze_documents_multithread(documents: List[Dict[str, Any]], num_workers: int = 48) -> Tuple[int, int, List[int], List[int]]:
    """
    多线程分析文档统计信息
    
    Args:
        documents: 文档列表
        num_workers: 工作线程数
        
    Returns:
        (有效文档数, 总文本长度, URL长度列表, 标题长度列表)
    """
    print(f"🚀 多线程数据预处理分析: {len(documents):,} 个文档，{num_workers} 个工作线程")
    
    # 计算批次大小
    batch_size = max(1000, len(documents) // (num_workers * 2))
    num_batches = math.ceil(len(documents) / batch_size)
    
    total_valid_docs = 0
    total_text_length = 0
    all_url_lengths = []
    all_title_lengths = []
    
    # 创建进度条
    progress_bar = tqdm(
        total=num_batches,
        desc="🚀 多线程预处理分析",
        unit="batch",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
        dynamic_ncols=True
    )
    
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 注册executor到进程管理器
            try:
                from .utils import get_process_manager
                process_manager = get_process_manager()
                process_manager.register_executor(executor)
            except ImportError:
                process_manager = None
            
            # 提交所有批次任务
            futures = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                future = executor.submit(_analyze_documents_batch, batch)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    # 检查是否正在关闭
                    if process_manager and process_manager.is_shutting_down:
                        print("🛑 检测到关闭信号，停止处理剩余批次")
                        break
                        
                    valid_docs, text_length, url_lengths, title_lengths = future.result(timeout=30)
                    
                    total_valid_docs += valid_docs
                    total_text_length += text_length
                    all_url_lengths.extend(url_lengths)
                    all_title_lengths.extend(title_lengths)
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        '有效文档': total_valid_docs,
                        '平均长度': f'{total_text_length // max(total_valid_docs, 1):,}字符',
                        '线程': num_workers
                    })
                    progress_bar.update(1)
                    
                except Exception as e:
                    if not (process_manager and process_manager.is_shutting_down):
                        tqdm.write(f"⚠️  批次分析失败: {e}")
        
        progress_bar.close()
        
    except KeyboardInterrupt:
        print(f"\n🛑 用户中断了多线程预处理分析")
        progress_bar.close()
        raise
    except Exception as e:
        print(f"❌ 多线程预处理分析失败: {e}")
        progress_bar.close()
        # 降级到单线程处理
        return _analyze_documents_single_thread(documents)
    
    print(f"✅ 多线程预处理分析完成: {total_valid_docs:,} 个有效文档")
    return total_valid_docs, total_text_length, all_url_lengths, all_title_lengths


def _analyze_documents_single_thread(documents: List[Dict[str, Any]]) -> Tuple[int, int, List[int], List[int]]:
    """
    单线程分析文档统计信息
    
    Args:
        documents: 文档列表
        
    Returns:
        (有效文档数, 总文本长度, URL长度列表, 标题长度列表)
    """
    print(f"📊 单线程数据预处理分析: {len(documents):,} 个文档")
    
    valid_docs = 0
    total_text_length = 0
    url_lengths = []
    title_lengths = []
    
    preprocess_progress = tqdm(documents, desc="📊 分析文档", unit="doc")
    
    for doc in preprocess_progress:
        if doc.get('url') and doc.get('text'):
            valid_docs += 1
            text_len = len(doc.get('text', ''))
            total_text_length += text_len
            url_lengths.append(len(doc.get('url', '')))
            title_lengths.append(len(doc.get('title', '')))
            
        preprocess_progress.set_postfix({
            '有效文档': valid_docs,
            '平均长度': f'{total_text_length // max(valid_docs, 1):,}字符'
        })
    
    preprocess_progress.close()
    return valid_docs, total_text_length, url_lengths, title_lengths


class PostgreSQLStorage:
    """PostgreSQL存储管理类"""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """连接到PostgreSQL数据库"""
        try:
            self.connection = psycopg2.connect(
                host=SQL_HOST,
                port=SQL_PORT,
                database=SQL_DATABASE,
                user=SQL_USER,
                password=SQL_PASSWORD
            )
            self.connection.autocommit = False
            self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            print(f"✅ 成功连接到PostgreSQL: {SQL_HOST}:{SQL_PORT}/{SQL_DATABASE}")
            return True
        except Exception as e:
            print(f"❌ PostgreSQL连接失败: {e}")
            return False
    
    def create_schema_and_table(self):
        """创建Schema和表结构（不创建索引）"""
        try:
            # 创建Schema
            self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {SQL_SCHEMA}")
            print(f"✅ Schema '{SQL_SCHEMA}' 已创建或已存在")
            
            # 创建表结构 - 统一使用标准模式（完整文档）
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {SQL_SCHEMA}.{SQL_TABLE} (
                id SERIAL PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            self.cursor.execute(create_table_sql)
            print(f"✅ 表 '{SQL_SCHEMA}.{SQL_TABLE}' 已创建或已存在")
            
            # # 🔧 检查并修复现有表的字段类型
            # self._ensure_table_schema_correct()
            
            print(f"📝 注意: 索引将在数据插入完成后创建以提高性能")
            
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"❌ 创建Schema和表失败: {e}")
            self.connection.rollback()
            return False
    
    def _create_indexes(self):
        """在数据插入完成后创建索引，带进度提示"""
        try:
            print(f"📊 开始创建数据库索引...")
            
            # 1. URL索引（快速）
            print(f"🔍 创建URL索引...")
            url_index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{SQL_TABLE}_url 
            ON {SQL_SCHEMA}.{SQL_TABLE}(url)
            """
            start_time = time.time()
            self.cursor.execute(url_index_sql)
            url_time = time.time() - start_time
            print(f"✅ URL索引创建完成 ({url_time:.2f}秒)")
            
            # 2. 全文搜索索引（耗时较长）
            if SQL_ENABLE_FULL_TEXT_SEARCH:
                print(f"🔍 创建全文搜索GIN索引...")
                print(f"⚠️  注意: 全文索引创建可能需要较长时间，请耐心等待...")
                
                # 获取表的大致记录数
                self.cursor.execute(f"SELECT COUNT(*) FROM {SQL_SCHEMA}.{SQL_TABLE}")
                record_count = self.cursor.fetchone()[0]
                print(f"📊 当前记录数: {record_count:,} 条")
                
                if record_count > 1000000:  # 100万条以上
                    print(f"⚠️  大数据集检测！预计索引创建时间: {record_count // 100000:.1f}-{record_count // 50000:.1f} 分钟")
                
                fulltext_index_sql = f"""
                CREATE INDEX IF NOT EXISTS idx_{SQL_TABLE}_fulltext 
                ON {SQL_SCHEMA}.{SQL_TABLE} USING gin(to_tsvector('english', title || ' ' || text))
                """
                
                fulltext_start = time.time()
                self.cursor.execute(fulltext_index_sql)
                fulltext_time = time.time() - fulltext_start
                
                print(f"✅ 全文搜索索引创建完成 ({fulltext_time:.2f}秒)")
                
                if fulltext_time > 60:
                    print(f"😅 全文索引创建耗时较长，但将显著提升搜索性能！")
            else:
                print(f"🚫 全文搜索索引已禁用")
            
            total_time = time.time() - start_time if 'start_time' in locals() else 0
            print(f"✅ 所有索引创建完成 (总耗时: {total_time:.2f}秒)")
            
        except Exception as e:
            print(f"⚠️  创建索引时出现警告: {e}")
            print(f"📝 索引创建失败不会影响数据插入，但可能影响查询性能")
    

    

    

    
    def insert_documents_batch(self, documents: List[Dict[str, Any]]) -> bool:
        """批量插入文档 - 根据配置选择COPY或INSERT，并在完成后创建索引"""
        if not documents:
            return True
            
        try:
            print(f"💾 开始插入 {len(documents):,} 条文档...")
            
            # 根据配置选择插入方法
            if SQL_USE_COPY_INSERT:
                insert_success = self._insert_documents_copy_method(documents)
            else:
                insert_success = self._insert_standard_documents_batch(documents)
                
            # 如果数据插入成功，创建索引
            if insert_success:
                print(f"📊 数据插入完成，开始创建索引...")
                self._create_indexes()
                self.connection.commit()  # 确保索引也被提交
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ 批量插入文档失败: {e}")
            self.connection.rollback()
            return False


    def _insert_documents_copy_method(self, documents: List[Dict[str, Any]]) -> bool:
        """使用COPY命令分批次插入文档 - 多小批次处理，带进度条"""
        try:
            print(f"🚀 使用COPY命令分批次插入: {len(documents):,} 条记录")
            
            # 计算批次配置
            batch_size = SQL_BATCH_SIZE
            total_batches = math.ceil(len(documents) / batch_size)
            
            print(f"📊 分批次COPY配置:")
            print(f"   批次大小: {batch_size:,} 条记录")
            print(f"   总批次数: {total_batches:,}")
            print(f"   处理模式: 多小批次，实时进度显示")
            
            total_valid_rows = 0
            total_inserted = 0
            
            # 开始事务
            self.connection.autocommit = False
            
            # 创建批次进度条
            batch_progress = tqdm(
                range(total_batches),
                desc="💾 COPY批次插入",
                unit="batch",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                dynamic_ncols=True
            )
            
            # 逐批次处理
            for batch_idx in batch_progress:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(documents))
                batch_documents = documents[start_idx:end_idx]
                
                # 更新进度条状态 - 准备阶段
                batch_progress.set_postfix({
                    '阶段': '数据准备',
                    '当前批次': f'{batch_idx + 1}/{total_batches}',
                    '已插入': f'{total_inserted:,}'
                })
                
                # 为当前批次创建缓冲区
                copy_buffer = StringIO()
                csv_writer = csv.writer(copy_buffer, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                
                batch_valid_rows = 0
                batch_url_seen = set()  # 当前批次内的URL去重
                batch_duplicate_count = 0
                
                # 处理当前批次的文档（跳过清理，数据已在加载时清理过）
                for doc in batch_documents:
                    url = safe_truncate(doc.get('url', ''), SQL_URL_MAX_LENGTH)
                    title = doc.get('title', '')
                    text = doc.get('text', '')
                    
                    if url and text:  # 确保有URL和文本内容
                        # 批次内去重：避免同一批次内的重复URL导致COPY失败
                        if url not in batch_url_seen:
                            batch_url_seen.add(url)
                            csv_writer.writerow([url, title, text])
                            batch_valid_rows += 1
                        else:
                            batch_duplicate_count += 1
                
                if batch_valid_rows == 0:
                    copy_buffer.close()
                    continue
                
                # 更新进度条状态 - COPY阶段
                batch_progress.set_postfix({
                    '阶段': 'COPY插入',
                    '当前批次': f'{batch_idx + 1}/{total_batches}',
                    '本批次': f'{batch_valid_rows:,}条',
                    '已插入': f'{total_inserted:,}'
                })
                
                # 重置缓冲区位置
                copy_buffer.seek(0)
                
                # 使用COPY命令插入当前批次
                copy_sql = f"""
                COPY {SQL_SCHEMA}.{SQL_TABLE} (url, title, text)
                FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '', QUOTE '"')
                """
                
                batch_start = time.time()
                self.cursor.copy_expert(copy_sql, copy_buffer)
                batch_time = time.time() - batch_start
                batch_throughput = batch_valid_rows / batch_time if batch_time > 0 else 0
                
                # 释放当前批次的缓冲区
                copy_buffer.close()
                del copy_buffer
                
                # 更新统计
                total_valid_rows += batch_valid_rows
                total_inserted += batch_valid_rows
                
                # 更新进度条最终状态
                batch_progress.set_postfix({
                    '阶段': '完成',
                    '当前批次': f'{batch_idx + 1}/{total_batches}',
                    '本批次': f'{batch_valid_rows:,}条',
                    '已插入': f'{total_inserted:,}',
                    '速度': f'{batch_throughput:.0f}/s'
                })
                
                # 显示批次去重统计
                if batch_duplicate_count > 0:
                    tqdm.write(f"    批次 {batch_idx + 1}: 跳过了 {batch_duplicate_count} 个重复URL")
            
            # 关闭批次进度条
            batch_progress.close()
            
            if total_valid_rows == 0:
                print("⚠️  没有有效数据可插入")
                return True
            
            # 处理重复URL
            print("🔄 处理重复URL...")
            self._handle_duplicate_urls_after_copy()
            
            # 提交事务
            self.connection.commit()
            self.connection.autocommit = True
            
            print(f"✅ 分批次COPY插入完成:")
            print(f"   总记录数: {total_valid_rows:,}")
            print(f"   总批次数: {total_batches:,}")
            print(f"   批次大小: {batch_size:,}")
            print(f"   处理模式: 多小批次COPY插入（含批次内去重）")
            
            return True
            
        except Exception as e:
            print(f"❌ COPY命令插入失败: {e}")
            try:
                self.connection.rollback()
                self.connection.autocommit = True
            except:
                pass
            
            # 降级到传统批量插入方法
            print("🔄 降级到传统批量插入方法...")
            return self._insert_standard_documents_batch(documents)
    
    def _handle_duplicate_urls_after_copy(self):
        """COPY后处理重复URL问题"""
        try:
            # 删除重复的URL，只保留最新的记录
            dedup_sql = f"""
            DELETE FROM {SQL_SCHEMA}.{SQL_TABLE} a
            USING {SQL_SCHEMA}.{SQL_TABLE} b
            WHERE a.id < b.id AND a.url = b.url
            """
            self.cursor.execute(dedup_sql)
            deleted_count = self.cursor.rowcount
            
            if deleted_count > 0:
                print(f"🔄 删除了 {deleted_count} 个重复URL记录")
                
        except Exception as e:
            print(f"⚠️  处理重复URL失败: {e}")

    def _prepare_documents_batch_multithread(self, documents: List[Dict[str, Any]], num_workers: int = 16) -> tuple:
        """
        多线程处理文档批次去重
        
        Args:
            documents: 文档列表
            num_workers: 工作线程数
            
        Returns:
            tuple: (values列表, 重复数量)
        """
        import math
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_docs_chunk(docs_chunk):
            """处理一块文档数据"""
            chunk_url_to_doc = {}
            chunk_duplicate_count = 0
            
            for doc in docs_chunk:
                url = safe_truncate(doc.get('url', ''), SQL_URL_MAX_LENGTH)
                title = doc.get('title', '')
                text = doc.get('text', '')
                
                if url and text:  # 确保有URL和文本内容
                    # 块内去重：如果URL已存在，替换为新数据（保留最后一个）
                    if url in chunk_url_to_doc:
                        chunk_duplicate_count += 1
                    chunk_url_to_doc[url] = (title, text)
            
            return chunk_url_to_doc, chunk_duplicate_count
        
        # 计算块大小
        total_docs = len(documents)
        chunk_size = max(1000, total_docs // (num_workers * 2))  # 每个块至少1000个文档
        chunks = [documents[i:i + chunk_size] for i in range(0, total_docs, chunk_size)]
        
        print(f"🚀 多线程数据准备配置:")
        print(f"   工作线程数: {num_workers}")
        print(f"   数据块数: {len(chunks)}")
        print(f"   平均块大小: {chunk_size:,}")
        
        # 创建进度条
        progress_bar = tqdm(
            total=len(chunks),
            desc="🚀 多线程数据准备",
            unit="chunk",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            dynamic_ncols=True
        )
        
        final_url_to_doc = {}
        total_duplicate_count = 0
        
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 注册executor到进程管理器（如果可用）
                try:
                    from .utils import get_process_manager
                    process_manager = get_process_manager()
                    process_manager.register_executor(executor)
                except ImportError:
                    process_manager = None
                
                # 提交所有任务
                futures = [executor.submit(process_docs_chunk, chunk) for chunk in chunks]
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        # 检查是否正在关闭
                        if process_manager and process_manager.is_shutting_down:
                            print("🛑 检测到关闭信号，停止处理剩余数据块")
                            break
                        
                        chunk_url_to_doc, chunk_duplicate_count = future.result(timeout=30)
                        
                        # 合并结果，处理跨块的重复
                        for url, (title, text) in chunk_url_to_doc.items():
                            if url in final_url_to_doc:
                                total_duplicate_count += 1  # 跨块重复
                            final_url_to_doc[url] = (title, text)
                        
                        total_duplicate_count += chunk_duplicate_count
                        
                        # 更新进度条
                        progress_bar.set_postfix({
                            '唯一URL': len(final_url_to_doc),
                            '重复URL': total_duplicate_count,
                            '线程': num_workers
                        })
                        progress_bar.update(1)
                        
                    except Exception as e:
                        if not (process_manager and process_manager.is_shutting_down):
                            tqdm.write(f"⚠️  数据块处理失败: {e}")
                        progress_bar.update(1)
        
        except KeyboardInterrupt:
            print(f"\n🛑 用户中断了多线程数据准备")
            progress_bar.close()
            raise
        except Exception as e:
            print(f"❌ 多线程数据准备失败: {e}")
            progress_bar.close()
            raise
        
        progress_bar.close()
        
        # 转换为values列表
        values = [(url, title, text) for url, (title, text) in final_url_to_doc.items()]
        
        print(f"✅ 多线程数据准备完成:")
        print(f"   处理文档: {total_docs:,}")
        print(f"   唯一记录: {len(values):,}")
        print(f"   重复记录: {total_duplicate_count:,}")
        print(f"   去重率: {(total_duplicate_count / total_docs * 100):.1f}%")
        
        return values, total_duplicate_count

    def _prepare_documents_batch_singlethread(self, documents: List[Dict[str, Any]]) -> tuple:
        """
        单线程处理文档批次去重（原始方法，作为后备）
        
        Args:
            documents: 文档列表
            
        Returns:
            tuple: (values列表, 重复数量)
        """
        # 使用字典进行批次内去重：同一batch内相同URL只保留最后一个
        url_to_doc = {}  # URL -> (title, text) 映射
        duplicate_count = 0
        
        # 创建数据准备进度条
        doc_progress = tqdm(
            documents,
            desc="📝 单线程数据准备",
            unit="doc",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
            dynamic_ncols=True
        )
        
        for doc in doc_progress:
            url = safe_truncate(doc.get('url', ''), SQL_URL_MAX_LENGTH)
            title = doc.get('title', '')
            text = doc.get('text', '')
            
            # 更新进度条状态
            doc_progress.set_postfix({
                '唯一URL': len(url_to_doc),
                '重复URL': duplicate_count,
                'URL': url[:20] + '...' if len(url) > 20 else url
            })
            
            if url and text:  # 确保有URL和文本内容
                # 批次内去重：如果URL已存在，替换为新数据（保留最后一个）
                if url in url_to_doc:
                    duplicate_count += 1
                url_to_doc[url] = (title, text)
        
        # 关闭数据准备进度条
        doc_progress.close()
        
        # 转换为values列表
        values = [(url, title, text) for url, (title, text) in url_to_doc.items()]
        
        return values, duplicate_count
    
    def _insert_standard_documents_batch(self, documents: List[Dict[str, Any]]) -> bool:
        """使用INSERT分批次插入文档，支持多线程数据准备"""
        try:
            print(f"📝 使用INSERT分批次插入: {len(documents):,} 条记录")
            print(f"🚀 处理模式: 多线程数据准备 + 多小批次插入")
            
            insert_sql = f"""
            INSERT INTO {SQL_SCHEMA}.{SQL_TABLE} (url, title, text)
            VALUES %s
            ON CONFLICT (url) DO UPDATE SET
                title = EXCLUDED.title,
                text = EXCLUDED.text,
                created_at = CURRENT_TIMESTAMP
            """
            
            # 🚀 根据配置选择数据准备方式
            if SQL_ENABLE_MULTITHREAD_DEDUP and len(documents) > 10000:  # 大于1万条记录才启用多线程
                import os
                if SQL_DEDUP_THREAD_COUNT == 0:
                    num_workers = min(16, os.cpu_count() or 4)  # 自动选择
                else:
                    num_workers = min(SQL_DEDUP_THREAD_COUNT, os.cpu_count() or 4)  # 使用配置值
                
                print(f"🧵 启用多线程数据准备: {num_workers} 个工作线程")
                values, duplicate_count = self._prepare_documents_batch_multithread(documents, num_workers)
            else:
                # 单线程数据准备（原始方法）
                print(f"📝 使用单线程数据准备（文档数量: {len(documents):,}）")
                values, duplicate_count = self._prepare_documents_batch_singlethread(documents)
            
            print(f"📊 数据准备完成: {len(values):,} 条唯一记录（跳过清理，已在加载时处理）")
            if duplicate_count > 0:
                print(f"🔄 批次去重统计: 跳过了 {duplicate_count:,} 个重复URL（保留最后出现的）")
            
            if not values:
                print("⚠️  没有有效数据可插入")
                return True
            
            # 分批INSERT插入，显示进度
            total_values = len(values)
            batch_size = SQL_BATCH_SIZE
            total_batches = (total_values + batch_size - 1) // batch_size
            
            print(f"📊 INSERT分批配置:")
            print(f"   批次大小: {batch_size:,} 条记录")
            print(f"   总批次数: {total_batches:,}")
            
            # 创建INSERT批次进度条
            insert_progress = tqdm(
                range(total_batches),
                desc="📤 INSERT批次执行",
                unit="batch",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                dynamic_ncols=True
            )
            
            inserted_count = 0
            
            for batch_idx in insert_progress:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_values)
                batch_values = values[start_idx:end_idx]
                
                # 更新进度条状态
                insert_progress.set_postfix({
                    '当前批次': f'{batch_idx + 1}/{total_batches}',
                    '本批次': f'{len(batch_values):,}条',
                    '已插入': f'{inserted_count:,}条'
                })
                
                # 执行当前批次插入
                batch_start = time.time()
                psycopg2.extras.execute_values(
                    self.cursor, insert_sql, batch_values, template=None, page_size=len(batch_values)
                )
                batch_time = time.time() - batch_start
                batch_throughput = len(batch_values) / batch_time if batch_time > 0 else 0
                
                inserted_count += len(batch_values)
                
                # 更新进度条最终状态
                insert_progress.set_postfix({
                    '当前批次': f'{batch_idx + 1}/{total_batches}',
                    '本批次': f'{len(batch_values):,}条',
                    '已插入': f'{inserted_count:,}条',
                    '速度': f'{batch_throughput:.0f}/s'
                })
            
            # 关闭INSERT进度条
            insert_progress.close()
            
            self.connection.commit()
            print(f"✅ 分批次INSERT插入完成: {inserted_count:,} 条记录")
            return True
            
        except Exception as e:
            try:
                self.connection.rollback()
            except:
                pass
                
            error_msg = str(e)
            print(f"\n❌ INSERT一次性插入失败: {error_msg}")
            
            # 如果仍然有字符编码问题，尝试更激进的清理
            if "NUL" in error_msg or "0x00" in error_msg:
                print(f"⚠️  检测到NULL字符问题，尝试更激进的清理...")
                
                try:
                    # 如果还有NULL字符问题，这表示data_loader清理不彻底，进行备用清理
                    print(f"🧹 检测到data_loader清理不彻底，进行备用清理...")
                    cleaned_values = []
                    for url, title, text in values:
                        # 更激进的清理：只保留可打印字符
                        import string
                        printable = set(string.printable)
                        
                        clean_url = ''.join(filter(lambda x: x in printable, url))
                        clean_title = ''.join(filter(lambda x: x in printable, title))
                        clean_text = ''.join(filter(lambda x: x in printable, text))
                        
                        cleaned_values.append((clean_url, clean_title, clean_text))
                    
                    # 使用分批方式再次尝试插入
                    print(f"💾 使用分批方式再次尝试INSERT...")
                    
                    # 分批插入清理后的数据
                    batch_size = SQL_BATCH_SIZE
                    total_batches = (len(cleaned_values) + batch_size - 1) // batch_size
                    inserted_count = 0
                    
                    for i in range(0, len(cleaned_values), batch_size):
                        batch = cleaned_values[i:i + batch_size]
                        psycopg2.extras.execute_values(
                            self.cursor, insert_sql, batch, template=None, page_size=len(batch)
                        )
                        inserted_count += len(batch)
                        print(f"   插入进度: {inserted_count:,}/{len(cleaned_values):,}")
                    
                    self.connection.commit()
                    print(f"✅ 备用清理后成功插入 {len(cleaned_values):,} 条记录")
                    return True
                        
                except Exception as e2:
                    try:
                        self.connection.rollback()
                    except:
                        pass
                    print(f"❌ 备用清理后仍然失败: {e2}")
                    return False
            else:
                return False
    

    
    def get_document_by_url(self, url: str) -> Dict[str, Any]:
        """根据URL获取文档（返回url、title、text）"""
        try:
            query_sql = f"""
            SELECT url, title, text FROM {SQL_SCHEMA}.{SQL_TABLE}
            WHERE url = %s
            LIMIT 1
            """
            
            self.cursor.execute(query_sql, (url,))
            result = self.cursor.fetchone()
            
            if result:
                return dict(result)
            
            return {}
            
        except Exception as e:
            print(f"❌ 根据URL查询文档失败: {e}")
            return {}
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """全文搜索文档"""
        if not SQL_ENABLE_FULL_TEXT_SEARCH:
            print("⚠️  全文搜索未启用")
            return []
        
        try:
            search_sql = f"""
            SELECT *, ts_rank(to_tsvector('english', title || ' ' || text), query) as rank
            FROM {SQL_SCHEMA}.{SQL_TABLE}, to_tsquery('english', %s) query
            WHERE to_tsvector('english', title || ' ' || text) @@ query
            ORDER BY rank DESC
            LIMIT %s
            """
            
            # 简单的查询预处理
            processed_query = ' & '.join(query.split())
            
            self.cursor.execute(search_sql, (processed_query, limit))
            results = self.cursor.fetchall()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            print(f"❌ 全文搜索失败: {e}")
            return []
    
    def get_table_stats(self) -> Dict[str, Any]:
        """获取表统计信息 - 统一使用标准模式（完整文档）"""
        try:
            stats_sql = f"""
            SELECT 
                COUNT(*) as total_documents,
                COUNT(DISTINCT url) as unique_urls,
                MIN(created_at) as first_document,
                MAX(created_at) as last_document
            FROM {SQL_SCHEMA}.{SQL_TABLE}
            """
            
            self.cursor.execute(stats_sql)
            result = self.cursor.fetchone()
            
            stats = dict(result) if result else {}
            
            return stats
            
        except Exception as e:
            print(f"❌ 获取表统计信息失败: {e}")
            return {}
    
    def close(self):
        """关闭数据库连接"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            print("✅ PostgreSQL连接已关闭")
        except Exception as e:
            print(f"⚠️  关闭PostgreSQL连接时出现警告: {e}")


def insert_data_to_sql(data: list, enable_sql: bool = None) -> bool:
    """
    高层次SQL插入流程控制函数
    
    将原始数据（未分块）插入到PostgreSQL中，保持文档完整性。
    这个函数提供高层次的流程控制，包含完整的数据分析和进度监控。
    
    Args:
        data: 原始文档数据列表
        enable_sql: 是否启用SQL插入，None时使用配置文件设置
        
    Returns:
        bool: 是否插入成功
    """
    # 确定是否启用SQL存储
    should_enable_sql = enable_sql if enable_sql is not None else ENABLE_SQL_STORAGE
    
    if not should_enable_sql:
        print("⚠️  SQL存储未启用，跳过PostgreSQL插入")
        return True
    
    if not data:
        print("⚠️  没有数据需要插入到PostgreSQL")
        return True
    
    # 调用详细的分析和插入函数
    return insert_documents_to_postgres_with_analysis(data)


def insert_documents_to_postgres_with_analysis(documents: List[Dict[str, Any]]) -> bool:
    """
    将文档批量插入到PostgreSQL，包含详细的数据分析和进度监控
    
    插入逻辑：
    1. 数据预处理分析（统计有效文档、文本长度等）
    2. 连接PostgreSQL数据库
    3. 创建Schema和表结构（如果不存在）
    4. 分批插入完整文档数据（统一使用标准模式，不进行分块）
    5. 创建索引（URL索引、全文搜索索引等）
    6. 返回详细的插入统计信息
    
    Args:
        documents: 原始完整文档数据列表 [{"id": "xxx", "title": "xxx", "url": "xxx", "text": "xxx"}, ...]
        
    Returns:
        bool: 是否插入成功
    """
    import time
    from tqdm import tqdm
    
    if not ENABLE_SQL_STORAGE:
        print("⚠️  SQL存储未启用，跳过PostgreSQL插入")
        return True
    
    if not documents:
        print("⚠️  没有文档需要插入到PostgreSQL")
        return True
    
    print(f"\n📊 PostgreSQL完整文档插入（标准模式）")
    print(f"=" * 50)
    print(f"📋 插入配置:")
    print(f"   文档数量: {len(documents):,}")
    print(f"   SQL Schema: {SQL_SCHEMA}")
    print(f"   批次大小: {SQL_BATCH_SIZE}")
    print(f"   数据库: {SQL_HOST}:{SQL_PORT}/{SQL_DATABASE}")
    print(f"   存储模式: 标准模式（完整文档，不分块）")
    
    # 数据预处理统计 - 支持多线程加速
    print(f"\n🔍 数据预处理分析...")
    
    # 根据数据量决定是否使用多线程
    use_multithread = len(documents) > 10000  # 简化为固定阈值
    
    if use_multithread:
        from .config_loader import ENABLE_MULTITHREAD_FIELD_EXTRACTION, FIELD_EXTRACTION_WORKERS
        if ENABLE_MULTITHREAD_FIELD_EXTRACTION:
            valid_docs, total_text_length, url_lengths, title_lengths = _analyze_documents_multithread(
                documents, FIELD_EXTRACTION_WORKERS
            )
        else:
            valid_docs, total_text_length, url_lengths, title_lengths = _analyze_documents_single_thread(documents)
    else:
        valid_docs, total_text_length, url_lengths, title_lengths = _analyze_documents_single_thread(documents)
    
    # 详细统计报告
    print(f"\n✅ 预处理完成:")
    print(f"   有效文档: {valid_docs:,}/{len(documents):,} ({valid_docs/len(documents)*100:.1f}%)")
    print(f"   总文本长度: {total_text_length:,} 字符 ({total_text_length/1024/1024:.1f} MB)")
    print(f"   平均文档长度: {total_text_length // max(valid_docs, 1):,} 字符")
    if url_lengths:
        print(f"   URL长度范围: {min(url_lengths)} - {max(url_lengths)} 字符")
    if title_lengths:
        print(f"   标题长度范围: {min(title_lengths)} - {max(title_lengths)} 字符")
    
    if valid_docs == 0:
        print("❌ 没有有效文档可插入")
        return False
    
    start_time = time.time()
    
    try:
        # 调用底层插入函数
        print(f"\n💾 开始PostgreSQL批量插入...")
        success = insert_documents_to_postgres(documents)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if success:
            print(f"\n✅ PostgreSQL插入完成！")
            print(f"   总耗时: {total_time:.2f}秒")
            print(f"   插入速度: {valid_docs / total_time:.1f} 文档/秒")
            print(f"   文本处理速度: {total_text_length / total_time / 1024:.1f} KB/秒")
            print(f"=" * 50)
            return True
        else:
            print(f"❌ PostgreSQL插入失败")
            print(f"   总耗时: {total_time:.2f}秒")
            return False
            
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"❌ PostgreSQL插入过程中发生错误: {e}")
        print(f"   失败前耗时: {total_time:.2f}秒")
        return False


def insert_documents_to_postgres(documents: List[Dict[str, Any]]) -> bool:
    """
    将文档批量插入到PostgreSQL
    
    Args:
        documents: 文档列表
        
    Returns:
        bool: 是否插入成功
    """
    if not ENABLE_SQL_STORAGE:
        print("⚠️  SQL存储未启用，跳过PostgreSQL插入")
        return True
    
    if not documents:
        print("⚠️  没有文档需要插入到PostgreSQL")
        return True
    
    storage = PostgreSQLStorage()
    
    try:
        # 连接数据库
        if not storage.connect():
            return False
        
        # 创建Schema和表
        if not storage.create_schema_and_table():
            return False
        
        print(f"💾 开始插入 {len(documents):,} 条记录到PostgreSQL...")
        
        # 使用简化的插入策略
        success = storage.insert_documents_batch(documents)
        
        # 获取最终统计
        stats = storage.get_table_stats()
        
        print(f"✅ PostgreSQL插入完成！")
        print(f"   数据库记录: {stats.get('total_documents', 0):,}")
        print(f"   唯一URL: {stats.get('unique_urls', 0):,}")
        print(f"   存储模式: 标准模式（完整文档）")
        
        return success
        
    except Exception as e:
        print(f"❌ PostgreSQL插入过程失败: {e}")
        return False
    finally:
        storage.close()


def search_documents_by_url(url: str) -> Dict[str, Any]:
    """
    根据URL搜索文档
    
    Args:
        url: 文档URL
        
    Returns:
        Dict: 文档信息或None
    """
    if not ENABLE_SQL_STORAGE:
        print("⚠️  SQL存储未启用")
        return None
    
    storage = PostgreSQLStorage()
    
    try:
        if storage.connect():
            return storage.get_document_by_url(url)
        return None
    finally:
        storage.close()


def search_documents_fulltext(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    全文搜索文档
    
    Args:
        query: 搜索查询
        limit: 返回结果数量限制
        
    Returns:
        List: 搜索结果列表
    """
    if not ENABLE_SQL_STORAGE:
        print("⚠️  SQL存储未启用")
        return []
    
    storage = PostgreSQLStorage()
    
    try:
        if storage.connect():
            return storage.search_documents(query, limit)
        return []
    finally:
        storage.close()


# =================== 连接池实现 ===================

class SQLConnectionPool:
    """PostgreSQL连接池 - 全局单例"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.connection_pool = None
        self._initialized = False
        
    def initialize(self):
        """初始化连接池"""
        if self._initialized:
            return True
            
        try:
            print("🔗 初始化PostgreSQL连接池...")
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,  # 最小连接数
                maxconn=20,  # 最大连接数
                host=SQL_HOST,
                port=SQL_PORT,
                database=SQL_DATABASE,
                user=SQL_USER,
                password=SQL_PASSWORD,
                cursor_factory=psycopg2.extras.RealDictCursor
            )
            self._initialized = True
            print(f"✅ SQL连接池初始化成功 (2-20连接)")
            return True
        except Exception as e:
            print(f"❌ SQL连接池初始化失败: {e}")
            self.connection_pool = None
            return False
    
    def get_connection(self):
        """获取连接"""
        if not self._initialized or not self.connection_pool:
            return None
        try:
            return self.connection_pool.getconn()
        except Exception as e:
            print(f"⚠️ 获取数据库连接失败: {e}")
            return None
    
    def put_connection(self, connection):
        """归还连接"""
        if self.connection_pool:
            try:
                self.connection_pool.putconn(connection)
            except Exception as e:
                print(f"⚠️ 归还数据库连接失败: {e}")
    
    def close_all_connections(self):
        """关闭所有连接"""
        if self.connection_pool:
            try:
                self.connection_pool.closeall()
                print("✅ 所有数据库连接已关闭")
            except Exception as e:
                print(f"⚠️ 关闭连接池时出现警告: {e}")
            finally:
                self.connection_pool = None
                self._initialized = False

# 全局连接池实例
_connection_pool = None

def get_connection_pool():
    """获取连接池实例"""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = SQLConnectionPool()
    return _connection_pool

def initialize_sql_connection_pool():
    """初始化连接池 - 在服务器启动时调用"""
    if not ENABLE_SQL_STORAGE:
        print("⚠️ SQL存储未启用，跳过连接池初始化")
        return False
        
    pool = get_connection_pool()
    return pool.initialize()

def close_sql_connection_pool():
    """关闭连接池 - 在服务器关闭时调用"""
    global _connection_pool
    if _connection_pool:
        _connection_pool.close_all_connections()
        _connection_pool = None

def search_documents_by_url_fast(url: str) -> Dict[str, Any]:
    """
    使用连接池的快速URL查询 - 替代原来的search_documents_by_url
    """
    if not ENABLE_SQL_STORAGE:
        return {}
    
    pool = get_connection_pool()
    if not pool._initialized:
        # 连接池未初始化，降级到原方法
        return search_documents_by_url(url)
    
    connection = None
    try:
        connection = pool.get_connection()
        if not connection:
            # 无法获取连接，降级到原方法
            return search_documents_by_url(url)
        
        with connection.cursor() as cursor:
            query_sql = f"""
            SELECT url, title, text FROM {SQL_SCHEMA}.{SQL_TABLE}
            WHERE url = %s
            LIMIT 1
            """
            
            cursor.execute(query_sql, (url,))
            result = cursor.fetchone()
            
            if result:
                return dict(result)
            return {}
            
    except Exception as e:
        print(f"❌ 快速URL查询失败: {e}")
        # 发生错误时降级到原方法
        return search_documents_by_url(url)
    finally:
        if connection:
            pool.put_connection(connection)
