"""数据插入和处理功能"""

import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import List, Dict, Any, Tuple
import math

from .config_loader import (
    BATCH_SIZE, DEVICE, DB_MAX_URL_LENGTH,
    DOC_EMBEDDING_MAX_TOKENS, DOC_STORAGE_MAX_CHARS, DOC_STORAGE_MAX_TOKENS, DOC_USE_TOKEN_LIMIT,
    ENABLE_MULTIPROCESS_FIELD_EXTRACTION, MULTIPROCESS_WORKERS,
    USE_FP16
)
import numpy as np
from .utils import safe_truncate, get_process_manager, truncate_text_to_tokens


def smart_truncate_doc(text: str) -> str:
    """
    智能截断doc字段文本，根据配置选择token限制或字符限制
    
    Args:
        text (str): 原始文本
        
    Returns:
        str: 截断后的文本
    """
    if not text:
        return ""
    
    if DOC_USE_TOKEN_LIMIT:
        # 使用token限制（推荐方式）
        return truncate_text_to_tokens(text, DOC_STORAGE_MAX_TOKENS, use_bge_tokenizer=True)
    else:
        # 使用字符限制（向后兼容）
        return text[:DOC_STORAGE_MAX_CHARS]


def process_data_for_full_documents(data):
    """
    处理数据为完整文档模式
    
    Args:
        data: 文档列表
        
    Returns:
        处理后的完整文档列表
    """
    print("📄 使用完整文档模式（已禁用分块功能）")
    if DOC_USE_TOKEN_LIMIT:
        print(f"📊 文档配置: embedding最大{DOC_EMBEDDING_MAX_TOKENS} tokens, 存储doc前{DOC_STORAGE_MAX_TOKENS} tokens")
    else:
        print(f"📊 文档配置: embedding最大{DOC_EMBEDDING_MAX_TOKENS} tokens, 存储doc前{DOC_STORAGE_MAX_CHARS}字符")
    print(f"📝 Title处理: 自动截断到1024字符以确保Milvus兼容性")
    print(f"🔗 Embedding策略: 使用截断后的title生成向量，确保一致性")
    print(f"📈 文档统计: 共 {len(data):,} 个完整文档")
    return data


def process_items_batch(items_batch: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str], List[str], int, int, List[int]]:
    """
    多线程处理一批数据项的字段提取（完整文档模式）- 优化版本
    
    Args:
        items_batch: 数据项批次
        
    Returns:
        (urls, titles, docs, embedding_texts, truncated_urls, truncated_titles, embedding_lengths)
    """
    # 预先导入工具函数，避免循环内重复导入
    from .utils import get_token_count, truncate_text_to_tokens
    
    urls = []
    titles = []
    docs = []
    embedding_texts = []
    truncated_urls = 0
    truncated_titles = 0
    embedding_lengths = []  # 用于统计embedding文本长度
    
    # 预设一个较大的初始容量，减少list重新分配（Python中list会自动管理容量）
    batch_size = len(items_batch)
    
    for item in items_batch:
        title = item.get("title", "")
        url = item.get("url", item.get("id", ""))
        text = item.get("text", "")
        
        # 首先截断title到1024字符，确保后续所有处理都使用截断后的title
        truncated_title = safe_truncate(title, 1024)
        # 双重保险：确保截断后的title不超过1024字符
        if len(truncated_title) > 1024:
            truncated_title = truncated_title[:1024]
        if len(title) > 1024:
            truncated_titles += 1
        
        # embedding 文本: title + 正文；doc: 正文智能截断
        if truncated_title:
            embedding_text = f"Title: {truncated_title}\n\nContent: {text}"
        else:
            embedding_text = f"Title: \n\nContent: {text}"
        doc = smart_truncate_doc(text)

        # 记录embedding文本长度用于统计
        embedding_lengths.append(len(embedding_text))

        # 精确token计算 - 使用线程本地编码器，无锁竞争
        token_count = get_token_count(embedding_text)
        if token_count > DOC_EMBEDDING_MAX_TOKENS:
            # 允许稍微超出
            embedding_text = truncate_text_to_tokens(embedding_text, DOC_EMBEDDING_MAX_TOKENS+200)
        
        # 处理URL长度
        processed_url = safe_truncate(url, DB_MAX_URL_LENGTH)
        if len(str(url)) > DB_MAX_URL_LENGTH:
            truncated_urls += 1
        
        urls.append(processed_url)
        titles.append(truncated_title)  # 使用已截断的title
        docs.append(doc)
        embedding_texts.append(embedding_text)
    
    return urls, titles, docs, embedding_texts, truncated_urls, truncated_titles, embedding_lengths


def prepare_data_fields_multiprocess(process_data: List[Dict[str, Any]], num_workers: int = None) -> Dict[str, Any]:
    """
    多进程准备完整文档数据字段（突破GIL限制，适合CPU密集型任务）
    
    Args:
        process_data: 处理后的文档数据
        num_workers: 工作进程数
        
    Returns:
        dict: 包含urls、titles、docs、embedding_texts等字段的字典
    """
    if num_workers is None:
        num_workers = MULTIPROCESS_WORKERS if ENABLE_MULTIPROCESS_FIELD_EXTRACTION else mp.cpu_count()
    
    print(f"🚀 多进程准备完整文档字段（URL+title+doc+向量）... 工作进程: {num_workers}")
    
    # 优化批次大小计算 - 针对大数据集优化
    total_items = len(process_data)
    
    # 多进程模式下使用更大的批次以减少进程间通信开销
    if total_items > 5_000_000:  # 500万+数据
        batch_size = max(10000, min(50000, total_items // (num_workers * 4)))
    elif total_items > 1_000_000:  # 100万+数据
        batch_size = max(5000, min(30000, total_items // (num_workers * 2)))
    else:  # 小数据集
        batch_size = max(2000, total_items // num_workers)
    
    num_batches = math.ceil(total_items / batch_size)
    
    print(f"📊 多进程数据字段提取配置:")
    print(f"   总数据量: {total_items:,}")
    print(f"   批次大小: {batch_size:,}")
    print(f"   总批次数: {num_batches:,}")
    print(f"   工作进程: {num_workers}")
    
    # 结果收集
    all_urls = []
    all_titles = []
    all_docs = []
    all_embedding_texts = []
    total_truncated_urls = 0
    total_truncated_titles = 0
    all_embedding_lengths = []
    
    # 创建进度条
    progress_bar = tqdm(
        total=num_batches,
        desc="🚀 多进程字段提取",
        unit="batch",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
        dynamic_ncols=True
    )
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 使用spawn方法确保进程隔离（避免CUDA上下文问题）
        mp_context = mp.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
            print(f"📝 已启动ProcessPoolExecutor，进程数: {num_workers}")
            
            # 提交所有批次任务
            futures = []
            for i in range(0, total_items, batch_size):
                batch = process_data[i:i + batch_size]
                future = executor.submit(process_items_batch, batch)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    urls, titles, docs, embedding_texts, truncated_urls, truncated_titles, embedding_lengths = future.result(timeout=60)
                    
                    all_urls.extend(urls)
                    all_titles.extend(titles)
                    all_docs.extend(docs)
                    all_embedding_texts.extend(embedding_texts)
                    total_truncated_urls += truncated_urls
                    total_truncated_titles += truncated_titles
                    all_embedding_lengths.extend(embedding_lengths)
                    
                    # 更新进度条
                    completed_batches = len([f for f in futures if f.done()])
                    progress_percent = (completed_batches / len(futures)) * 100
                    docs_per_second = len(all_urls) / max(1, time.time() - start_time)
                    
                    progress_bar.set_postfix({
                        '已处理': f'{len(all_urls):,}',
                        '进度': f'{progress_percent:.1f}%',
                        '速度': f'{docs_per_second:.0f}/s',
                        '进程': num_workers
                    })
                    progress_bar.update(1)
                    
                except Exception as e:
                    tqdm.write(f"⚠️  批次字段提取失败: {e}")
        
        progress_bar.close()
        
    except KeyboardInterrupt:
        print(f"\n🛑 用户中断了多进程字段提取")
        progress_bar.close()
        raise
    except Exception as e:
        print(f"❌ 多进程字段提取失败: {e}")
        progress_bar.close()
        # 降级到单线程处理
        return prepare_data_fields(process_data)
    
    print(f"✅ 多进程字段提取完成:")
    print(f"   总处理数: {len(all_urls):,}")
    print(f"   完整文档模式: 是")
    print(f"   截断URL: {total_truncated_urls:,} 个")
    print(f"   截断Title: {total_truncated_titles:,} 个")
    
    # 添加embedding长度统计
    if all_embedding_lengths:
        avg_embedding_length = sum(all_embedding_lengths) / len(all_embedding_lengths)
        max_embedding_length = max(all_embedding_lengths)
        min_embedding_length = min(all_embedding_lengths)
        print(f"📊 标准模式Embedding统计:")
        print(f"   平均长度: {avg_embedding_length:.1f} 字符")
    
    if total_truncated_urls > 0:
        print(f"⚠️  共有 {total_truncated_urls:,} 个URL被截断以符合数据库限制")
    
    if total_truncated_titles > 0:
        print(f"📝 共有 {total_truncated_titles:,} 个Title被截断到1024字符以确保Milvus兼容性")
    
    print(f"✅ 准备处理: {len(all_urls):,} 个完整文档")
    
    result = {
        'urls': all_urls,
        'titles': all_titles,
        'docs': all_docs,
        'embedding_texts': all_embedding_texts
    }
    
    return result



def prepare_data_fields(process_data):
    """
    准备完整文档数据字段（单线程版本）
    
    Args:
        process_data: 处理后的文档数据
        
    Returns:
        dict: 包含urls、titles、docs、embedding_texts等字段的字典
    """
    print("📋 准备完整文档数据字段（URL+title+doc+向量）...")
    
    urls = []
    titles = []
    docs = []
    embedding_texts = []
    truncated_urls = 0
    truncated_titles = 0
    embedding_lengths = []  # 用于统计embedding长度

    # 创建数据字段提取的详细进度条
    field_progress = tqdm(
        process_data, 
        desc="📊 提取并处理数据字段", 
        unit="item",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
        dynamic_ncols=True
    )

    for item_idx, item in enumerate(field_progress):
        title = item.get("title", "")
        url = item.get("url", item.get("id", ""))
        text = item.get("text", "")
        
        # 首先截断title到1024字符，确保后续所有处理都使用截断后的title
        truncated_title = safe_truncate(title, 1024)
        # 双重保险：确保截断后的title不超过1024字符
        if len(truncated_title) > 1024:
            truncated_title = truncated_title[:1024]
        if len(title) > 1024:
            truncated_titles += 1
        
        # 更新进度条状态 - 完整文档模式
        field_progress.set_postfix({
            '模式': '完整文档',
            '当前': item.get("id", "unknown")[:15],
            '文本长度': f'{len(text):,}',
            'URL截断': truncated_urls,
            'Title截断': truncated_titles
        })
        
        # embedding 文本: title + 正文；doc: 正文智能截断
        if truncated_title:
            embedding_text = f"Title: {truncated_title}\n\nContent: {text}"
        else:
            embedding_text = f"Title: \n\nContent: {text}"
        doc = smart_truncate_doc(text)

        # 记录embedding文本长度用于统计
        embedding_lengths.append(len(embedding_text))

        # 截断到DOC_EMBEDDING_MAX_TOKENS
        from .utils import get_token_count, truncate_text_to_tokens
        token_count = get_token_count(embedding_text)
        if token_count > DOC_EMBEDDING_MAX_TOKENS:
            embedding_text = truncate_text_to_tokens(embedding_text, DOC_EMBEDDING_MAX_TOKENS)
        
        # 处理URL长度
        processed_url = safe_truncate(url, DB_MAX_URL_LENGTH)
        if len(str(url)) > DB_MAX_URL_LENGTH:
            truncated_urls += 1
        
        urls.append(processed_url)
        titles.append(truncated_title)  # 使用已截断的title
        docs.append(doc)
        embedding_texts.append(embedding_text)

    # 关闭字段提取进度条
    field_progress.close()

    print(f"✅ 完整文档数据字段处理完成:")
    print(f"   总处理数: {len(urls):,}")
    print(f"   完整文档模式: 是")
    print(f"   截断URL: {truncated_urls:,} 个")
    print(f"   截断Title: {truncated_titles:,} 个") 

    # 添加embedding长度统计
    if embedding_lengths:
        avg_embedding_length = sum(embedding_lengths) / len(embedding_lengths)
        max_embedding_length = max(embedding_lengths)
        min_embedding_length = min(embedding_lengths)
        print(f"📊 标准模式Embedding统计:")
        print(f"   平均长度: {avg_embedding_length:.1f} 字符")

    if truncated_urls > 0:
        print(f"⚠️  共有 {truncated_urls:,} 个URL被截断以符合数据库限制")
    
    if truncated_titles > 0:
        print(f"📝 共有 {truncated_titles:,} 个Title被截断到1024字符以确保Milvus兼容性")
    
    print(f"✅ 准备处理: {len(urls):,} 个完整文档")
    
    result = {
        'urls': urls,
        'titles': titles,
        'docs': docs,
        'embedding_texts': embedding_texts
    }
    
    return result


def validate_batch_data(batch_urls, batch_titles):
    """
    批次级别最终验证 - 确保URL和Title符合长度限制
    
    Args:
        batch_urls (list[str]): URL列表
        batch_titles (list[str]): Title列表
        
    Returns:
        int: 验证问题数量
    """
    validation_issues = 0
    
    # 验证URL长度
    for idx, url in enumerate(batch_urls):
        if len(url) > DB_MAX_URL_LENGTH:
            batch_urls[idx] = url[:DB_MAX_URL_LENGTH]
            validation_issues += 1
    
    # 验证Title长度 - 强制截断到1024字符
    for idx, title in enumerate(batch_titles):
        if len(title) > 1024:
            batch_titles[idx] = title[:1024]
            validation_issues += 1
    
    return validation_issues


def prepare_batch_entities(i, batch_size, prepared_data, batch_embeddings):
    """
    准备批次实体数据（完整文档模式）
    
    Args:
        i (int): 批次起始索引
        batch_size (int): 批次大小
        prepared_data (dict): 准备好的数据字典
        batch_embeddings (dict): 批次embeddings
        
    Returns:
        tuple: (批次实体列表, URL列表)
    """
    batch_urls = prepared_data['urls'][i:i + batch_size]
    batch_titles = prepared_data['titles'][i:i + batch_size]
    batch_docs = prepared_data['docs'][i:i + batch_size]
    
    batched_entities = [
        batch_urls,
        batch_titles,
        batch_docs,
        batch_embeddings["sparse"],
        batch_embeddings["dense"],
    ]
    
    return batched_entities, batch_urls


def perform_search_test(col, ef, batch_num):
    """
    在第一个batch插入后进行搜索测试
    
    Args:
        col: Milvus集合对象
        ef: embedding函数
        batch_num (int): 批次编号
        
    Returns:
        None
    """
    if batch_num != 1:
        return
        
    print(f"\n🧪 第一批次插入完成，开始简化搜索测试...")
    try:
        col.flush()
        test_query = "test document content"
        print(f"🔍 测试查询: '{test_query}'")
        print(f"⚠️  搜索测试暂时跳过（搜索逻辑更新中）")
    except Exception as e:
        print(f"❌ 搜索测试失败: {e}")
    print(f"🔄 继续处理剩余批次...\n")


def insert_data_to_milvus(col, data: list, ef) -> dict:
    """
    将数据插入到Milvus中，完整文档模式
    
    Args:
        col: Milvus集合对象
        data (list): 文档数据列表
        ef: embedding函数
        
    Returns:
        dict: 插入统计信息
    """
    print("💾 开始插入数据到Milvus...")
    
    # 只有log打印，可忽视
    process_data = process_data_for_full_documents(data)
    
    # 显示批次配置
    batch_size = BATCH_SIZE
    print(f"🚀 使用统一批次大小: {batch_size}")
    
    if DEVICE and len(DEVICE) > 0:
        print(f"🎯 多GPU处理配置:")
        print(f"   批次大小: {batch_size}")
        print(f"   每GPU处理: ~{batch_size // len(DEVICE)} 个文本")
        print(f"   并行GPU数: {len(DEVICE)}")
    
    # 准备数据字段 - 根据配置选择处理方式
    if ENABLE_MULTIPROCESS_FIELD_EXTRACTION:
        print(f"🚀 启用多进程字段提取: {MULTIPROCESS_WORKERS} 个工作进程")
        prepared_data = prepare_data_fields_multiprocess(process_data)
    else:
        print("📋 使用单线程字段提取")
        prepared_data = prepare_data_fields(process_data)
    
    # 批量生成embeddings并插入
    print("🤖 生成embeddings并批量插入...")
    total_items = len(prepared_data['urls'])
    total_batches = (total_items + batch_size - 1) // batch_size
    print(f"💡 预估总批次数: {total_batches:,}")
    
    successful_batches = 0
    failed_batches = 0
    total_docs_processed = 0
    total_embedding_time = 0
    
    # 创建详细的进度条
    progress_bar = tqdm(
        range(0, total_items, batch_size),
        desc="🚀 向量生成与插入",
        unit="batch",
        total=total_batches,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
        dynamic_ncols=True
    )
    
    for i in progress_bar:
        batch_num = i // batch_size + 1
        current_batch_size = min(batch_size, total_items - i)
        
        try:
            # 更新进度条状态：正在生成embeddings
            progress_bar.set_postfix({
                '阶段': 'Embedding',
                '批次': f'{batch_num}/{total_batches}',
                '项目': f'{current_batch_size}个'
            })
            progress_bar.refresh()
            
            # 生成embeddings（记录时间）
            embedding_start = time.time()
            
            # 使用预处理的embedding文本
            batch_embedding_texts = prepared_data['embedding_texts'][i:i + batch_size]
            
            # 生成embeddings
            batch_embeddings = ef(batch_embedding_texts)
            embedding_time = time.time() - embedding_start
            
            # 数据类型处理（根据模式决定是否转换）
            if USE_FP16:
                conversion_start = time.time()
                
                if batch_embeddings["dense"] is not None:
                    dense_vectors = batch_embeddings["dense"]

                    # 标准模式：FP16转换为FP32
                    batch_embeddings["dense"] = [vec.astype(np.float32) for vec in dense_vectors]
                    conversion_time = time.time() - conversion_start
                    tqdm.write(f"🔄 批次 {batch_num}: FP16→FP32转换完成 ({conversion_time:.3f}s) - 转换了{len(dense_vectors)}个向量")
                else:
                    conversion_time = 0
                
                embedding_time += conversion_time  # 包含转换时间
            
            total_embedding_time += embedding_time
            
            # 更新进度条状态：正在插入数据
            progress_bar.set_postfix({
                '阶段': '插入数据',
                '批次': f'{batch_num}/{total_batches}',
                'embedding': f'{embedding_time:.2f}s'
            })
            progress_bar.refresh()
        
            # 准备批次数据
            batched_entities, batch_urls = prepare_batch_entities(
                i, batch_size, prepared_data, batch_embeddings
            )
            
            # 获取对应的title数据进行验证
            batch_titles = prepared_data['titles'][i:i + batch_size]
            
            # 批次级别最终验证 - 验证URL和Title长度
            validation_issues = validate_batch_data(batch_urls, batch_titles)
            if validation_issues > 0:
                tqdm.write(f"✅ 批次 {batch_num}: 修复了 {validation_issues} 个长度问题（URL/Title截断）")
                
                # 更新batched_entities中的title数据
                batched_entities[1] = batch_titles
            
            # 插入数据
            insert_start = time.time()
            col.insert(batched_entities)
            insert_time = time.time() - insert_start
            
            successful_batches += 1
            total_docs_processed += len(batch_urls)
            
            # 计算统计信息
            avg_time_per_batch = total_embedding_time / successful_batches if successful_batches > 0 else 0
            throughput = total_docs_processed / total_embedding_time if total_embedding_time > 0 else 0
            
            # 更新进度条状态：完成
            progress_bar.set_postfix({
                '阶段': '完成',
                '吞吐量': f'{throughput:.1f}/s',
                '成功率': f'{successful_batches}/{batch_num}',
                '已处理': f'{total_docs_processed:,}'
            })
            
            # 在第一个batch插入后进行搜索测试
            perform_search_test(col, ef, batch_num)
            
            # 每3个batch更新详细统计（减少频率以避免刷屏）
            if batch_num % 3 == 0:
                progress_percent = (batch_num / total_batches) * 100
                eta_seconds = (total_batches - batch_num) * avg_time_per_batch if avg_time_per_batch > 0 else 0
                eta_minutes = int(eta_seconds // 60)
                eta_seconds = int(eta_seconds % 60)
                
                tqdm.write(f"📊 批次 {batch_num}/{total_batches} ({progress_percent:.1f}%) | "
                          f"已处理: {total_docs_processed:,} | "
                          f"吞吐量: {throughput:.1f} docs/sec | "
                          f"预计剩余: {eta_minutes:02d}:{eta_seconds:02d}")
            
        except Exception as e:
            failed_batches += 1
            progress_bar.set_postfix({
                '阶段': '❌失败',
                '错误': str(e)[:20],
                '失败数': failed_batches
            })
            tqdm.write(f"⚠️  批次 {batch_num}/{total_batches} 处理失败: {e}")
            
            # 如果是第一批次就失败，停止处理
            if batch_num == 1:
                progress_bar.close()
                print(f"\n❌ 第一批次插入失败: {e}")
                print("🚨 第一批次失败，停止处理")
                raise e
    
    # 关闭进度条
    progress_bar.close()
    
    # 刷新数据
    print("\n💾 正在刷新数据到持久化存储...")
    col.flush()
    
    # 生成最终统计报告
    return generate_final_statistics(
        col, total_batches, successful_batches, failed_batches, 
        total_docs_processed, total_embedding_time
    )


def generate_final_statistics(col, total_batches: int, successful_batches: int, failed_batches: int, 
                            total_docs_processed: int, total_embedding_time: float) -> dict:
    """
    生成最终统计报告
    
    Args:
        col: Milvus集合对象
        total_batches (int): 总批次数
        successful_batches (int): 成功批次数
        failed_batches (int): 失败批次数
        total_docs_processed (int): 已处理文档数
        total_embedding_time (float): 总embedding时间
        
    Returns:
        dict: 统计信息字典
    """
    final_count = col.num_entities
    avg_throughput = total_docs_processed / total_embedding_time if total_embedding_time > 0 else 0
    
    print(f"\n🎉 数据插入完成！")
    print(f"📊 最终统计报告:")
    print(f"   总批次数: {total_batches:,}")
    print(f"   成功批次: {successful_batches:,}")
    print(f"   失败批次: {failed_batches:,}")
    print(f"   处理文档: {total_docs_processed:,}")
    print(f"   数据库记录: {final_count:,}")
    print(f"   总embedding时间: {total_embedding_time:.2f}秒")
    print(f"   平均吞吐量: {avg_throughput:.1f} docs/sec")
    
    # 多GPU性能统计
    if DEVICE and len(DEVICE) > 1:
        gpu_throughput = avg_throughput / len(DEVICE)
        print(f"   每GPU吞吐量: {gpu_throughput:.1f} docs/sec")
        print(f"   多GPU加速倍数: ~{len(DEVICE):.1f}x")
    
    # 计算成功率
    if total_batches > 0:
        success_rate = successful_batches / total_batches * 100
        print(f"   成功率: {success_rate:.2f}%")
        
        if success_rate < 95:
            print(f"⚠️  注意: 成功率较低，请检查数据质量或系统资源")
    
    return {
        'total_batches': total_batches,
        'successful_batches': successful_batches,
        'failed_batches': failed_batches,
        'total_docs_processed': total_docs_processed,
        'final_count': final_count,
        'total_embedding_time': total_embedding_time,
        'avg_throughput': avg_throughput,
        'success_rate': success_rate if total_batches > 0 else 0
    }