"""
进度管理和配置验证工具
从data.py中移出的辅助功能
"""

import json
import os
import glob
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from .config_loader import (
    DATA_FOLDER_PATH, DATA_FILE_PATTERN, DATA_FILE_PATH, BGE_MODEL_PATH,
    DATA_COLLECTION_NAME, BATCH_SIZE, DEVICE, USE_FP16, ENABLE_SQL_STORAGE, SQL_SCHEMA,
    DOC_EMBEDDING_MAX_TOKENS, DOC_STORAGE_MAX_CHARS
)


def get_progress_file_path(collection_name):
    """获取进度文件路径"""
    return f"{collection_name}_progress.json"


def get_processed_files_txt_path(collection_name):
    """获取已处理文件txt记录路径"""
    return f"{collection_name}_processed_files.txt"


def get_sql_progress_file_path(sql_schema):
    """获取SQL专用进度文件路径"""
    return f"{sql_schema}_sql_progress.json"


def get_sql_processed_files_txt_path(sql_schema):
    """获取SQL专用已处理文件txt记录路径"""
    return f"{sql_schema}_sql_processed_files.txt"


def extract_file_number(file_path):
    """
    从文件名中提取四位数字编号
    例如: cc_en_head-0001.json.gz -> 1
         cc_en_middle-0002.json.gz -> 2
    """
    filename = os.path.basename(file_path)
    # 匹配.json.gz前的四位数字
    match = re.search(r'-(\d{4})\.json\.gz$', filename)
    if match:
        return int(match.group(1))
    return 9999  # 如果没有匹配到，放到最后


def sort_files_by_number(file_list):
    """
    按照文件名中的四位数字排序
    返回排序后的文件列表
    """
    return sorted(file_list, key=extract_file_number)


def save_processed_file_to_txt(collection_name, filename):
    """
    将已处理的文件名追加到txt文件中
    """
    txt_path = get_processed_files_txt_path(collection_name)
    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write(f"{filename}\n")


def load_processed_files_from_txt(collection_name):
    """
    从txt文件中加载已处理的文件名列表
    """
    txt_path = get_processed_files_txt_path(collection_name)
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except:
            pass
    return []


def load_sql_processed_files_from_txt(sql_schema):
    """
    从SQL专用txt文件中加载已处理的文件名列表
    """
    txt_path = get_sql_processed_files_txt_path(sql_schema)
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except:
            pass
    return []


def save_sql_processed_file_to_txt(sql_schema, filename):
    """
    将已处理的文件名追加到SQL专用txt文件中
    """
    txt_path = get_sql_processed_files_txt_path(sql_schema)
    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write(f"{filename}\n")


def load_progress(collection_name):
    """加载进度文件"""
    progress_file = get_progress_file_path(collection_name)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"processed_files": [], "total_files": 0, "created_at": datetime.now().isoformat()}


def save_progress(collection_name, processed_files, total_files=None):
    """保存进度"""
    progress_file = get_progress_file_path(collection_name)
    progress = {
        "collection_name": collection_name,
        "processed_files": processed_files,
        "total_files": total_files or len(processed_files),
        "last_updated": datetime.now().isoformat()
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    print(f"📝 进度已保存到 {progress_file}")


def load_sql_progress(sql_schema):
    """加载SQL专用进度文件"""
    progress_file = get_sql_progress_file_path(sql_schema)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"processed_files": [], "total_files": 0, "created_at": datetime.now().isoformat()}


def save_sql_progress(sql_schema, processed_files, total_files=None):
    """保存SQL专用进度"""
    progress_file = get_sql_progress_file_path(sql_schema)
    progress = {
        "sql_schema": sql_schema,
        "processed_files": processed_files,
        "total_files": total_files or len(processed_files),
        "last_updated": datetime.now().isoformat()
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    print(f"📝 SQL进度已保存到 {progress_file}")


def show_progress(collection_name):
    """显示当前进度"""
    progress = load_progress(collection_name)
    processed_count = len(progress["processed_files"])
    total_count = progress.get("total_files", processed_count)
    
    print(f"\n📊 集合 '{collection_name}' 处理进度:")
    print(f"   已处理: {processed_count} 个文件")
    if total_count > processed_count:
        print(f"   总计: {total_count} 个文件")
        print(f"   进度: {processed_count/total_count*100:.1f}%")
    print(f"   上次更新: {progress.get('last_updated', '未知')}")
    
    if processed_count > 0:
        print(f"   最近处理的文件:")
        for file_path in progress["processed_files"][-3:]:  # 显示最近3个
            print(f"     • {os.path.basename(file_path)}")
    
    return progress


def show_sql_progress(sql_schema):
    """显示SQL专用进度"""
    progress = load_sql_progress(sql_schema)
    processed_count = len(progress["processed_files"])
    total_count = progress.get("total_files", processed_count)
    
    print(f"\n📊 SQL Schema '{sql_schema}' 处理进度:")
    print(f"   已处理: {processed_count} 个文件")
    if total_count > processed_count:
        print(f"   总计: {total_count} 个文件")
        print(f"   进度: {processed_count/total_count*100:.1f}%")
    print(f"   上次更新: {progress.get('last_updated', '未知')}")
    
    if processed_count > 0:
        print(f"   最近处理的文件:")
        for file_path in progress["processed_files"][-3:]:  # 显示最近3个
            print(f"     • {os.path.basename(file_path)}")
    
    return progress


def get_existing_vectorized_files(existing_collections=None):
    """
    获取现有的已向量化文件列表（但不合并到SQL进度中）
    
    Args:
        existing_collections: 现有collection列表，默认为["rag_242", "rag_243"]
    
    Returns:
        已向量化的文件名列表
    """
    if existing_collections is None:
        existing_collections = ["rag_242", "rag_243"]
    
    print(f"\n🔍 读取现有已向量化文件列表...")
    
    vectorized_filenames = set()
    
    # 从现有的txt文件中读取已处理的文件名
    for collection in existing_collections:
        txt_path = get_processed_files_txt_path(collection)
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    filenames = [line.strip() for line in f if line.strip()]
                    vectorized_filenames.update(filenames)
                    print(f"✅ 从 {txt_path} 读取了 {len(filenames)} 个已向量化文件")
            except Exception as e:
                print(f"⚠️  读取 {txt_path} 失败: {e}")
        else:
            print(f"⚠️  文件不存在: {txt_path}")
    
    print(f"📊 总计发现 {len(vectorized_filenames)} 个已向量化文件")
    print(f"⚠️  注意：这些文件只是已向量化，还需要进行SQL插入")
    
    return list(vectorized_filenames)


def validate_config(use_batch_mode=False, data_folder_path=None):
    """
    验证配置参数
    
    Args:
        use_batch_mode: 是否使用批量文件模式
        data_folder_path: 批量模式下的文件夹路径
    """
    print("🔍 验证配置参数...")
    
    if use_batch_mode:
        # 批量模式验证
        folder_path = data_folder_path or DATA_FOLDER_PATH
        if not Path(folder_path).exists():
            print(f"❌ 数据文件夹不存在: {folder_path}")
            return False
        print(f"✅ 数据文件夹: {folder_path}")
        print(f"✅ 文件模式: {DATA_FILE_PATTERN}")
    else:
        # 单文件模式验证
        if not Path(DATA_FILE_PATH).exists():
            print(f"❌ 数据文件不存在: {DATA_FILE_PATH}")
            return False
        print(f"✅ 数据文件: {DATA_FILE_PATH}")
    
    # 检查模型路径
    if not Path(BGE_MODEL_PATH).exists():
        print(f"❌ BGE模型路径不存在: {BGE_MODEL_PATH}")
        return False
    
    print(f"✅ BGE模型: {BGE_MODEL_PATH}")
    print(f"✅ 集合名称: {DATA_COLLECTION_NAME}")
    print(f"✅ 批次大小: {BATCH_SIZE}")
    print(f"✅ GPU设备: {DEVICE}")
    print(f"✅ FP16模式: {USE_FP16}")
    print(f"✅ 完整文档模式: 已启用")
    print(f"✅ SQL存储: {ENABLE_SQL_STORAGE}")
    if ENABLE_SQL_STORAGE:
        print(f"✅ SQL Schema: {SQL_SCHEMA}")
    
    return True


def create_and_load_data_batch(data_folder=None, file_pattern=None, import_strategy=None, enable_sql_storage=None):
    """
    批量处理文件夹中的文件 - 改进版：按数字排序+实时文件名记录
    
    Args:
        data_folder: 数据文件夹路径
        file_pattern: 文件名模式
        import_strategy: 导入策略
        enable_sql_storage: 是否启用SQL存储，None时使用配置文件设置
    """
    print("\n🚀 开始批量文件处理（按数字排序）...")
    
    # 扫描文件
    folder_path = data_folder or DATA_FOLDER_PATH
    pattern = file_pattern or DATA_FILE_PATTERN
    pattern_path = os.path.join(folder_path, pattern)
    file_list = glob.glob(pattern_path)
    
    if not file_list:
        print(f"❌ 在 {folder_path} 中没有找到匹配 {pattern} 的文件")
        return False
    
    # 🔄 按文件名中的四位数字排序（重要改进）
    file_list = sort_files_by_number(file_list)
    print(f"📊 文件已按数字编号排序")
    
    # 显示前几个文件的排序结果
    print(f"📁 排序后的文件列表（前5个）:")
    for i, file_path in enumerate(file_list[:5]):
        filename = os.path.basename(file_path)
        file_num = extract_file_number(file_path)
        print(f"   {i+1}. {filename} (编号: {file_num:04d})")
    if len(file_list) > 5:
        print(f"   ... 还有 {len(file_list)-5} 个文件")
    
    # 🔄 从txt文件加载已处理的文件名（新增功能）
    processed_filenames = set(load_processed_files_from_txt(DATA_COLLECTION_NAME))
    
    # 也从JSON进度文件加载已处理的文件（保持兼容性）
    progress = load_progress(DATA_COLLECTION_NAME)
    processed_files_set = set(progress["processed_files"])
    
    # 过滤已处理的文件（同时检查文件名和绝对路径）
    remaining_files = []
    for f in file_list:
        filename = os.path.basename(f)
        abs_path = os.path.abspath(f)
        if filename not in processed_filenames and abs_path not in processed_files_set:
            remaining_files.append(f)
    
    print(f"📁 找到 {len(file_list)} 个文件，已处理 {len(file_list) - len(remaining_files)} 个，剩余 {len(remaining_files)} 个")
    
    if not remaining_files:
        print("✅ 所有文件都已处理完成！")
        return True
    
    # 🔄 保持数字排序，不再按文件大小排序
    print(f"📊 将按数字顺序处理剩余 {len(remaining_files)} 个文件")
    
    successful_files = 0
    total_files = len(remaining_files)
    
    # 导入核心处理函数
    from .data_processor import create_and_load_single_data
    
    # 🔄 使用进度条循环处理每个文件（显示当前文件信息）
    for idx, file_path in enumerate(tqdm(remaining_files, desc="🔄 按序处理文件", unit="file")):
        filename = os.path.basename(file_path)
        file_num = extract_file_number(file_path)
        
        # 🔄 实时显示当前处理的文件（重要改进）
        print(f"\n📂 正在处理第 {idx+1}/{total_files} 个文件")
        print(f"📄 文件名: {filename}")
        print(f"🔢 文件编号: {file_num:04d}")
        print(f"📍 进度: {(idx+1)/total_files*100:.1f}%")
        
        try:
            # 调用单文件处理逻辑
            success = create_and_load_single_data(file_path, import_strategy, enable_sql_storage)
            if success:
                successful_files += 1
                print(f"✅ 文件 {filename} (编号:{file_num:04d}) 处理成功")
                
                # 🔄 同时更新两个记录（重要改进）
                # 1. 将文件名写入txt文件
                save_processed_file_to_txt(DATA_COLLECTION_NAME, filename)
                
                # 2. 更新JSON进度文件（保持兼容性）
                progress["processed_files"].append(os.path.abspath(file_path))
                save_progress(DATA_COLLECTION_NAME, progress["processed_files"], len(file_list))
                
                print(f"📝 文件名已记录到 {get_processed_files_txt_path(DATA_COLLECTION_NAME)}")
            else:
                print(f"❌ 文件 {filename} (编号:{file_num:04d}) 处理失败")
        
        except Exception as e:
            print(f"❌ 文件 {filename} (编号:{file_num:04d}) 处理出错: {e}")
    
    # 输出最终统计
    print(f"\n✅ 批量处理完成！")
    print(f"   总文件数: {total_files}")
    print(f"   成功文件: {successful_files}")
    print(f"   失败文件: {total_files - successful_files}")
    print(f"   成功率: {successful_files / total_files * 100:.1f}%")
    print(f"📝 已处理文件记录: {get_processed_files_txt_path(DATA_COLLECTION_NAME)}")
    
    return successful_files > 0


def create_and_load_data_batch_sql_only(data_folder=None, file_pattern=None, enable_sql_storage=None):
    """
    纯SQL版本的批量处理文件夹中的文件 - 独立于Milvus RAG的处理逻辑
    
    处理逻辑：
    1. 优先处理已向量化但未SQL插入的文件（来自rag_242和rag_243）
    2. 然后处理文件夹中的所有其他文件（不受DATA_FILE_PATTERN限制）
    
    Args:
        data_folder: 数据文件夹路径
        file_pattern: 文件名模式（对SQL处理无效，保留参数兼容性）
        enable_sql_storage: 是否启用SQL存储，None时使用配置文件设置
    """
    from .config_loader import SQL_SCHEMA
    
    print("\n🚀 开始SQL专用批量文件处理（独立处理逻辑）...")
    print(f"💾 SQL Schema: {SQL_SCHEMA}")
    print(f"📝 注意：SQL处理不受Milvus RAG的DATA_FILE_PATTERN限制")
    
    # 获取现有已向量化文件列表
    vectorized_filenames = get_existing_vectorized_files()
    
    # 扫描整个文件夹中的所有.json.gz文件（不受模式限制）
    folder_path = data_folder or DATA_FOLDER_PATH
    # SQL处理：扫描所有.json.gz文件
    pattern_path = os.path.join(folder_path, "*.json.gz")
    file_list = glob.glob(pattern_path)
    
    if not file_list:
        print(f"❌ 在 {folder_path} 中没有找到.json.gz文件")
        return False
    
    # 🔄 按文件名中的四位数字排序（重要改进）
    file_list = sort_files_by_number(file_list)
    print(f"📊 SQL专用处理：扫描整个文件夹，找到 {len(file_list)} 个.json.gz文件")
    print(f"📊 文件已按数字编号排序")
    
    # 显示前几个文件的排序结果
    print(f"📁 文件夹中的文件（前5个）:")
    for i, file_path in enumerate(file_list[:5]):
        filename = os.path.basename(file_path)
        file_num = extract_file_number(file_path)
        print(f"   {i+1}. {filename} (编号: {file_num:04d})")
    if len(file_list) > 5:
        print(f"   ... 还有 {len(file_list)-5} 个文件")
    
    # 🔄 从SQL专用txt文件加载已SQL插入的文件名
    sql_processed_filenames = set(load_sql_processed_files_from_txt(SQL_SCHEMA))
    
    # 也从SQL专用JSON进度文件加载已处理的文件
    sql_progress = load_sql_progress(SQL_SCHEMA)
    sql_processed_files_set = set(sql_progress["processed_files"])
    
    # 🎯 智能分类文件：
    # 1. 已向量化但未SQL插入的文件（优先处理）
    # 2. 全新的文件（其次处理）
    # 3. 已SQL插入的文件（跳过）
    
    vectorized_set = set(vectorized_filenames)
    priority_files = []  # 已向量化但未SQL插入
    new_files = []       # 全新文件
    
    for f in file_list:
        filename = os.path.basename(f)
        abs_path = os.path.abspath(f)
        
        # 跳过已经SQL插入的文件
        if filename in sql_processed_filenames or abs_path in sql_processed_files_set:
            continue
        
        # 分类未SQL插入的文件
        if filename in vectorized_set:
            priority_files.append(f)  # 已向量化，优先处理
        else:
            new_files.append(f)       # 全新文件，其次处理
    
    # 🚀 只处理已向量化的文件，忽略全新文件
    remaining_files = priority_files
    
    print(f"📁 SQL专用文件分类结果:")
    print(f"   文件夹总数: {len(file_list)} 个.json.gz文件")
    print(f"   已向量化待SQL插入: {len(priority_files)} 个 (本次处理)")
    print(f"   全新文件: {len(new_files)} 个 (暂不处理，等待向量化)") 
    print(f"   已SQL插入(跳过): {len(file_list) - len(priority_files) - len(new_files)} 个")
    print(f"   本次需处理: {len(remaining_files)} 个")
    
    if len(priority_files) > 0:
        print(f"\n🔥 第1阶段：优先处理已向量化文件")
        for i, f in enumerate(priority_files[:5]):
            filename = os.path.basename(f)
            print(f"     {i+1}. {filename} (已向量化→SQL插入)")
        if len(priority_files) > 5:
            print(f"     ... 还有 {len(priority_files)-5} 个已向量化文件")
    
    if not remaining_files:
        print("✅ 所有文件都已处理完成！")
        return True
    
    # 🔄 只处理已向量化的文件，忽略全新文件
    print(f"📊 SQL专用处理模式:")
    print(f"   处理策略: 只处理已向量化文件")
    print(f"   第1阶段: 处理 {len(priority_files)} 个已向量化文件")
    print(f"   监控模式: 完成后监控rag txt文件更新")
    print(f"💾 纯SQL模式：只进行PostgreSQL插入，跳过Milvus操作")
    
    successful_files = 0
    total_files = len(remaining_files)
    
    # 导入纯SQL处理函数
    from .data_sql_processor import create_and_load_single_data_sql_only
    
    # 🔄 使用进度条循环处理每个文件（显示当前文件信息）
    for idx, file_path in enumerate(tqdm(remaining_files, desc="🔄 按序处理文件(SQL专用)", unit="file")):
        filename = os.path.basename(file_path)
        file_num = extract_file_number(file_path)
        
        # 判断文件状态
        is_vectorized = filename in vectorized_set
        file_status = "已向量化→SQL插入" if is_vectorized else "全新文件→SQL插入"
        phase = "第1阶段" if is_vectorized else "第2阶段"
        
        # 🔄 实时显示当前处理的文件（重要改进）
        print(f"\n📂 正在处理第 {idx+1}/{total_files} 个文件（{phase}）")
        print(f"📄 文件名: {filename}")
        print(f"🔢 文件编号: {file_num:04d}")
        print(f"📍 进度: {(idx+1)/total_files*100:.1f}%")
        print(f"🏷️  文件状态: {file_status}")
        print(f"💾 SQL Schema: {SQL_SCHEMA}")
        
        try:
            # 调用纯SQL处理逻辑
            success = create_and_load_single_data_sql_only(file_path, enable_sql_storage)
            if success:
                successful_files += 1
                print(f"✅ 文件 {filename} (编号:{file_num:04d}) SQL处理成功")
                
                # 🔄 更新SQL专用进度记录
                # 1. 将文件名写入SQL专用txt文件
                save_sql_processed_file_to_txt(SQL_SCHEMA, filename)
                
                # 2. 更新SQL专用JSON进度文件
                sql_progress["processed_files"].append(os.path.abspath(file_path))
                save_sql_progress(SQL_SCHEMA, sql_progress["processed_files"], len(file_list))
                
                print(f"📝 文件名已记录到 {get_sql_processed_files_txt_path(SQL_SCHEMA)}")
            else:
                print(f"❌ 文件 {filename} (编号:{file_num:04d}) SQL处理失败")
        
        except Exception as e:
            print(f"❌ 文件 {filename} (编号:{file_num:04d}) SQL处理出错: {e}")
    
    # 输出最终统计
    print(f"\n✅ SQL专用批量处理完成！")
    print(f"   总文件数: {total_files}")
    print(f"   成功文件: {successful_files}")
    print(f"   失败文件: {total_files - successful_files}")
    print(f"   成功率: {successful_files / total_files * 100:.1f}%")
    print(f"   处理模式: SQL专用（无向量化）")
    print(f"   SQL Schema: {SQL_SCHEMA}")
    print(f"📝 已处理文件记录: {get_sql_processed_files_txt_path(SQL_SCHEMA)}")
    
    # 启动监控模式
    print(f"\n🔄 进入监控模式...")
    start_rag_txt_monitoring(SQL_SCHEMA, enable_sql_storage)
    
    return successful_files > 0


def start_rag_txt_monitoring(sql_schema, enable_sql_storage=None):
    """
    启动RAG txt文件监控模式
    持续监控rag_242和rag_243的processed_files.txt更新
    一旦发现新文件，立即进行SQL插入
    
    Args:
        sql_schema: SQL schema名称
        enable_sql_storage: 是否启用SQL存储
    """
    import time
    from .config_loader import DATA_FOLDER_PATH
    
    print(f"🔄 启动RAG文件监控模式...")
    print(f"📁 监控目标:")
    print(f"   - rag_242_processed_files.txt")
    print(f"   - rag_243_processed_files.txt")
    print(f"💾 SQL Schema: {sql_schema}")
    print(f"⏱️  检查间隔: 30秒")
    print(f"🛑 按 Ctrl+C 停止监控")
    
    # 记录当前已知的文件列表
    last_known_files = set(get_existing_vectorized_files())
    processed_count = 0
    
    try:
        while True:
            print(f"\n🔍 检查RAG txt文件更新... (已处理 {processed_count} 个新文件)")
            
            # 获取当前的已向量化文件列表
            current_files = set(get_existing_vectorized_files())
            
            # 找出新增的文件
            new_files = current_files - last_known_files
            
            if new_files:
                print(f"🎉 发现 {len(new_files)} 个新的已向量化文件！")
                
                # 获取已SQL插入的文件列表
                sql_processed_filenames = set(load_sql_processed_files_from_txt(sql_schema))
                
                # 过滤掉已SQL插入的文件
                files_to_process = []
                for filename in new_files:
                    if filename not in sql_processed_filenames:
                        # 构建完整文件路径
                        file_path = os.path.join(DATA_FOLDER_PATH, filename)
                        if os.path.exists(file_path):
                            files_to_process.append(file_path)
                
                if files_to_process:
                    print(f"📝 开始处理 {len(files_to_process)} 个新文件...")
                    
                    # 导入纯SQL处理函数
                    from .data_sql_processor import create_and_load_single_data_sql_only
                    
                    for file_path in files_to_process:
                        filename = os.path.basename(file_path)
                        print(f"\n🚀 监控模式：处理新文件 {filename}")
                        
                        try:
                            success = create_and_load_single_data_sql_only(file_path, enable_sql_storage)
                            if success:
                                processed_count += 1
                                print(f"✅ 文件 {filename} SQL插入成功")
                                
                                # 更新SQL专用进度记录
                                save_sql_processed_file_to_txt(sql_schema, filename)
                                
                                # 更新SQL专用JSON进度文件
                                sql_progress = load_sql_progress(sql_schema)
                                sql_progress["processed_files"].append(os.path.abspath(file_path))
                                save_sql_progress(sql_schema, sql_progress["processed_files"])
                                
                                print(f"📝 文件名已记录到 {get_sql_processed_files_txt_path(sql_schema)}")
                            else:
                                print(f"❌ 文件 {filename} SQL插入失败")
                        except Exception as e:
                            print(f"❌ 处理文件 {filename} 时出错: {e}")
                
                # 更新已知文件列表
                last_known_files = current_files
                
                if files_to_process:
                    print(f"\n✅ 监控处理完成，继续监控...")
            else:
                print(f"📊 没有发现新的已向量化文件")
            
            # 等待下次检查
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n🛑 监控模式已停止")
        print(f"📊 监控期间处理了 {processed_count} 个新文件")
        print(f"👋 退出SQL监控服务")
    except Exception as e:
        print(f"\n❌ 监控模式出错: {e}")
        print(f"📊 监控期间处理了 {processed_count} 个新文件")
