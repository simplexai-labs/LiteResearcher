#!/usr/bin/env python3
"""
配置备份工具
每次数据导入时自动保存配置文件副本，以便追踪和审计
"""

import os
import json
import shutil
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def extract_config_variables(config_file_path: str = "config.py") -> Dict[str, Any]:
    """
    从config.py文件中提取所有配置变量
    
    Args:
        config_file_path (str): 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置变量字典
    """
    try:
        # 动态导入config模块
        spec = importlib.util.spec_from_file_location("config", config_file_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # 提取所有大写的配置变量（Python约定）
        config_vars = {}
        for attr_name in dir(config_module):
            if not attr_name.startswith('_') and attr_name.isupper():
                attr_value = getattr(config_module, attr_name)
                # 确保值是可序列化的
                try:
                    json.dumps(attr_value)
                    config_vars[attr_name] = attr_value
                except (TypeError, ValueError):
                    # 如果不能序列化，转换为字符串
                    config_vars[attr_name] = str(attr_value)
        
        return config_vars
        
    except Exception as e:
        print(f"❌ 提取配置变量失败: {e}")
        return {}


def create_config_backup(collection_name: str, 
                        config_file_path: str = "config.py",
                        backup_base_dir: str = "config_backups") -> Optional[str]:
    """
    创建配置备份
    
    Args:
        collection_name (str): 集合名称，用作备份文件夹名
        config_file_path (str): 配置文件路径
        backup_base_dir (str): 备份根目录
        
    Returns:
        Optional[str]: 备份目录路径，失败时返回None
    """
    try:
        # 创建备份根目录
        backup_root = Path(backup_base_dir)
        backup_root.mkdir(exist_ok=True)
        
        # 创建以集合名称命名的子目录
        collection_backup_dir = backup_root / collection_name
        collection_backup_dir.mkdir(exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 复制原始config.py文件
        config_backup_path = collection_backup_dir / f"config_{timestamp}.py"
        if os.path.exists(config_file_path):
            shutil.copy2(config_file_path, config_backup_path)
            print(f"✅ 配置文件已备份: {config_backup_path}")
        
        # 2. 提取配置变量并保存为JSON
        config_vars = extract_config_variables(config_file_path)
        if config_vars:
            json_backup_path = collection_backup_dir / f"config_{timestamp}.json"
            
            # 添加元数据
            backup_metadata = {
                "backup_timestamp": datetime.now().isoformat(),
                "collection_name": collection_name,
                "config_file_source": os.path.abspath(config_file_path),
                "backup_creator": "milvus_rag_data_importer",
                "config_variables": config_vars
            }
            
            with open(json_backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 配置JSON已保存: {json_backup_path}")
        
        # 3. 创建最新配置的软链接（方便查看）
        latest_config_link = collection_backup_dir / "latest_config.py"
        latest_json_link = collection_backup_dir / "latest_config.json"
        
        # 删除旧的软链接
        if latest_config_link.exists():
            latest_config_link.unlink()
        if latest_json_link.exists():
            latest_json_link.unlink()
        
        # 创建新的软链接
        try:
            latest_config_link.symlink_to(config_backup_path.name)
            latest_json_link.symlink_to(f"config_{timestamp}.json")
            print(f"✅ 最新配置链接已创建: {latest_config_link}")
        except OSError:
            # 在某些系统上软链接可能失败，忽略这个错误
            pass
        
        # 4. 创建说明文件
        readme_path = collection_backup_dir / "README.md"
        if not readme_path.exists():
            readme_content = f"""# 配置备份 - {collection_name}

## 📋 说明

这个文件夹包含了集合 `{collection_name}` 数据导入时使用的配置备份。

## 📁 文件结构

- `config_YYYYMMDD_HHMMSS.py`: 原始配置文件备份
- `config_YYYYMMDD_HHMMSS.json`: 配置变量JSON格式备份（包含元数据）
- `latest_config.py`: 指向最新配置文件的软链接
- `latest_config.json`: 指向最新配置JSON的软链接

## 🔍 查看配置历史

```bash
# 查看所有备份
ls -la {collection_name}/

# 查看最新配置
cat {collection_name}/latest_config.json

# 对比不同时间的配置
diff {collection_name}/config_20241201_120000.py {collection_name}/config_20241201_140000.py
```

## ⚠️ 注意事项

- 这些是只读备份文件，请勿直接修改
- 如需恢复配置，请复制内容到主配置文件
- 配置文件按时间戳命名，便于追踪变更历史

自动生成于: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
        
        print(f"📂 配置备份目录: {collection_backup_dir}")
        return str(collection_backup_dir)
        
    except Exception as e:
        print(f"❌ 创建配置备份失败: {e}")
        return None


def list_config_backups(collection_name: str, backup_base_dir: str = "config_backups") -> list:
    """
    列出指定集合的所有配置备份
    
    Args:
        collection_name (str): 集合名称
        backup_base_dir (str): 备份根目录
        
    Returns:
        list: 备份文件列表
    """
    try:
        collection_backup_dir = Path(backup_base_dir) / collection_name
        if not collection_backup_dir.exists():
            return []
        
        backups = []
        for file in collection_backup_dir.glob("config_*.py"):
            stat = file.stat()
            backups.append({
                "file": str(file),
                "timestamp": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size": stat.st_size
            })
        
        # 按时间排序
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        return backups
        
    except Exception as e:
        print(f"❌ 列出备份失败: {e}")
        return []


def get_latest_config(collection_name: str, backup_base_dir: str = "config_backups") -> Optional[Dict[str, Any]]:
    """
    获取指定集合的最新配置
    
    Args:
        collection_name (str): 集合名称
        backup_base_dir (str): 备份根目录
        
    Returns:
        Optional[Dict[str, Any]]: 最新配置内容，失败时返回None
    """
    try:
        latest_json = Path(backup_base_dir) / collection_name / "latest_config.json"
        if latest_json.exists():
            with open(latest_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
        
    except Exception as e:
        print(f"❌ 获取最新配置失败: {e}")
        return None


def cleanup_old_backups(collection_name: str, 
                       keep_count: int = 10, 
                       backup_base_dir: str = "config_backups") -> int:
    """
    清理旧的配置备份，只保留最新的N个
    
    Args:
        collection_name (str): 集合名称
        keep_count (int): 保留的备份数量
        backup_base_dir (str): 备份根目录
        
    Returns:
        int: 删除的备份数量
    """
    try:
        collection_backup_dir = Path(backup_base_dir) / collection_name
        if not collection_backup_dir.exists():
            return 0
        
        # 获取所有配置文件，按修改时间排序
        config_files = list(collection_backup_dir.glob("config_*.py"))
        json_files = list(collection_backup_dir.glob("config_*.json"))
        
        config_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        deleted_count = 0
        
        # 删除多余的.py文件
        for file in config_files[keep_count:]:
            file.unlink()
            deleted_count += 1
            print(f"🗑️ 删除旧备份: {file}")
        
        # 删除多余的.json文件
        for file in json_files[keep_count:]:
            file.unlink()
            deleted_count += 1
            print(f"🗑️ 删除旧备份: {file}")
        
        return deleted_count
        
    except Exception as e:
        print(f"❌ 清理备份失败: {e}")
        return 0


def backup_config_for_import(collection_name: str) -> bool:
    """
    为数据导入创建配置备份的便捷函数
    
    Args:
        collection_name (str): 集合名称
        
    Returns:
        bool: 备份是否成功
    """
    print(f"\n📋 为集合 '{collection_name}' 创建配置备份...")
    
    backup_dir = create_config_backup(collection_name)
    if backup_dir:
        # 自动清理旧备份，保留最新10个
        deleted_count = cleanup_old_backups(collection_name, keep_count=10)
        if deleted_count > 0:
            print(f"🧹 已清理 {deleted_count} 个旧备份")
        
        print(f"✅ 配置备份完成: {backup_dir}")
        return True
    else:
        print("❌ 配置备份失败")
        return False


if __name__ == "__main__":
    # 测试功能
    test_collection = "test_collection"
    print("🧪 测试配置备份功能...")
    
    success = backup_config_for_import(test_collection)
    if success:
        # 列出备份
        backups = list_config_backups(test_collection)
        print(f"\n📁 当前备份数量: {len(backups)}")
        
        # 获取最新配置
        latest = get_latest_config(test_collection)
        if latest:
            print(f"📊 最新配置集合名称: {latest.get('collection_name')}")
            print(f"📅 备份时间: {latest.get('backup_timestamp')}")