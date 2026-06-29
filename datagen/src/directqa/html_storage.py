import os
import mmap
import struct
import json
from datetime import datetime
from threading import Lock
import hashlib
import time

class HTMLStorage:
    """
    纯mmap方案：索引和数据都用mmap（分布式版本）

    该实现将任意唯一键（现在用于完整的URL）映射到保存的HTML位置。

    索引文件格式（固定长度记录，方便随机访问）:
    [64字节key_hash][4字节storage_uid][4字节file_index][8字节offset][4字节length][8字节timestamp]

    数据文件格式:
    [4字节长度][8字节时间戳][2字节key长度][key][HTML]
    """

    INDEX_STRUCT_FMT = '<64sIIQIQ'  # 增加了 storage_uid (I)
    INDEX_RECORD_SIZE = struct.calcsize(INDEX_STRUCT_FMT)  # 动态计算，避免对齐差异
    
    def __init__(self, data_dirs=None):
        """
        初始化存储
        
        Args:
            data_dirs: None（新建） 或 list of dirs（多目录读写）
        """
        self.write_lock = Lock()
        
        # 多目录支持
        if data_dirs is None:
            # 新建目录，生成唯一 UID
            self.storage_uid = int(time.time() * 1000) % (2**32)  # 32位UID
            self.data_dir = f"./storage_{self.storage_uid}"
            os.makedirs(self.data_dir, exist_ok=True)
            self.data_dirs = [self.data_dir]
            print(f"🆕 创建新存储目录: {self.data_dir} (UID: {self.storage_uid})")
        else:
            # 使用已有目录
            self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
            # 从第一个目录提取 UID（写入用）
            self.data_dir = self.data_dirs[0]
            os.makedirs(self.data_dir, exist_ok=True)
            self.storage_uid = self._extract_uid_from_path(self.data_dir)
            print(f"📂 使用已有目录: {self.data_dirs} (主UID: {self.storage_uid})")
        
        # 目录到UID的映射
        self.dir_to_uid = {}
        for d in self.data_dirs:
            os.makedirs(d, exist_ok=True)
            uid = self._extract_uid_from_path(d)
            self.dir_to_uid[uid] = d
        
        # 索引文件路径（主目录）
        self.index_file_path = os.path.join(self.data_dir, "index.bin")
        
        # 哈希表（内存中，启动时快速加载）
        self.hash_table = {}  # {key_hash: (storage_uid, index_offset)}
        
        # 数据文件设置
        self.current_file_index = 0
        self.max_file_size = 2 * 1024 * 1024 * 1024  # 2GB
        self.current_offset = 0
        
        # mmap缓存（按 storage_uid 和 file_index）
        self.mmap_cache = {}  # {(storage_uid, file_index): {'file': f, 'mmap': mm}}
        self.index_mmap = None
        
        # 初始化
        self._init_index_file()
        self._load_hash_table()
        self._load_current_position()
    
    def _extract_uid_from_path(self, path):
        """从路径提取 storage_uid"""
        basename = os.path.basename(path.rstrip('/\\'))
        if '_' in basename:
            try:
                return int(basename.split('_')[-1])
            except ValueError:
                pass
        # 若无法提取，使用目录的哈希值
        return abs(hash(path)) % (2**32)
    
    def _init_index_file(self):
        """初始化索引文件"""
        if not os.path.exists(self.index_file_path):

            print("🆕 创建新的索引文件")
            # 创建空索引文件
            with open(self.index_file_path, 'wb') as f:
                pass
    
    def _hash_key(self, key: str) -> bytes:
        """将唯一键(例如完整URL)转为64字节固定长度哈希"""
        # 使用SHA256生成32字节，然后重复到64字节
        hash_bytes = hashlib.sha256(key.encode('utf-8')).digest()
        return hash_bytes + hash_bytes  # 64字节

    # 为向后兼容保留旧名称（其他模块可能仍然调用）
    def _hash_name(self, name: str) -> bytes:  # pragma: no cover - compatibility shim
        return self._hash_key(name)
    
    def _load_hash_table(self):
        """启动时加载所有目录的哈希表到内存（快速查找）"""
        total_records = 0
        
        for data_dir in self.data_dirs:
            index_file = os.path.join(data_dir, "index.bin")
            if not os.path.exists(index_file) or os.path.getsize(index_file) == 0:
                continue
            
            storage_uid = self._extract_uid_from_path(data_dir)
            
            with open(index_file, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
                offset = 0
                file_size = mm.size()
                
                while offset < file_size:
                    # 读取记录
                    if offset + self.INDEX_RECORD_SIZE > file_size:
                        break
                    
                    key_hash = mm[offset:offset+64]
                    
                    # 检查是否为空记录
                    if key_hash == b'\x00' * 64:
                        break
                    
                    # 存储：(storage_uid, index_offset)
                    self.hash_table[key_hash] = (storage_uid, offset)
                    offset += self.INDEX_RECORD_SIZE
                    total_records += 1
                
                mm.close()
        
        print(f"📚 从 {len(self.data_dirs)} 个目录加载了 {total_records} 条索引记录")
    
    def _load_current_position(self):
        """加载当前写入位置（支持从损坏的meta.json恢复）"""
        meta_file = os.path.join(self.data_dir, "meta.json")
        position_loaded = False
        
        # 尝试从 meta.json 加载
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    if meta and 'file_index' in meta and 'offset' in meta:
                        self.current_file_index = meta['file_index']
                        self.current_offset = meta['offset']
                        position_loaded = True
                        print(f"✓ 从 meta.json 加载位置: file_index={self.current_file_index}, offset={self.current_offset}")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"⚠️ meta.json 读取失败或为空: {str(e)}")
        
        # 如果 meta.json 不存在或损坏，扫描数据文件
        if not position_loaded:
            print(f"🔍 扫描数据文件以恢复写入位置...")
            self._scan_and_recover_position()
    
    def _scan_and_recover_position(self):
        """扫描数据文件，找到最大的 file_index 和对应的末尾 offset"""
        import glob
        
        # 查找所有 data_*.bin 文件
        pattern = os.path.join(self.data_dir, "data_*.bin")
        data_files = glob.glob(pattern)
        
        if not data_files:
            # 没有数据文件，从头开始
            self.current_file_index = 0
            self.current_offset = 0
            print(f"✓ 未找到数据文件，从头开始: file_index=0, offset=0")
            return
        
        # 提取文件名中的 index
        file_indices = []
        for file_path in data_files:
            basename = os.path.basename(file_path)
            # data_000123.bin -> 123
            try:
                index_str = basename.replace('data_', '').replace('.bin', '')
                file_index = int(index_str)
                file_indices.append((file_index, file_path))
            except ValueError:
                continue
        
        if not file_indices:
            self.current_file_index = 0
            self.current_offset = 0
            print(f"✓ 未找到有效数据文件，从头开始: file_index=0, offset=0")
            return
        
        # 找到最大的 file_index
        max_file_index, max_file_path = max(file_indices, key=lambda x: x[0])
        
        # 获取该文件的大小作为 offset
        file_size = os.path.getsize(max_file_path)
        
        self.current_file_index = max_file_index
        self.current_offset = file_size
        
        print(f"✓ 从数据文件恢复位置: file_index={self.current_file_index}, offset={self.current_offset} ({file_size/1024/1024:.2f} MB)")
        
        # 保存恢复的位置到 meta.json
        self._save_meta()
    
    def _save_meta(self):
        """保存元数据"""
        meta_file = os.path.join(self.data_dir, "meta.json")
        with open(meta_file, 'w') as f:
            json.dump({
                'file_index': self.current_file_index,
                'offset': self.current_offset
            }, f)
    
    def _get_data_file_path(self, file_index):
        """获取数据文件路径"""
        return os.path.join(self.data_dir, f"data_{file_index:06d}.bin")
    
    def _append_to_index(self, key, file_index, offset, length, timestamp):
        """追加索引记录"""
        key_hash = self._hash_key(key)
        
        # 构造索引记录（包含 storage_uid）
        record = struct.pack(
            self.INDEX_STRUCT_FMT,
            key_hash,
            self.storage_uid,  # 加入当前存储UID
            file_index,
            offset,
            length,
            timestamp
        )
        
        # 追加到索引文件
        with open(self.index_file_path, 'ab') as f:
            index_offset = f.tell()
            f.write(record)
        
        # 更新内存哈希表（存储 storage_uid, index_offset）
        self.hash_table[key_hash] = (self.storage_uid, index_offset)
    
    def save(self, key, html_content):
        """保存HTML，key应为完整URL"""
        timestamp = int(datetime.now().timestamp())
        key_bytes = key.encode('utf-8')
        html_bytes = html_content.encode('utf-8')
        
        key_len = len(key_bytes)
        html_len = len(html_bytes)
        total_len = 4 + 8 + 2 + key_len + html_len
        
        with self.write_lock:
            # 检查是否需要新文件
            if self.current_offset + total_len > self.max_file_size:
                self.current_file_index += 1
                self.current_offset = 0
            
            file_path = self._get_data_file_path(self.current_file_index)
            file_index = self.current_file_index
            offset = self.current_offset
            
            # 写入数据（显式小端，避免平台差异）
            with open(file_path, 'ab') as f:
                f.write(struct.pack('<I', total_len))
                f.write(struct.pack('<Q', timestamp))
                f.write(struct.pack('<H', key_len))
                f.write(key_bytes)
                f.write(html_bytes)
            
            # 追加索引
            self._append_to_index(key, file_index, offset, total_len, timestamp)
            
            # 更新当前位置
            self.current_offset += total_len
            self._save_meta()

            return (file_index, offset, total_len)
    
    def _get_data_mmap(self, storage_uid, file_index):
        """获取数据文件mmap（带缓存，支持多目录）"""
        cache_key = (storage_uid, file_index)
        
        if cache_key not in self.mmap_cache:
            # 根据 storage_uid 找到对应目录
            data_dir = self.dir_to_uid.get(storage_uid, self.data_dir)
            file_path = os.path.join(data_dir, f"data_{file_index:06d}.bin")
            
            if os.path.exists(file_path):
                f = open(file_path, 'rb')
                self.mmap_cache[cache_key] = {
                    'file': f,
                    'mmap': mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                }
        
        return self.mmap_cache[cache_key]['mmap']
    
    def get(self, key):
        """读取HTML（纯mmap，超快，支持多目录）"""
        key_hash = self._hash_key(key)
        
        # 从内存哈希表查找索引位置
        if key_hash not in self.hash_table:
            # 兼容旧数据：旧版本使用URL最后一段作为键
            legacy_key = key.rstrip('/').split('/')[-1]
            legacy_hash = self._hash_key(legacy_key) if legacy_key else key_hash
            if legacy_hash not in self.hash_table:
                return None
            storage_uid, index_offset = self.hash_table[legacy_hash]
        else:
            storage_uid, index_offset = self.hash_table[key_hash]
        
        # 根据 storage_uid 获取目录路径
        dir_path = self.dir_to_uid.get(storage_uid, self.data_dir)
        
        # 读取索引（使用mmap，从正确的目录）
        index_file = os.path.join(dir_path, "index.bin")
        with open(index_file, 'rb') as f:
            index_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # 解析索引记录（包含 storage_uid）
            record = index_mm[index_offset:index_offset+self.INDEX_RECORD_SIZE]
            _, stored_uid, file_index, offset, length, timestamp = struct.unpack(self.INDEX_STRUCT_FMT, record)
            
            index_mm.close()
        
        # 读取数据（使用mmap，从正确的目录）
        data_mm = self._get_data_mmap(storage_uid, file_index)
        
        # 读取总长度
        total_len = struct.unpack('<I', data_mm[offset:offset+4])[0]
        
        # 读取时间戳
        ts = struct.unpack('<Q', data_mm[offset+4:offset+12])[0]
        
        # 读取名字长度
        key_len = struct.unpack('<H', data_mm[offset+12:offset+14])[0]
        
        # 读取键（完整URL）
        stored_key = data_mm[offset+14:offset+14+key_len].decode('utf-8')
        
        # 读取HTML
        html_start = offset + 14 + key_len
        html_end = offset + total_len
        html_content = data_mm[html_start:html_end].decode('utf-8')
        
        return {
            'key': stored_key,
            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
            'html': html_content,
            'storage_uid': storage_uid
        }
    
    def batch_save(self, items):
        """批量保存"""
        results = []
        for key, html_content in items:
            result = self.save(key, html_content)
            results.append(result)
        return results
    
    def close(self):
        """关闭所有资源"""
        for cache in self.mmap_cache.values():
            cache['mmap'].close()
            cache['file'].close()
        if self.index_mmap:
            self.index_mmap.close()