# DISKANN RAG服务 - 快速开始

## 📂 目录结构

```
backend_services/rag_diskann/
├── local_rag_diskann_server.py  # DISKANN RAG服务 (端口8018)
├── embedding_server_diskann.py  # 独立Embedding服务 (端口8028, 可选)
├── diskann_config.py            # 本地配置文件 (独立配置)
├── start.sh                     # 启动脚本
├── stop.sh                      # 停止脚本
├── test.py                      # 测试脚本
├── README.md                    # 本文档
└── logs/                        # 日志目录
    └── diskann_server.log
```

## 🚀 快速启动（3步）

### 1. 进入目录
```bash
cd backend_services/rag_diskann
```

### 2. 启动服务
```bash
bash start.sh
```

### 3. 测试服务
```bash
python test.py
```

看到 `🎉 所有测试通过！` 就成功了！

---

## 🏗️ 架构说明

```
┌──────────────────────────────────────────┐
│  DISKANN RAG服务 (8018)                  │
│  • 内嵌BGE-M3模型                        │
│  • FP32向量 (DISKANN要求)                │
│  • Milvus DISKANN索引查询                │
│  • 支持千万级以上数据                    │
│  • 混合搜索 (Sparse + Dense)             │
└──────────────────────────────────────────┘
```

**技术特性**：
- ✅ FP32向量精度 (DISKANN要求)
- ✅ 内嵌Embedding模型 (无需独立服务)
- ✅ DISKANN磁盘索引
- ✅ 适用于千万级以上数据
- ✅ 高吞吐量，低延迟

**与FP16版本的区别**：
- 🎯 向量精度: FP32 vs FP16
- 📈 索引类型: DISKANN vs HNSW
- 💾 数据规模: 千万级+ vs 百万级
- 🔧 架构: 单体服务 vs 微服务

---

## 📝 详细命令

### 启动服务

#### 方式1: 自动启动（推荐）
```bash
bash start.sh
```

#### 方式2: 手动启动
```bash
python local_rag_diskann_server.py
```

### 停止服务
```bash
bash stop.sh
```

### 查看日志
```bash
# 实时查看DISKANN服务日志
tail -f logs/diskann_server.log

# 查看完整日志
cat logs/diskann_server.log
```

### 健康检查
```bash
# 检查DISKANN服务
curl http://localhost:8018/health
```

---

## 🧪 测试示例

### 1. 测试健康检查
```bash
curl http://localhost:8018/health | jq
```

### 2. 测试搜索（Hybrid模式）
```bash
curl -X POST http://47.111.147.142:8010/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "limit": 10,
    "search_type": "hybrid",
    "sparse_weight": 0.7,
    "dense_weight": 1.0
  }'
```

### 3. 测试Dense搜索
```bash
curl -X POST http://localhost:8018/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deep learning neural networks",
    "limit": 5,
    "search_type": "dense"
  }'
```

### 4. 运行完整测试
```bash
python test.py
```

---

## 📊 API文档

- **服务首页**: http://localhost:8018/
- **Swagger文档**: http://localhost:8018/docs
- **健康检查**: http://localhost:8018/health

---

## ⚙️ 配置说明

### 本地配置文件

DISKANN服务使用独立的本地配置文件 `diskann_config.py`，不依赖父级 `api_config.py`。

**主要配置项**:
```python
# 端口配置
DISKANN_RAG_PORT = 8018          # RAG服务端口
DISKANN_EMBEDDING_PORT = 8028    # Embedding服务端口

# 模型配置
API_BGE_MODEL_PATH = "/path/to/bge-m3"
API_BGE_MAX_LENGTH = 128
API_BATCH_SIZE = 32

# GPU配置
API_DEVICE = ["cuda:1", "cuda:2", "cuda:3", "cuda:4"]

# 搜索集合
SEARCH_COLLECTION_NAME = "serper_test"

# DISKANN搜索参数
DISKANN_SEARCH_LIST = 100  # 50-300，越大越精确但越慢
```

### 配置验证

运行配置验证:
```bash
python diskann_config.py
```

---

## ⚙️ 配置修改

所有配置集中在 `diskann_config.py` 文件中，修改后重启服务即可生效。

### 修改端口

编辑 `diskann_config.py`:
```python
DISKANN_RAG_PORT = 8018          # 改为你想要的端口
DISKANN_EMBEDDING_PORT = 8028    # Embedding服务端口
```

### 修改GPU设备

编辑 `diskann_config.py`:
```python
API_DEVICE = ["cuda:0"]  # 使用第1块GPU
# 或
API_DEVICE = ["cuda:1", "cuda:2"]  # 使用多块GPU
```

### 修改集合名称

编辑 `diskann_config.py`:
```python
SEARCH_COLLECTION_NAME = "your_collection_name"
```

### 修改DISKANN搜索参数

编辑 `diskann_config.py`:
```python
DISKANN_SEARCH_LIST = 200  # 增大可提高精度，但会降低速度
# 默认100，可尝试范围: 50-300
```

---

## 🔧 故障排查

### 问题1: 启动失败，端口被占用
```bash
# 查看占用端口的进程
lsof -i:8018

# 停止服务
bash stop.sh

# 或手动杀掉进程
kill <PID>
```

### 问题2: GPU内存不足
**解决**: 编辑 `../api_config.py`，使用不同的GPU
```python
API_DEVICE = ["cuda:1"]  # 换成空闲的GPU
```

### 问题3: 向量类型错误
**症状**: 日志中出现 "dtype 错误" 或 "转换失败"

**原因**: DISKANN要求FP32向量

**检查**:
```bash
# 查看服务日志
tail -f logs/diskann_server.log

# 确认使用FP32
curl http://localhost:8018/health | jq '.vector_precision'
```

### 问题4: 集合未找到
**症状**: "Collection not found" 错误

**解决**:
1. 确认Milvus服务运行正常
2. 检查集合名称配置 `api_config.py -> SEARCH_COLLECTION_NAME`
3. 确认数据已导入到Milvus

### 问题5: 查看详细错误
```bash
# 查看最新100行日志
tail -100 logs/diskann_server.log

# 实时查看日志
tail -f logs/diskann_server.log

# 搜索错误
grep -i error logs/diskann_server.log
```

---

## 🚦 服务监控

### 检查服务状态
```bash
# 方式1: 健康检查
curl http://localhost:8018/health | jq

# 方式2: 查看进程
lsof -i:8018

# 方式3: 运行测试
python test.py
```

### 查看GPU使用
```bash
nvidia-smi

# 实时监控
watch -n 1 nvidia-smi
```

### 性能指标
查看日志中的性能统计：
```bash
grep "时间统计" logs/diskann_server.log | tail -20
```

---

## 📈 性能优化建议

### 1. 调整worker数量
编辑 `local_rag_diskann_server.py` 底部:
```python
workers=6  # 改为 CPU核心数 / 4
```

### 2. 调整DISKANN搜索参数
增大 `search_list` 可提高精度但会降低速度：
```python
search_params = {
    "params": {
        "search_list": 200  # 默认100，可尝试50-300
    }
}
```

### 优化批处理大小

编辑 `diskann_config.py`:
```python
API_BATCH_SIZE = 64  # 根据GPU内存调整
MAX_BATCH_SIZE = 32  # Embedding服务批处理大小
```

---

## 🔄 与FP16版本对比

| 特性 | DISKANN (本服务) | FP16 (rag/) |
|-----|----------------|-------------|
| 向量精度 | FP32 | FP16 |
| 索引类型 | DISKANN | HNSW |
| 适用规模 | 千万级+ | 百万级 |
| 架构 | 单体 | 微服务 |
| GPU显存 | ~20GB | ~2GB (微服务) |
| 启动端口 | 8018 | 8017 + 8020 |
| 存储方式 | 磁盘+内存 | 内存 |
| 性能特点 | 高吞吐量 | 低延迟 |

---

## 💡 使用场景

### DISKANN版本（本服务）适合：
- ✅ 千万级以上数据
- ✅ 磁盘存储为主
- ✅ 高吞吐量需求
- ✅ NVMe SSD环境

### FP16版本适合：
- ✅ 百万级数据
- ✅ 内存存储
- ✅ 低延迟需求
- ✅ GPU显存受限

---

## ❓ 常见问题

### Q: 为什么DISKANN必须使用FP32？
A: DISKANN索引的设计和实现要求使用FP32精度向量，FP16会导致索引构建和搜索失败。

### Q: 性能会比FP16慢吗？
A: 单次查询可能略慢（FP32计算），但在千万级数据上吞吐量更高，整体性能更好。

### Q: 能否与FP16服务同时运行？
A: 可以！它们使用不同端口（8018 vs 8017），可以同时运行。

### Q: 需要多大的磁盘空间？
A: 取决于数据量，一般需要数据量2-3倍的空间用于索引。NVMe SSD性能最佳。

---

## 📞 获取帮助

遇到问题？
1. 查看日志: `tail -f logs/diskann_server.log`
2. 运行测试: `python test.py`
3. 检查端口: `lsof -i:8018`
4. 查看GPU: `nvidia-smi`
5. 检查Milvus: `docker ps | grep milvus`

---

**专为千万级数据优化！** 🎯
