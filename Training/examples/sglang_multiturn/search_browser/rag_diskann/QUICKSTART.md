# 🚀 DISKANN RAG服务 - 3步启动

## 第1步: 进入目录
```bash
cd backend_services/rag_diskann
```

## 第2步: 启动服务
```bash
bash start.sh
```

等待看到：
```
✅ DISKANN RAG服务启动成功！

📊 服务信息:
   DISKANN RAG服务:  http://localhost:8018
   API文档:          http://localhost:8018/docs
```

## 第3步: 测试
```bash
python test.py
```

看到 `🎉 所有测试通过！` 就成功了！

---

## 📖 详细文档
查看 [README.md](README.md) 获取完整文档。

## 🛑 停止服务
```bash
bash stop.sh
```

## 📋 查看日志
```bash
tail -f logs/diskann_server.log
```

## 💡 重要提示
- DISKANN服务内嵌BGE-M3模型（无需独立embedding服务）
- 使用FP32向量精度（DISKANN要求）
- 适用于千万级以上数据
- 需要DISKANN索引的Milvus集合
- 端口8018（不与FP16服务冲突）

## 🎯 快速测试
```bash
# 测试健康
curl http://localhost:8018/health

# 测试搜索
curl -X POST http://localhost:8018/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "limit": 5}'
```
