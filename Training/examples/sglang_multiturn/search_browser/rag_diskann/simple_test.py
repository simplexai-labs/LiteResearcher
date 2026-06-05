#!/usr/bin/env python3
"""
简单测试脚本 - DISKANN RAG搜索
直接修改query变量即可测试
"""

import requests
import json

# ======================== 配置区域 ========================
QUERY = "Linear progression"  # 👈 在这里修改你的查询
LIMIT = 10  # Top 10结果
SEARCH_TYPE = "hybrid"  # hybrid/dense/sparse
SERVER_URL = "http://47.111.147.142:8010"
# =========================================================  http://47.111.147.142:8010/search

def search(query, limit=10):
    """执行搜索"""
    payload = {
        "query": query,
        "limit": limit,
        "search_type": SEARCH_TYPE,
        "sparse_weight": 0.7,
        "dense_weight": 1.0
    }

    response = requests.post(f"{SERVER_URL}/search", json=payload, timeout=30)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ 搜索失败: HTTP {response.status_code}")
        print(response.text)
        return None

def print_results(data):
    """打印搜索结果"""
    if not data:
        return

    # 标题栏
    print("\n" + "="*100)
    print(f"🔍 查询: {QUERY}")
    print("="*100)
    print(f"📊 返回结果: {data['total']}个")
    print(f"⏱️  搜索时间: {data['search_time']:.3f}秒 (Embedding: {data['embedding_time']:.3f}秒, Milvus: {data['milvus_time']:.3f}秒)")
    print(f"🎯 搜索模式: {data['search_type']} | 向量精度: {data['vector_dtype']} | 索引: {data['index_type']}")
    print("="*100)

    # 结果列表
    for i, result in enumerate(data['results'], 1):
        print(f"\n【{i}】评分: {result['score']:.4f}")
        print(f"📌 标题: {result['title']}")
        print(f"🔗 链接: {result['link']}")
        print(f"📝 摘要: {result['snippet']}")
        print("-"*100)

    print()

if __name__ == "__main__":
    print("🚀 DISKANN RAG搜索测试")
    print(f"📡 服务地址: {SERVER_URL}")

    # 执行搜索
    results = search(QUERY, LIMIT)

    # 打印结果
    if results:
        print_results(results)
        print("✅ 搜索完成！")
    else:
        print("❌ 搜索失败，请检查服务是否启动")
        print("💡 启动命令: bash start.sh")
