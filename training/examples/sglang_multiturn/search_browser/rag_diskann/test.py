#!/usr/bin/env python3
"""
快速测试脚本 - 测试DISKANN RAG服务
"""

import requests
import json
import time

# 服务地址
RAG_DISKANN_URL = "http://localhost:8018"

def test_diskann_health():
    """测试DISKANN RAG服务健康"""
    print("1️⃣  测试DISKANN RAG服务健康检查...")
    try:
        response = requests.get(f"{RAG_DISKANN_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ DISKANN RAG服务正常")
            print(f"      状态: {data.get('status')}")
            print(f"      服务ID: {data.get('server_id')}")
            print(f"      版本: {data.get('version')}")
            print(f"      向量精度: {data.get('vector_precision')}")
            print(f"      索引类型: {data.get('index_type')}")
            print(f"      模型加载: {'✅' if data.get('model_loaded') else '❌'}")
            print(f"      集合加载: {'✅' if data.get('collection_loaded') else '❌'}")
            print(f"      文档数量: {data.get('collection_entities', 0):,}")
            print(f"      优化特性: {data.get('optimization')}")
            return True
        else:
            print(f"   ❌ 健康检查失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ 连接失败: {e}")
        print(f"      请确保DISKANN服务已启动: python local_rag_diskann_server.py")
        return False

def test_diskann_search(search_type="hybrid"):
    """测试DISKANN搜索"""
    print(f"\n2️⃣  测试DISKANN搜索功能 ({search_type}模式)...")
    try:
        payload = {
            "query": "artificial intelligence and machine learning",
            "limit": 5,
            "search_type": search_type,
            "sparse_weight": 0.7,
            "dense_weight": 1.0
        }

        print(f"   查询: {payload['query']}")
        start_time = time.time()
        response = requests.post(f"{RAG_DISKANN_URL}/search", json=payload, timeout=30)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ 搜索成功")
            print(f"      返回结果数: {data.get('total')}")
            print(f"      总时间: {elapsed:.3f}秒")
            print(f"      Embedding时间: {data.get('embedding_time', 0):.3f}秒")
            print(f"      Milvus时间: {data.get('milvus_time', 0):.3f}秒")
            print(f"      搜索类型: {data.get('search_type')}")
            print(f"      向量精度: {data.get('vector_dtype')}")
            print(f"      索引类型: {data.get('index_type')}")

            # 显示前2个结果
            if data.get('results'):
                print(f"\n   📄 前2个搜索结果:")
                for i, result in enumerate(data['results'][:2], 1):
                    print(f"      {i}. {result.get('title', 'N/A')[:60]}")
                    print(f"         评分: {result.get('score', 0):.4f}")
                    print(f"         摘要: {result.get('snippet', '')[:100]}...")

            return True
        else:
            print(f"   ❌ 搜索失败: HTTP {response.status_code}")
            print(f"      错误: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ 请求失败: {e}")
        return False

def test_diskann_search_types():
    """测试所有搜索类型"""
    print("\n3️⃣  测试不同搜索类型...")

    types = ["hybrid", "dense", "sparse"]
    results = {}

    for search_type in types:
        print(f"\n   测试 {search_type} 搜索...")
        try:
            payload = {
                "query": "deep learning neural networks",
                "limit": 3,
                "search_type": search_type,
                "sparse_weight": 0.7,
                "dense_weight": 1.0
            }

            start_time = time.time()
            response = requests.post(f"{RAG_DISKANN_URL}/search", json=payload, timeout=30)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                results[search_type] = {
                    "success": True,
                    "count": data.get('total'),
                    "time": elapsed
                }
                print(f"      ✅ 成功: {data.get('total')}个结果, 耗时{elapsed:.3f}秒")
            else:
                results[search_type] = {"success": False}
                print(f"      ❌ 失败: HTTP {response.status_code}")
        except Exception as e:
            results[search_type] = {"success": False}
            print(f"      ❌ 失败: {e}")

    # 汇总
    success_count = sum(1 for r in results.values() if r.get("success"))
    print(f"\n   总结: {success_count}/{len(types)} 搜索类型测试通过")

    return success_count == len(types)

def main():
    print("="*60)
    print("🧪 DISKANN RAG服务 - 功能测试")
    print("="*60)
    print("📈 索引类型: DISKANN (适用于千万级数据)")
    print("🎯 向量精度: FP32 (DISKANN要求)")
    print("💾 优化特性: 磁盘索引，高吞吐量")
    print("")

    # 测试流程
    tests = [
        ("DISKANN服务健康检查", test_diskann_health),
        ("DISKANN混合搜索", lambda: test_diskann_search("hybrid")),
        ("所有搜索类型", test_diskann_search_types),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

        # 如果健康检查失败，跳过后续测试
        if not result and name == "DISKANN服务健康检查":
            print(f"\n⚠️  {name}失败，跳过后续测试")
            break

    # 汇总结果
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status} - {name}")

    print("")
    print(f"总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！DISKANN RAG服务运行正常")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查服务状态和日志")
        return 1

if __name__ == "__main__":
    exit(main())
