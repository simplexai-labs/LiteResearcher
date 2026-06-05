#!/usr/bin/env python3
"""
测试 GoogleSearchTool 的去重功能
使用配置: search_service_url: http://47.111.147.142:8010/search
"""
import json
import requests
import re

# 配置
SEARCH_SERVICE_URL = "http://47.111.147.142:8010/search"
TIMEOUT = 30

def call_search_api(query: str, limit: int = 10) -> dict:
    """调用搜索 API"""
    payload = {"query": query, "search_type": "hybrid", "limit": limit}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(SEARCH_SERVICE_URL, json=payload, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        return {}

def format_search_results(query: str, results: list) -> str:
    """格式化搜索结果"""
    num_results = len(results)
    output = f"A Google search for '{query}' found {num_results} results:\n\n## Web Results\n"
    for idx, result in enumerate(results, 1):
        title = result.get("title", "No title")
        link = result.get("link", "")
        snippet = result.get("snippet", "")
        output += f"{idx}. [{title}]({link})\n{snippet}\n\n"
    return output.strip()

def parse_search_result_block(block: str) -> list:
    """解析搜索结果块"""
    results = []
    pattern = r'(\d+)\.\s*\[([^\]]*)\]\(([^)]+)\)\n(.*?)(?=\n\d+\.\s*\[|\Z)'
    matches = re.findall(pattern, block, re.DOTALL)
    for match in matches:
        idx, title, link, snippet = match
        results.append({
            'title': title.strip(),
            'link': link.strip(),
            'snippet': snippet.strip()
        })
    return results

def deduplicate_results(all_results: list, masked_url: str = "") -> tuple:
    """去重"""
    seen_links = set()
    deduped = []
    for result in all_results:
        link = result.get('link', '')
        if masked_url and masked_url in link:
            continue
        if link and link not in seen_links:
            seen_links.add(link)
            deduped.append(result)
    return deduped, len(all_results), len(deduped)

def format_deduped_results(query_list: list, deduped_results: list) -> str:
    """格式化去重后的结果"""
    if not deduped_results:
        return f"A Google search for '{', '.join(query_list)}' found 0 results."
    
    num_results = len(deduped_results)
    queries_str = "', '".join(query_list)
    output = f"A Google search for '{queries_str}' found {num_results} results:\n\n## Web Results\n"
    for idx, result in enumerate(deduped_results, 1):
        title = result.get('title', 'No title')
        link = result.get('link', '')
        snippet = result.get('snippet', '')
        output += f"{idx}. [{title}]({link})\n{snippet}\n\n"
    return output.strip()

def test_dedup():
    """测试去重功能"""
    print("=" * 60)
    print("测试 GoogleSearchTool 去重功能")
    print("=" * 60)
    
    # 使用相似的 query 来增加重复概率
    query_list = [
        "capital punishment in the United States",
        "death row inmates in USA",
        "US death penalty statistics"
    ]
    
    print(f"\n📝 测试查询: {query_list}")
    print("-" * 60)
    
    # 收集所有结果
    all_parsed_results = []
    all_raw_results = []
    
    for query in query_list:
        print(f"\n🔍 查询: {query}")
        api_response = call_search_api(query, limit=10)
        
        if not api_response:
            print(f"  ⚠️  无结果")
            continue
        
        results = api_response.get("results", [])
        print(f"  ✅ 返回 {len(results)} 个结果")
        
        # 格式化并解析
        formatted = format_search_results(query, results)
        all_raw_results.append(formatted)
        
        parsed = parse_search_result_block(formatted)
        all_parsed_results.extend(parsed)
        
        # 打印前 3 个 URL
        for i, r in enumerate(parsed[:3], 1):
            print(f"     {i}. {r['link'][:60]}...")
    
    print("\n" + "=" * 60)
    print("📊 去重统计")
    print("=" * 60)
    
    # 统计原始重复
    all_links = [r['link'] for r in all_parsed_results]
    unique_links = set(all_links)
    
    print(f"\n原始结果总数: {len(all_parsed_results)}")
    print(f"唯一 URL 数: {len(unique_links)}")
    print(f"重复 URL 数: {len(all_parsed_results) - len(unique_links)}")
    print(f"重复比例: {(len(all_parsed_results) - len(unique_links)) / len(all_parsed_results) * 100:.1f}%")
    
    # 执行去重
    deduped_results, original_count, deduped_count = deduplicate_results(all_parsed_results)
    
    print(f"\n去重后结果数: {deduped_count}")
    print(f"节省: {original_count - deduped_count} 个结果")
    
    # 格式化去重后的结果
    deduped_text = format_deduped_results(query_list, deduped_results)
    
    # 原始合并文本
    original_text = "\n\n=======\n\n".join(all_raw_results)
    
    print(f"\n原始文本长度: {len(original_text)} 字符")
    print(f"去重后文本长度: {len(deduped_text)} 字符")
    print(f"节省字符: {len(original_text) - len(deduped_text)} ({(len(original_text) - len(deduped_text)) / len(original_text) * 100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("📄 去重后结果预览 (前 1000 字符)")
    print("=" * 60)
    print(deduped_text[:1000])
    
    print("\n✅ 测试完成!")
    return True

def test_url_mask():
    """测试 URL mask + 去重"""
    print("\n" + "=" * 60)
    print("测试 URL Mask + 去重")
    print("=" * 60)
    
    query = "capital punishment United States"
    api_response = call_search_api(query, limit=10)
    
    if not api_response:
        print("❌ API 调用失败")
        return False
    
    results = api_response.get("results", [])
    formatted = format_search_results(query, results)
    parsed = parse_search_result_block(formatted)
    
    if not parsed:
        print("❌ 无结果")
        return False
    
    # 选择第一个 URL 作为 masked_url
    masked_url = parsed[0]['link']
    print(f"\n🎭 Masked URL: {masked_url[:60]}...")
    
    # 去重 + mask
    deduped, orig, final = deduplicate_results(parsed, masked_url)
    
    print(f"\n原始数量: {orig}")
    print(f"去重+Mask后: {final}")
    print(f"被移除: {orig - final}")
    
    # 验证 masked_url 不在结果中
    masked_in_result = any(masked_url in r['link'] for r in deduped)
    print(f"\nMasked URL 是否被移除: {'✅ 是' if not masked_in_result else '❌ 否'}")
    
    return not masked_in_result

if __name__ == "__main__":
    print("\n🚀 开始测试\n")
    
    # 测试 1: 去重功能
    test_dedup()
    
    # 测试 2: URL mask + 去重
    test_url_mask()
    
    print("\n🎉 所有测试完成!")
