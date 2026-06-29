#!/usr/bin/env python3
"""
统计 rollout_trajectory 和 validation_trajectory 中所有 JSONL 文件的
search tool call 次数（按 query 数量计）和 visit tool call 次数（按 url 数量计）。

用法: python count_tool_calls.py [--workers N] [--dirs DIR1 DIR2 ...]
"""
import json
import re
import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 预编译正则: 匹配 <tool_call>...</tool_call> 中的 JSON
TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL
)


def count_file(filepath: str) -> dict:
    """统计单个 JSONL 文件中的 search query 次数和 visit url 次数。"""
    total_search_queries = 0
    total_visit_urls = 0
    total_search_calls = 0
    total_visit_calls = 0
    total_lines = 0
    errors = 0

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                # 只从 output 字段统计（input 只含模板示例，不含实际 tool call）
                full_text = record.get('output', '')
                tool_calls = TOOL_CALL_PATTERN.findall(full_text)

                for tc_str in tool_calls:
                    try:
                        tc = json.loads(tc_str)
                    except json.JSONDecodeError:
                        continue

                    name = tc.get('name', '')
                    args = tc.get('arguments', {})

                    if name == 'search':
                        total_search_calls += 1
                        queries = args.get('query', [])
                        if isinstance(queries, list):
                            total_search_queries += len(queries)
                        elif isinstance(queries, str):
                            total_search_queries += 1
                    elif name == 'visit':
                        total_visit_calls += 1
                        urls = args.get('url', [])
                        if isinstance(urls, list):
                            total_visit_urls += len(urls)
                        elif isinstance(urls, str):
                            total_visit_urls += 1

    except Exception as e:
        return {
            'filepath': filepath,
            'error': str(e),
            'total_lines': 0,
            'search_calls': 0,
            'search_queries': 0,
            'visit_calls': 0,
            'visit_urls': 0,
            'parse_errors': 0,
        }

    return {
        'filepath': filepath,
        'error': None,
        'total_lines': total_lines,
        'search_calls': total_search_calls,
        'search_queries': total_search_queries,
        'visit_calls': total_visit_calls,
        'visit_urls': total_visit_urls,
        'parse_errors': errors,
    }


def find_jsonl_files(dirs: list) -> list:
    """递归查找所有 .jsonl 文件。"""
    files = []
    for d in dirs:
        for root, _, filenames in os.walk(d):
            for fn in filenames:
                if fn.endswith('.jsonl'):
                    files.append(os.path.join(root, fn))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description='统计 JSONL 文件中的 search/visit tool call 次数')
    parser.add_argument('--dirs', nargs='+', default=[
        './rollout_trajectory',
        './validation_trajectory',
    ], help='要扫描的目录列表 (相对/绝对路径均可)')
    parser.add_argument('--workers', type=int, default=32, help='并行 worker 数')
    parser.add_argument('--debug', type=str, default=None, help='调试模式：只处理单个文件')
    args = parser.parse_args()

    if args.debug:
        # 调试模式：单文件
        result = count_file(args.debug)
        print(f"文件: {result['filepath']}")
        print(f"  记录数: {result['total_lines']}")
        print(f"  search 调用次数: {result['search_calls']}")
        print(f"  search query 总数: {result['search_queries']}")
        print(f"  visit 调用次数: {result['visit_calls']}")
        print(f"  visit url 总数: {result['visit_urls']}")
        print(f"  解析错误: {result['parse_errors']}")
        if result['error']:
            print(f"  文件级错误: {result['error']}")
        return

    # 查找所有 JSONL 文件
    all_files = find_jsonl_files(args.dirs)
    print(f"找到 {len(all_files)} 个 JSONL 文件，使用 {args.workers} 个 worker 并行处理...")

    # 汇总统计
    grand_total = {
        'files': 0,
        'lines': 0,
        'search_calls': 0,
        'search_queries': 0,
        'visit_calls': 0,
        'visit_urls': 0,
        'parse_errors': 0,
        'file_errors': 0,
    }

    # 按目录统计
    dir_stats = {}

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(count_file, fp): fp for fp in all_files}
        for future in as_completed(futures):
            result = future.result()
            done += 1

            if done % 500 == 0 or done == len(all_files):
                print(f"  进度: {done}/{len(all_files)} ({done*100//len(all_files)}%)")

            # 汇总
            grand_total['files'] += 1
            grand_total['lines'] += result['total_lines']
            grand_total['search_calls'] += result['search_calls']
            grand_total['search_queries'] += result['search_queries']
            grand_total['visit_calls'] += result['visit_calls']
            grand_total['visit_urls'] += result['visit_urls']
            grand_total['parse_errors'] += result['parse_errors']
            if result['error']:
                grand_total['file_errors'] += 1

            # 按顶级实验目录统计
            fp = result['filepath']
            # 提取顶层目录如: rollout_trajectory/xxx/ 或 validation_trajectory/xxx/
            for base_dir in args.dirs:
                if fp.startswith(base_dir):
                    rel = os.path.relpath(fp, base_dir)
                    top_exp = rel.split(os.sep)[0]
                    key = os.path.basename(base_dir) + '/' + top_exp
                    if key not in dir_stats:
                        dir_stats[key] = {
                            'files': 0, 'lines': 0,
                            'search_calls': 0, 'search_queries': 0,
                            'visit_calls': 0, 'visit_urls': 0,
                        }
                    dir_stats[key]['files'] += 1
                    dir_stats[key]['lines'] += result['total_lines']
                    dir_stats[key]['search_calls'] += result['search_calls']
                    dir_stats[key]['search_queries'] += result['search_queries']
                    dir_stats[key]['visit_calls'] += result['visit_calls']
                    dir_stats[key]['visit_urls'] += result['visit_urls']
                    break

    # 打印结果
    print("\n" + "=" * 80)
    print("总统计结果")
    print("=" * 80)
    print(f"  文件总数:          {grand_total['files']}")
    print(f"  记录总数 (JSONL行): {grand_total['lines']}")
    print(f"  search 调用次数:    {grand_total['search_calls']}")
    print(f"  search query 总数:  {grand_total['search_queries']}")
    print(f"  visit 调用次数:     {grand_total['visit_calls']}")
    print(f"  visit url 总数:     {grand_total['visit_urls']}")
    print(f"  解析错误数:         {grand_total['parse_errors']}")
    print(f"  文件读取错误数:     {grand_total['file_errors']}")

    # 按实验目录排序输出
    print("\n" + "=" * 80)
    print("按实验目录统计 (按 search_queries 降序)")
    print("=" * 80)
    print(f"{'目录':<90} {'文件':>6} {'记录':>8} {'search调用':>10} {'search_query':>12} {'visit调用':>10} {'visit_url':>10}")
    print("-" * 150)
    for key in sorted(dir_stats, key=lambda k: dir_stats[k]['search_queries'], reverse=True):
        s = dir_stats[key]
        print(f"{key:<90} {s['files']:>6} {s['lines']:>8} {s['search_calls']:>10} {s['search_queries']:>12} {s['visit_calls']:>10} {s['visit_urls']:>10}")


if __name__ == '__main__':
    main()
