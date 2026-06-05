#!/usr/bin/env python3
"""
Browse 工具统计脚本
统计 rollout trajectory jsonl 文件中 browse/visit 工具的成功/失败比例

用法: python browse_stats.py <jsonl_path>
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def analyze_visit_responses(jsonl_path: str):
    """分析 visit 工具的响应"""
    
    visit_responses = []
    total_samples = 0
    samples_with_visit = 0
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            total_samples += 1
            d = json.loads(line)
            text = d.get('input', '') + d.get('output', '')
            
            # 找到所有 tool_response
            responses = re.findall(r'<tool_response>(.*?)</tool_response>', text, re.DOTALL)
            
            sample_has_visit = False
            for resp in responses:
                # 检查是否是 visit 工具的响应:
                # 1. 正常响应包含 "The useful information in"
                # 2. Connection Error 响应包含 "Browse error"
                if 'The useful information in' in resp or 'Browse error' in resp:
                    visit_responses.append(resp)
                    sample_has_visit = True
            
            if sample_has_visit:
                samples_with_visit += 1
    
    print(f"=" * 60)
    print(f"Browse/Visit 工具统计报告")
    print(f"=" * 60)
    print(f"JSONL 文件: {jsonl_path}")
    print(f"总样本数: {total_samples}")
    print(f"使用 visit 工具的样本数: {samples_with_visit} ({samples_with_visit/total_samples*100:.2f}%)")
    print(f"visit 工具调用总次数: {len(visit_responses)}")
    print()
    
    # 分类统计
    success_count = 0
    failure_count = 0
    failure_details = Counter()
    
    for resp in visit_responses:
        is_failure = False
        failure_reason = None
        
        # 类型1: HTTP 连接错误 (Connection Error)
        if 'Browse error:' in resp or 'API Call Failed' in resp:
            is_failure = True
            if 'ConnectionResetError' in resp:
                failure_reason = 'Connection Error: ConnectionResetError'
            elif 'RemoteDisconnected' in resp:
                failure_reason = 'Connection Error: RemoteDisconnected'
            elif 'Timeout' in resp:
                failure_reason = 'Connection Error: Timeout'
            else:
                failure_reason = 'Connection Error: Other'
        else:
            # 类型2: 后端返回的标准失败格式
            evidence_match = re.search(r'Evidence in page:\s*\n(.+?)\n\nSummary:', resp, re.DOTALL)
            
            if evidence_match:
                evidence = evidence_match.group(1).strip()
                
                # 标准失败响应的 Evidence 内容
                if 'could not be accessed' in evidence or 'could not be processed' in evidence:
                    is_failure = True
                    failure_reason = '后端失败: URL 无法访问 (Cache miss/Summary失败)'
                elif not evidence or evidence == 'N/A':
                    is_failure = True
                    failure_reason = '后端失败: 无 Evidence 内容'
                # 其他情况（有实际 Evidence）算成功
            else:
                # 没有 Evidence 标记，可能是格式异常
                if 'could not be accessed' in resp or 'could not be processed' in resp:
                    is_failure = True
                    failure_reason = '后端失败: 格式异常'
        
        if is_failure:
            failure_count += 1
            failure_details[failure_reason] += 1
        else:
            success_count += 1
    
    total = success_count + failure_count
    
    print(f"-" * 60)
    print(f"Visit 工具调用结果统计")
    print(f"-" * 60)
    print(f"✅ 成功: {success_count:>6} ({success_count/total*100:>6.2f}%)")
    print(f"❌ 失败: {failure_count:>6} ({failure_count/total*100:>6.2f}%)")
    print()
    
    if failure_details:
        print(f"-" * 60)
        print(f"失败原因分解 (基于响应内容推断)")
        print(f"-" * 60)
        for reason, count in failure_details.most_common():
            print(f"  {reason}: {count} ({count/total*100:.2f}%)")
    
    print()
    print(f"=" * 60)
    print(f"总结")
    print(f"=" * 60)
    print(f"Visit 成功率: {success_count/total*100:.2f}%")
    print(f"Visit 失败率: {failure_count/total*100:.2f}%")
    print()
    print("注: 具体失败原因 (Connection Error, Cache miss, Summary failed 等)")
    print("    需要从 browser_service 后端日志分析")
    
    return {
        'total_samples': total_samples,
        'samples_with_visit': samples_with_visit,
        'total_visit_calls': len(visit_responses),
        'success': success_count,
        'failure': failure_count,
        'success_rate': success_count/total*100 if total > 0 else 0,
        'failure_rate': failure_count/total*100 if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Browse 工具统计脚本')
    parser.add_argument('jsonl_path', help='JSONL 文件路径')
    args = parser.parse_args()
    
    if not Path(args.jsonl_path).exists():
        print(f"错误: 文件不存在 - {args.jsonl_path}")
        return
    
    analyze_visit_responses(args.jsonl_path)


if __name__ == '__main__':
    main()
