#!/usr/bin/env python3
"""
为现有的 JSONL 文件添加 token 数量统计

用法: python add_token_counts.py <jsonl_path> [--model_path MODEL_PATH]

默认使用 tiktoken 的 cl100k_base 编码器，也可以指定 HuggingFace 模型路径使用其 tokenizer
"""
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm


def count_tokens_tiktoken(text: str, encoding) -> int:
    """使用 tiktoken 计算 token 数量"""
    if not text:
        return 0
    return len(encoding.encode(text))


def count_tokens_hf(text: str, tokenizer) -> int:
    """使用 HuggingFace tokenizer 计算 token 数量"""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def process_jsonl(input_path: str, output_path: str = None, model_path: str = None):
    """处理 JSONL 文件，添加 token 数量"""
    
    # 初始化 tokenizer
    if model_path:
        print(f"Loading HuggingFace tokenizer from: {model_path}")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        count_tokens = lambda text: count_tokens_hf(text, tokenizer)
    else:
        print("Using tiktoken cl100k_base encoding")
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        count_tokens = lambda text: count_tokens_tiktoken(text, encoding)
    
    # 读取文件
    input_path = Path(input_path)
    if output_path is None:
        # 默认输出到同目录下，添加 _with_tokens 后缀
        output_path = input_path.parent / f"{input_path.stem}_with_tokens{input_path.suffix}"
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # 先统计行数
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Total lines: {total_lines}")
    
    # 处理每一行
    processed_lines = []
    stats = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "max_input_tokens": 0,
        "max_output_tokens": 0,
        "max_total_tokens": 0,
    }
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing"):
            try:
                data = json.loads(line.strip())
                
                # 计算 token 数量（如果还没有）
                if 'input_tokens' not in data:
                    inp = data.get('input', '')
                    data['input_tokens'] = count_tokens(inp)
                
                if 'output_tokens' not in data:
                    out = data.get('output', '')
                    data['output_tokens'] = count_tokens(out)
                
                if 'total_tokens' not in data:
                    data['total_tokens'] = data['input_tokens'] + data['output_tokens']
                
                # 更新统计
                stats["total_input_tokens"] += data['input_tokens']
                stats["total_output_tokens"] += data['output_tokens']
                stats["total_tokens"] += data['total_tokens']
                stats["max_input_tokens"] = max(stats["max_input_tokens"], data['input_tokens'])
                stats["max_output_tokens"] = max(stats["max_output_tokens"], data['output_tokens'])
                stats["max_total_tokens"] = max(stats["max_total_tokens"], data['total_tokens'])
                
                processed_lines.append(json.dumps(data, ensure_ascii=False))
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                processed_lines.append(line.strip())
    
    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines) + '\n')
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("Token Statistics:")
    print("=" * 50)
    print(f"Total samples: {total_lines}")
    print(f"Avg input tokens: {stats['total_input_tokens'] / total_lines:.1f}")
    print(f"Avg output tokens: {stats['total_output_tokens'] / total_lines:.1f}")
    print(f"Avg total tokens: {stats['total_tokens'] / total_lines:.1f}")
    print(f"Max input tokens: {stats['max_input_tokens']}")
    print(f"Max output tokens: {stats['max_output_tokens']}")
    print(f"Max total tokens: {stats['max_total_tokens']}")
    print("=" * 50)
    print(f"\nOutput written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Add token counts to JSONL file')
    parser.add_argument('jsonl_path', help='Path to the JSONL file')
    parser.add_argument('--output', '-o', help='Output path (default: add _with_tokens suffix)')
    parser.add_argument('--model_path', '-m', help='HuggingFace model path for tokenizer (default: use tiktoken)')
    parser.add_argument('--inplace', '-i', action='store_true', help='Modify file in place')
    
    args = parser.parse_args()
    
    output_path = args.jsonl_path if args.inplace else args.output
    process_jsonl(args.jsonl_path, output_path, args.model_path)


if __name__ == "__main__":
    main()



