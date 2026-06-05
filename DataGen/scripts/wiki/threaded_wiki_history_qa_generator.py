#!/usr/bin/env python3
"""
Wiki History QA Generator - 从历史版本中提取QA对
"""
import argparse
import json
import re
import hashlib
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock
import time
import random
import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


SCRAPEDO_API_KEY = os.getenv("SCRAPEDO_API_KEY", "")


def parse_wikipedia_date(date_str: str) -> datetime:
    """
    解析 Wikipedia 的日期格式
    例如: "04:17, 27 July 2005" 或 "19:38, 22 February 2026"
    """
    try:
        # 移除可能的前导空格
        date_str = date_str.strip()
        
        # Wikipedia 格式: "HH:MM, DD Month YYYY"
        parsed = datetime.strptime(date_str, "%H:%M, %d %B %Y")
        return parsed
    except:
        try:
            # 尝试其他格式
            parsed = datetime.strptime(date_str, "%d %B %Y")
            return parsed
        except:
            # 如果解析失败，返回当前时间
            return datetime.now()


def format_date_for_question(date_str: str) -> str:
    """
    解析日期并加1天，格式化为问题中的日期格式
    例如: "04:17, 27 July 2005" -> "28 July 2005"
    """
    try:
        parsed_date = parse_wikipedia_date(date_str)
        # 加1天
        next_day = parsed_date + timedelta(days=1)
        # 格式化为 "DD Month YYYY"
        return next_day.strftime("%d %B %Y")
    except:
        return "unknown date"


def html_to_markdown(raw_html: str) -> str:
    """将 HTML 转换为 Markdown 格式"""
    if not raw_html:
        return ""
    
    try:
        soup = BeautifulSoup(raw_html, 'html.parser')
        
        # 移除不需要的标签
        for tag in soup(['script', 'style', 'noscript', 'iframe', 'svg', 'canvas']):
            tag.decompose()
        
        # 简单的文本提取
        text = soup.get_text(separator='\n', strip=True)
        
        # 清理多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    except:
        return ""


def fetch_html_via_scrape_api(url: str, proxies: dict, max_retries: int = 3) -> str:
    """使用 scrape.do API 获取 HTML"""
    if not SCRAPEDO_API_KEY:
        raise RuntimeError("SCRAPEDO_API_KEY is required.")
    api_url = f"http://api.scrape.do/?token={SCRAPEDO_API_KEY}&url={urllib.parse.quote(url)}"
    
    for attempt in range(max_retries):
        try:
            # 增加超时时间到60秒
            response = requests.get(api_url, timeout=60, proxies=proxies)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:
                # 速率限制 - 等待更长时间
                wait_time = 5 * (2 ** attempt)  # 5, 10, 20 秒
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                # 其他错误，等待后重试
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.Timeout:
            # 超时，记录并重试
            if attempt < max_retries - 1:
                print(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(2)
                continue
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    
    return None


def generate_qa_with_llm(markdown_content: str, timestamp_constraint: str, 
                         llm_host: str, llm_port: int, llm_api_key: str, 
                         llm_model: str) -> dict:
    """调用 LLM 生成 QA 对"""
    url = f"http://{llm_host}:{llm_port}/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are a data extraction and Q&A generation expert. Analyze the following Wikipedia webpage content from a historical version and extract all specific, factual data points.

For each concrete data point you find (numbers, statistics, dates, names, amounts, percentages, etc.), create a high-quality question-answer pair.

CRITICAL REQUIREMENTS:
1. ALL questions MUST start with the temporal constraint: "According to Wikipedia as of {timestamp_constraint}, "
2. Questions should be specific and include sufficient context to be self-contained
3. Answers should be concise and factual (like "500 billion", "2025", "15%", etc.)
4. Focus on extractable, verifiable data points - NOT opinions or general statements
5. Each Q&A pair should be completely independent and self-contained

Example format:
Q: According to Wikipedia as of {timestamp_constraint}, what is the population of Greenland?
A: 56,081

Now analyze this historical Wikipedia webpage and generate Q&A pairs:

{markdown_content}

Output format:
Return a JSON object with this structure:
{{
    "qa_pairs": [
        {{"question": "...", "answer": "..."}},
        {{"question": "...", "answer": "..."}}
    ],
    "total_pairs": <number>
}}
"""
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4000
    }
    
    if llm_model:
        payload["model"] = llm_model
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # 解析 JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(1))
        else:
            json_match = re.search(r'\{[^{}]*"qa_pairs"[^{}]*\[.*?\][^{}]*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(content)
        
        return {
            "success": True,
            "parsed_qa": parsed,
            "raw_response": content
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "raw_response": None
        }


def fix_revision_url(revision_url: str, oldid: str) -> str:
    """
    修复 revision URL，确保是纯 oldid 格式而不是 diff 格式
    例如：将 https://en.wikipedia.org/w/index.php?title=XXX&diff=prev&oldid=123
    转换为：https://en.wikipedia.org/w/index.php?title=XXX&oldid=123
    """
    try:
        if 'oldid=' in revision_url:
            # 提取 title 和 oldid
            if 'title=' in revision_url:
                title = revision_url.split('title=')[1].split('&')[0]
                return f'https://en.wikipedia.org/w/index.php?title={title}&oldid={oldid}'
            else:
                return f'https://en.wikipedia.org/w/index.php?oldid={oldid}'
        return revision_url
    except:
        return f'https://en.wikipedia.org/w/index.php?oldid={oldid}'


def process_single_revision(revision_data: dict, args, proxies: dict, 
                            progress_lock: Lock, stats: dict) -> dict:
    """处理单个 revision"""
    
    # 添加随机延迟避免速率限制 (0.5-1.5秒)
    time.sleep(random.uniform(0.5, 1.5))
    
    result = {
        "success": False,
        "revision_url": revision_data.get("revision_url", ""),
        "oldid": revision_data.get("oldid", ""),
        "original_timestamp": revision_data.get("timestamp", ""),
        "error": None
    }
    
    try:
        # 1. 修复 URL（确保是纯 oldid 格式）
        url = fix_revision_url(
            revision_data.get("revision_url", ""),
            revision_data.get("oldid", "")
        )
        result["revision_url"] = url
        
        # 2. 获取 HTML
        html = fetch_html_via_scrape_api(url, proxies)
        
        if not html:
            result["error"] = "Failed to fetch HTML"
            return result
        
        # 3. 转换为 Markdown
        markdown = html_to_markdown(html)
        
        if len(markdown) < 100:
            result["error"] = "Markdown too short"
            return result
        
        # 截断过长的内容
        if len(markdown) > 30000:
            markdown = markdown[:30000] + "\n\n[Content truncated...]"
        
        # 4. 生成时间约束
        timestamp_constraint = format_date_for_question(revision_data.get("timestamp_text", ""))
        
        # 5. 调用 LLM 生成 QA
        qa_result = generate_qa_with_llm(
            markdown, 
            timestamp_constraint,
            args.llm_host,
            args.llm_port,
            args.llm_api_key,
            args.llm_model_name
        )
        
        if not qa_result.get("success"):
            result["error"] = f"LLM failed: {qa_result.get('error')}"
            return result
        
        # 6. 组装结果
        result.update({
            "success": True,
            "timestamp_constraint": timestamp_constraint,
            "markdown_length": len(markdown),
            "html_length": len(html),
            "qa_pairs": qa_result.get("parsed_qa", {}).get("qa_pairs", []),
            "total_pairs": qa_result.get("parsed_qa", {}).get("total_pairs", 0),
            "model_raw_output": qa_result.get("raw_response", ""),
            "generation_timestamp": datetime.now().isoformat()
        })
        
        # 更新统计
        with progress_lock:
            stats["success"] += 1
            stats["total_qa_pairs"] += result["total_pairs"]
        
    except Exception as e:
        result["error"] = str(e)
        with progress_lock:
            stats["failed"] += 1
    
    return result


def load_middle_revisions_from_json_files(wiki_revisions_dir: Path) -> list:
    """从所有 JSON 文件中加载中间的 revision"""
    json_files = list(wiki_revisions_dir.glob("*.json"))
    
    revisions = []
    
    for json_file in tqdm(json_files, desc="Loading revisions"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data.get("success") or not data.get("revisions"):
                continue
            
            revision_list = data["revisions"]
            
            # 取中间的 revision
            middle_index = len(revision_list) // 2
            middle_revision = revision_list[middle_index]
            
            # 添加来源信息
            middle_revision["source_file"] = json_file.name
            middle_revision["source_url"] = data.get("url", "")
            
            revisions.append(middle_revision)
        
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    return revisions


def main():
    global SCRAPEDO_API_KEY

    parser = argparse.ArgumentParser(description="Wiki History QA Generator")
    
    # 输入输出
    parser.add_argument("--wiki-revisions-dir", type=str, required=True,
                       help="Directory containing revision JSON files")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for QA pairs")
    
    # LLM 配置
    parser.add_argument("--llm-host", type=str, default="127.0.0.1")
    parser.add_argument("--llm-port", type=int, default=8000)
    parser.add_argument("--llm-api-key", type=str, required=True)
    parser.add_argument("--llm-model-name", type=str, required=True)
    
    # 并发配置
    parser.add_argument("--llm-workers", type=int, default=10)
    
    # 索引范围
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--scrapedo-api-key", type=str, default=os.getenv("SCRAPEDO_API_KEY", ""),
                       help="scrape.do API key. Defaults to SCRAPEDO_API_KEY.")
    parser.add_argument("--proxy", type=str, default=os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or "",
                       help="Optional proxy URL used for both http and https requests.")
    
    args = parser.parse_args()
    if not args.scrapedo_api_key:
        parser.error("--scrapedo-api-key is required, or set SCRAPEDO_API_KEY.")
    SCRAPEDO_API_KEY = args.scrapedo_api_key
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 代理配置
    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }
    proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else {}
    
    print(f"📂 Loading revisions from: {args.wiki_revisions_dir}")
    
    # 加载所有中间 revisions
    revisions = load_middle_revisions_from_json_files(Path(args.wiki_revisions_dir))
    
    print(f"✅ Loaded {len(revisions)} middle revisions")
    
    # 过滤掉已经完成的revisions
    existing_oldids = {f.stem for f in output_dir.glob("*.json") if f.name != "wiki_history_summary.json" and f.name != "failed_revisions.json"}
    original_count = len(revisions)
    revisions = [r for r in revisions if str(r.get("oldid", "")) not in existing_oldids]
    skipped = original_count - len(revisions)
    
    if skipped > 0:
        print(f"⏭️  Skipped {skipped} already completed revisions")
    
    # 应用索引范围
    if args.end_index:
        revisions = revisions[args.start_index:args.end_index]
    else:
        revisions = revisions[args.start_index:]
    
    print(f"📊 Processing {len(revisions)} new revisions")
    
    # 统计
    stats = {
        "success": 0,
        "failed": 0,
        "total_qa_pairs": 0
    }
    progress_lock = Lock()
    
    # 存储失败记录
    failed_records = []
    failed_lock = Lock()
    
    # 并发处理
    with ThreadPoolExecutor(max_workers=args.llm_workers) as executor:
        futures = {
            executor.submit(process_single_revision, rev, args, proxies, progress_lock, stats): rev
            for rev in revisions
        }
        
        with tqdm(total=len(revisions), desc="Processing revisions") as pbar:
            for future in as_completed(futures):
                revision = futures[future]
                try:
                    result = future.result()
                    
                    # 保存结果
                    if result.get("success"):
                        oldid = result.get("oldid", "unknown")
                        output_file = output_dir / f"{oldid}.json"
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                    else:
                        # 如果处理失败但没有抛异常，也要计入 failed
                        if result.get("error"):
                            with progress_lock:
                                stats["failed"] += 1
                            # 记录失败详情
                            with failed_lock:
                                failed_records.append({
                                    "oldid": result.get("oldid", "unknown"),
                                    "revision_url": result.get("revision_url", ""),
                                    "error": result.get("error", "Unknown error"),
                                    "timestamp": result.get("original_timestamp", "")
                                })
                    
                except Exception as e:
                    with progress_lock:
                        stats["failed"] += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    "Success": stats["success"],
                    "Failed": stats["failed"],
                    "QA Pairs": stats["total_qa_pairs"]
                })
    
    # 保存统计和失败记录
    summary_file = output_dir / "wiki_history_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "avg_qa_pairs": stats["total_qa_pairs"] / max(stats["success"], 1),
            "total_processed": len(revisions),
            "success_rate": stats["success"] / len(revisions) * 100
        }, f, ensure_ascii=False, indent=2)
    
    # 保存失败记录
    if failed_records:
        failed_file = output_dir / "failed_revisions.json"
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_failed": len(failed_records),
                "failed_revisions": failed_records
            }, f, ensure_ascii=False, indent=2)
        print(f"📄 Failed records saved to: {failed_file}")
    
    print(f"\n{'='*80}")
    print(f"🎉 Processing Complete!")
    print(f"{'='*80}")
    print(f"✅ Success: {stats['success']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"📊 Total QA Pairs: {stats['total_qa_pairs']}")
    print(f"📈 Avg QA Pairs per revision: {stats['total_qa_pairs'] / max(stats['success'], 1):.1f}")
    print(f"📁 Output: {output_dir}")


if __name__ == "__main__":
    main()
