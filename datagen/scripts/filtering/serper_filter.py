#!/usr/bin/env python3
"""
Serper-based QA pair filter with producer-consumer pattern.
Producer: Search questions via Serper API
Consumer: Use string matching to verify if search results contain the answer
"""
import os
import json
import time
import http.client
import logging
import uuid
import re
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
from queue import Queue
from threading import Thread, Lock, Lock
from tqdm import tqdm
import glob

# ==============================================================================
# Configuration
# ==============================================================================
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
LOG_DIR = "serper_filter_wiki_history/logs"
OUTPUT_DIR = "serper_filter_wiki_history"
SEARCH_PROCESS_DIR = "serper_filter_wiki_history/search_process"

logging.basicConfig(level=logging.ERROR, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Global counters with thread-safe lock
stats_lock = Lock()
stats = {"positive": 0, "negative": 0, "total": 0, "positive_part": 1, "positive_current_count": 0}


# ==============================================================================
# String Matching Logic
# ==============================================================================
def extract_numbers(text: str) -> list:
    """Extract all numbers from text, including those with commas, decimals, etc."""
    # Find numbers with optional commas, decimals, and currency symbols
    # Patterns: 10,000 | 10000 | 10.5 | $10,000 | etc.
    number_pattern = r'[\$€£¥]?\s*\d+(?:[,\s]\d{3})*(?:\.\d+)?'
    matches = re.findall(number_pattern, text)
    
    numbers = []
    for match in matches:
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[\$€£¥\s]', '', match)
        # Remove commas
        cleaned = cleaned.replace(',', '')
        # Convert to float if decimal, else int
        try:
            if '.' in cleaned:
                numbers.append(float(cleaned))
            else:
                numbers.append(int(cleaned))
        except:
            pass
    
    return numbers


def chinese_number_to_int(text: str) -> list:
    """Extract Chinese numbers like 1万, 10万, etc. and convert to integers"""
    numbers = []
    
    # Pattern for Chinese numbers: digits + 万/千/百
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*万', 10000),
        (r'(\d+(?:\.\d+)?)\s*千', 1000),
        (r'(\d+(?:\.\d+)?)\s*百', 100),
    ]
    
    for pattern, multiplier in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                value = float(match) * multiplier
                numbers.append(int(value) if value.is_integer() else value)
            except:
                pass
    
    return numbers


def normalize_text(text: str) -> str:
    """
    Normalize text for matching:
    - Convert to lowercase
    - Replace various apostrophe types with standard '
    - Replace underscores and hyphens with spaces
    - Remove special characters except alphanumeric, spaces, apostrophes and decimals
    - Collapse multiple spaces
    """
    text = text.lower()
    # Normalize different apostrophe types
    text = text.replace("'", "'").replace("'", "'").replace("`", "'")
    # Replace underscores and hyphens with spaces
    text = text.replace("_", " ").replace("-", " ")
    # Remove special characters except alphanumeric, spaces, apostrophes and decimal points
    text = re.sub(r'[^\w\s\'\.]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def number_to_words(num: int) -> list:
    """
    Convert a number to various word representations in Chinese and English.
    For example: 2 → ['两', '二', '两人', '二人', '两名', '二名', 'two', 'Two']
    """
    # Chinese number words (0-20)
    chinese_map = {
        0: ['零'], 1: ['一'], 2: ['两', '二'], 3: ['三'], 4: ['四'],
        5: ['五'], 6: ['六'], 7: ['七'], 8: ['八'], 9: ['九'],
        10: ['十'], 11: ['十一'], 12: ['十二'], 13: ['十三'], 14: ['十四'],
        15: ['十五'], 16: ['十六'], 17: ['十七'], 18: ['十八'], 19: ['十九'],
        20: ['二十', '二〇', '廿']
    }
    
    # English number words (0-20)
    english_map = {
        0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
        5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
        10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
        15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen',
        20: 'twenty'
    }
    
    words = []
    
    # Chinese variants
    if num in chinese_map:
        for ch_word in chinese_map[num]:
            words.append(ch_word)
            # Add common suffixes
            words.extend([
                ch_word + '人', ch_word + '名', ch_word + '个',
                ch_word + '位', ch_word + '条', ch_word + '项'
            ])
    
    # English variants
    if num in english_map:
        eng_word = english_map[num]
        words.extend([eng_word, eng_word.capitalize(), eng_word.upper()])
        # Add common patterns
        words.extend([
            eng_word + ' people', eng_word + ' person',
            eng_word + ' items', eng_word + ' workers'
        ])
    
    return words


def answer_in_search_result(answer: str, search_result: str, question: str = "") -> bool:
    """
    Check if answer appears in search result.
    Returns True if answer is found (negative case - bad data)
    Returns False if answer is NOT found (positive case - good data)
    
    Logic:
    1. Extract numbers from answer (skip single digits unless only number)
       a) If no numbers, go to step 4, 5
       b) If answer numbers also in question, skip numeric matching (go to 4, 5)
       c) Otherwise, continue to step 2, 3
    2. Extract numbers from search result
    3. If search contains ALL answer numbers + text forms match -> negative
    4. Check text-only portions
    5. Check text-form numbers (e.g., "两人", "二名")
    """
    
    # Step 1: Extract numbers from answer
    answer_numbers = extract_numbers(answer)
    answer_chinese_nums = chinese_number_to_int(answer)
    all_answer_numbers = list(set(answer_numbers + answer_chinese_nums))
    
    # Filter significant numbers (skip single chars)
    significant_answer_numbers = []
    for n in all_answer_numbers:
        if n < 10 and len(str(n)) == 1:  # Skip single digit
            continue
        significant_answer_numbers.append(n)
    
    # Step 1a: No numbers -> skip to text matching
    # Step 1b: Check if answer numbers are in question
    should_check_numbers = False
    if significant_answer_numbers:
        if question:
            question_numbers = extract_numbers(question)
            question_chinese_nums = chinese_number_to_int(question)
            all_question_numbers = list(set(question_numbers + question_chinese_nums))
            
            # Check if any significant answer number is NOT in question
            # If a number is unique to answer (not in question), we should check it
            for num in significant_answer_numbers:
                if num not in all_question_numbers:
                    should_check_numbers = True
                    break
        else:
            # No question provided, check all numbers
            should_check_numbers = True
    
    # Steps 2-3: Numeric matching (only if should_check_numbers is True)
    if should_check_numbers:
        search_numbers = extract_numbers(search_result)
        search_chinese_nums = chinese_number_to_int(search_result)
        all_search_numbers = list(set(search_numbers + search_chinese_nums))
        
        # Filter numbers to check (exclude years, dates)
        numbers_to_check = []
        for n in significant_answer_numbers:
            if question:
                question_numbers = extract_numbers(question)
                question_chinese_nums = chinese_number_to_int(question)
                all_question_numbers = list(set(question_numbers + question_chinese_nums))
                if n in all_question_numbers:
                    continue  # Skip numbers that appear in question
            numbers_to_check.append(n)
        
        # Check if ALL numbers_to_check are in search
        if numbers_to_check:
            all_numbers_found = all(num in all_search_numbers for num in numbers_to_check)
            
            if all_numbers_found:
                # For small numbers (<= 31), require text form confirmation
                text_confirmed = False
                for num in numbers_to_check:
                    if num <= 31:
                        num_words = number_to_words(int(num))
                        if any(word in search_result for word in num_words):
                            text_confirmed = True
                            break
                
                # If any small number has text confirmation, or all are large numbers
                if text_confirmed or all(num > 31 for num in numbers_to_check):
                    return True  # negative - bad data
    
    # Steps 4-5: Text matching
    # Remove numbers from answer for text-only comparison
    answer_text_only = re.sub(r'\d+(?:,\d{3})*(?:\.\d+)?', '', answer)
    answer_text_only = re.sub(r'[$€¥£~]', '', answer_text_only).strip()
    
    normalized_answer = normalize_text(answer)
    normalized_search = normalize_text(search_result)
    normalized_text_only = normalize_text(answer_text_only)
    
    # Direct match of full answer
    if len(normalized_answer) > 3 and normalized_answer in normalized_search:
        return True
    
    # Generate text-form number variants (Step 5)
    text_variants = []
    for n in all_answer_numbers:
        if n <= 31:  # Only for small numbers
            num_words = number_to_words(int(n))
            for word in num_words:
                # Check if text variant appears in search
                if word and len(word) > 1 and word in search_result:
                    return True  # Found text form of number
    
    # Check text-only portions (Step 4)
    if normalized_text_only and len(normalized_text_only) > 3:
        answer_words = [w for w in normalized_text_only.split() if len(w) > 2]
        if answer_words:
            words_found = sum(1 for w in answer_words if w in normalized_search)
            threshold = max(1, int(0.7 * len(answer_words)))
            if words_found >= threshold:
                return True
    
    return False  # Not found - positive (good data)


# ==============================================================================
# Serper Search Function
# ==============================================================================
def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters"""
    return any('\u4E00' <= c <= '\u9FFF' for c in text)


def google_search(query: str, req_id: str = "", num_results: int = 50) -> str:
    """
    Perform Google search via Serper API with pagination support.
    Fetches up to num_results (default 50) by paginating through results (10 per page).
    """
    all_snippets = []
    pages_to_fetch = (num_results + 9) // 10  # Calculate pages needed (e.g., 50 results = 5 pages)
    
    for page_num in range(1, pages_to_fetch + 1):
        conn = http.client.HTTPSConnection("google.serper.dev")

        if contains_chinese(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn",
                "page": page_num
            })
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en",
                "page": page_num
            })

        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

        for retry in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                if retry == 4:
                    logger.error(f"{req_id} Search timeout on page {page_num}: {e}")
                    if page_num == 1:
                        return f"Google search timeout for query: '{query}'"
                    # If we have some results from previous pages, continue
                    break
                time.sleep(0.5 * (2 ** retry))
                conn = http.client.HTTPSConnection("google.serper.dev")

        try:
            data = json.loads(res.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"{req_id} Failed to parse response on page {page_num}: {e}")
            if page_num == 1:
                return f"Failed to parse search results for '{query}'"
            break

        if "organic" not in data or len(data["organic"]) == 0:
            # No more results on this page, stop pagination
            break

        # Process results from this page
        page_snippets = []
        for idx, page in enumerate(data["organic"], 1 + (page_num - 1) * 10):
            if len(all_snippets) + len(page_snippets) >= num_results:
                break
            
            date = f"\nDate published: {page['date']}" if "date" in page else ""
            source = f"\nSource: {page['source']}" if "source" in page else ""
            snippet = f"\n{page['snippet']}" if "snippet" in page else ""
            line = f"{idx}. [{page['title']}]({page['link']}){date}{source}\n{snippet}"
            line = line.replace("Your browser can't play this video.", "")
            page_snippets.append(line)
        
        all_snippets.extend(page_snippets)
        
        # Stop if we have enough results
        if len(all_snippets) >= num_results:
            break
        
        # Rate limiting between pages
        if page_num < pages_to_fetch:
            time.sleep(0.5)

    if not all_snippets:
        return f"No results found for '{query}'."

    result = f"A Google search for '{query}' found {len(all_snippets)} results:\n\n## Web Results\n" + "\n\n".join(all_snippets)
    return result


# ==============================================================================
# Verification Function
# ==============================================================================
def verify_answer_in_search(
    question: str,
    search_result: str, 
    answer: str,
    req_id: str = ""
) -> tuple[bool, Dict[str, Any]]:
    """
    Use string matching to verify if search results contain the answer.
    Returns (is_positive, process_dict)
    - is_positive=True: answer NOT found in search (positive/good data)
    - is_positive=False: answer found in search (negative/bad data)
    """
    start_time = time.time()
    
    # Extract numbers for debugging
    answer_numbers = extract_numbers(answer)
    search_numbers = extract_numbers(search_result)
    search_chinese_numbers = chinese_number_to_int(search_result)
    question_numbers = extract_numbers(question)
    
    # Check if answer appears in search result
    answer_found = answer_in_search_result(answer, search_result, question)
    
    elapsed = time.time() - start_time
    
    # is_positive = answer NOT found (good data, needs search)
    is_positive = not answer_found
    
    # Build process record
    process = {
        "request_id": req_id,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "answer_normalized": normalize_text(answer),
        "answer_numbers": answer_numbers,
        "question_numbers": question_numbers,
        "search_numbers": search_numbers,
        "search_chinese_numbers": search_chinese_numbers,
        "search_result": search_result,
        "answer_found_in_search": answer_found,
        "is_positive": is_positive,  # True = good data (answer NOT found)
        "elapsed_seconds": round(elapsed, 6)
    }
    
    return is_positive, process


# ==============================================================================
# Producer: Search Worker
# ==============================================================================
def producer_worker(input_queue: Queue, output_queue: Queue, worker_id: int, pbar: tqdm = None):
    """Producer: Read QA pairs and perform searches"""
    while True:
        item = input_queue.get()
        if item is None:  # Poison pill
            input_queue.task_done()
            break
        
        req_id = item["req_id"]
        qa_pair = item["data"]
        question = qa_pair["question"]
        
        try:
            search_result = google_search(question, req_id)
            output_queue.put({
                "req_id": req_id,
                "qa_pair": qa_pair,
                "search_result": search_result
            })
        except Exception as e:
            logger.error(f"[Producer-{worker_id}][{req_id}] Search failed: {e}")
            output_queue.put({
                "req_id": req_id,
                "qa_pair": qa_pair,
                "search_result": f"Error: {str(e)}"
            })
        finally:
            input_queue.task_done()


# ==============================================================================
# Consumer: Matching Verification Worker
# ==============================================================================
def consumer_worker(
    queue: Queue, 
    positive_file_base: str,
    negative_file: str,
    worker_id: int,
    jsonl_max_items: int,
    pbar: tqdm = None
):
    """Consumer: Verify answers using string matching and save results"""
    while True:
        item = queue.get()
        if item is None:  # Poison pill
            queue.task_done()
            break
        
        req_id = item["req_id"]
        qa_pair = item["qa_pair"]
        search_result = item["search_result"]
        
        try:
            # Verify with string matching
            is_positive, process = verify_answer_in_search(
                question=qa_pair["question"],
                search_result=search_result,
                answer=qa_pair["answer"],
                req_id=req_id
            )
            
            # Save process record
            process_file = os.path.join(SEARCH_PROCESS_DIR, f"{req_id}.json")
            with open(process_file, "w", encoding="utf-8") as f:
                json.dump(process, f, ensure_ascii=False, indent=2)
            
            # Save to positive or negative file
            if is_positive:
                # Check if need to split positive file
                with stats_lock:
                    if jsonl_max_items > 0 and stats["positive_current_count"] >= jsonl_max_items:
                        stats["positive_part"] += 1
                        stats["positive_current_count"] = 0
                    
                    current_part = stats["positive_part"]
                    stats["positive_current_count"] += 1
                
                # Determine positive file name
                if jsonl_max_items > 0:
                    positive_file = f"{positive_file_base}_part{current_part}.jsonl"
                else:
                    positive_file = f"{positive_file_base}.jsonl"
                    
                with open(positive_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
            else:
                with open(negative_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
            
            # Update stats
            with stats_lock:
                if is_positive:
                    stats["positive"] += 1
                else:
                    stats["negative"] += 1
                stats["total"] += 1
                
                if pbar:
                    pbar.set_postfix({
                        'pos': stats["positive"],
                        'neg': stats["negative"],
                        'part': stats["positive_part"] if jsonl_max_items > 0 else 1
                    })
                    pbar.update(1)
            
        except Exception as e:
            logger.error(f"[Consumer-{worker_id}][{req_id}] Processing failed: {e}")
        finally:
            queue.task_done()


# ==============================================================================
# Helper: Initialize resume state
# ==============================================================================
def initialize_resume_state(positive_file_base: str, jsonl_max_items: int):
    """
    Initialize stats for resume mode by scanning existing positive files.
    Returns: (processed_questions_set, positive_count, negative_count)
    """
    processed_questions = set()
    positive_count = 0
    negative_count = 0
    
    # Find all positive part files
    positive_pattern = f"{positive_file_base}_part*.jsonl"
    positive_files = sorted(glob.glob(positive_pattern))
    
    # Also check for single positive file (no split)
    single_positive = f"{positive_file_base}.jsonl"
    if os.path.exists(single_positive) and not positive_files:
        positive_files = [single_positive]
    
    # Process positive files
    max_part = 0
    last_part_count = 0
    
    for pos_file in positive_files:
        # Extract part number from filename
        match = re.search(r'_part(\d+)\.jsonl$', pos_file)
        if match:
            part_num = int(match.group(1))
            max_part = max(max_part, part_num)
        
        # Count items in this file
        file_count = 0
        with open(pos_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_questions.add(item.get("question", ""))
                    positive_count += 1
                    file_count += 1
                except:
                    pass
        
        # If this is the last (highest numbered) part, save its count
        if match and part_num == max_part:
            last_part_count = file_count
        elif not match:  # Single file without part number
            last_part_count = file_count
    
    # Check negative file
    negative_file = os.path.join(OUTPUT_DIR, "negative.jsonl")
    if os.path.exists(negative_file):
        with open(negative_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_questions.add(item.get("question", ""))
                    negative_count += 1
                except:
                    pass
    
    # Initialize stats
    with stats_lock:
        stats["positive"] = positive_count
        stats["negative"] = negative_count
        stats["total"] = positive_count + negative_count
        
        if jsonl_max_items > 0 and positive_files:
            # Resume from the last part
            stats["positive_part"] = max(max_part, 1)
            stats["positive_current_count"] = last_part_count
        else:
            stats["positive_part"] = 1
            stats["positive_current_count"] = positive_count
    
    return processed_questions, positive_count, negative_count


# ==============================================================================
# Main Function
# ==============================================================================
def main():
    global SERPER_API_KEY, OUTPUT_DIR, LOG_DIR, SEARCH_PROCESS_DIR

    parser = argparse.ArgumentParser(description="Serper-based QA filter")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--serper-api-key", type=str, default=os.getenv("SERPER_API_KEY", ""),
                        help="Serper API key. Defaults to SERPER_API_KEY.")
    parser.add_argument("--max-items", type=int, default=None, help="Max items to process (for debugging)")
    parser.add_argument("--producers", type=int, default=4, help="Number of producer threads")
    parser.add_argument("--consumers", type=int, default=8, help="Number of consumer threads")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--jsonl-max-items", type=int, default=0, help="Max items per positive JSONL file (0=no split)")
    
    args = parser.parse_args()
    if not args.serper_api_key:
        parser.error("--serper-api-key is required, or set SERPER_API_KEY.")

    SERPER_API_KEY = args.serper_api_key
    OUTPUT_DIR = args.output_dir
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    SEARCH_PROCESS_DIR = os.path.join(OUTPUT_DIR, "search_process")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SEARCH_PROCESS_DIR, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    
    # Setup output files
    positive_file_base = os.path.join(OUTPUT_DIR, "positive")
    negative_file = os.path.join(OUTPUT_DIR, "negative.jsonl")
    
    # Get existing processed items for resume
    processed_questions = set()
    skipped_count = 0
    
    if args.resume:
        processed_questions, pos_count, neg_count = initialize_resume_state(
            positive_file_base, args.jsonl_max_items
        )
        print(f"Resume mode: Found {len(processed_questions)} already processed items")
        print(f"  - Positive: {pos_count} (part {stats['positive_part']}, current count: {stats['positive_current_count']})")
        print(f"  - Negative: {neg_count}")
    
    # Count total items to process
    total_items = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if args.max_items and total_items + skipped_count >= args.max_items:
                break
            qa_pair = json.loads(line.strip())
            if args.resume and qa_pair.get("question", "") in processed_questions:
                skipped_count += 1
                continue
            total_items += 1
    
    if skipped_count > 0:
        print(f"Skipping {skipped_count} already processed items")
    
    print(f"Will process {total_items} new items")
    
    # Create queues
    input_queue = Queue(maxsize=500)
    output_queue = Queue(maxsize=500)
    
    # Create progress bar
    pbar = tqdm(total=total_items, desc="Processing", unit="item",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    # Start producer threads
    producer_threads = []
    for i in range(args.producers):
        t = Thread(target=producer_worker, args=(input_queue, output_queue, i, pbar))
        t.start()
        producer_threads.append(t)
    
    # Start consumer threads
    consumer_threads = []
    for i in range(args.consumers):
        t = Thread(target=consumer_worker, args=(
            output_queue, positive_file_base, negative_file, i, args.jsonl_max_items, pbar
        ))
        t.start()
        consumer_threads.append(t)
    
    # Read input and feed to producers
    with open(args.input, "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            if args.max_items and idx >= args.max_items:
                break
            
            qa_pair = json.loads(line.strip())
            
            # Skip if already processed (resume mode)
            if args.resume and qa_pair.get("question", "") in processed_questions:
                idx += 1
                continue
            
            req_id = f"{idx:06d}_{uuid.uuid4().hex[:8]}"
            
            input_queue.put({
                "req_id": req_id,
                "data": qa_pair
            })
            idx += 1
    
    # Send stop signals to producers
    for _ in range(args.producers):
        input_queue.put(None)
    
    # Wait for producers to finish
    for t in producer_threads:
        t.join()
    
    # Send stop signals to consumers
    for _ in range(args.consumers):
        output_queue.put(None)
    
    # Wait for consumers to finish
    for t in consumer_threads:
        t.join()
    
    pbar.close()
    
    # Print final results
    print("\n" + "=" * 72)
    print("Processing complete!")
    print(f"Total processed: {stats['total']}")
    print(f"  - Positive (good data): {stats['positive']}")
    print(f"  - Negative (bad data): {stats['negative']}")
    if args.jsonl_max_items > 0:
        print(f"  - Positive files: {stats['positive_part']} parts")
    print(f"\nResults saved to:")
    if args.jsonl_max_items > 0:
        print(f"  - Positive: {positive_file_base}_part*.jsonl")
    else:
        print(f"  - Positive: {positive_file_base}.jsonl")
    print(f"  - Negative: {negative_file}")
    print(f"  - Search Process: {SEARCH_PROCESS_DIR}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
