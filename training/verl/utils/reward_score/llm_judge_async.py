import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List

import httpx
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "examples", "sglang_multiturn", "search_browser", "tool_backend", ".env"))

REWARD_LOG_DIR = os.path.join(os.getcwd(), "logs", "llm_judge_rewards")
os.makedirs(REWARD_LOG_DIR, exist_ok=True)

LLM_JUDGE_API_BASE = os.getenv("LLM_JUDGE_API_BASE", "http://172.24.11.31/v1")
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "qwen")
LLM_JUDGE_MAX_RETRIES = int(os.getenv("LLM_JUDGE_MAX_RETRIES", "3"))
LLM_JUDGE_TIMEOUT = int(os.getenv("LLM_JUDGE_TIMEOUT", "120"))
LLM_JUDGE_CONCURRENCY = 512

_llm_judge_client = None

EVALUATION_PROMPT = """You are an evaluation assistant. Please determine if the predicted answer is semantically equivalent to the labeled answer.

Question: {question}

Labeled Answer: {correct_answer}

Predicted Answer: {response}

Please evaluate the answer and return a JSON object with the following format:
{{
  "reasoning": "A concise explanation of why the predicted answer is equivalent or not equivalent to the labeled answer.",
  "judgment": "Correct"
}}

If the answers are not equivalent, the "judgment" field should be "Incorrect".
Output ONLY the JSON object, without any markdown formatting or additional text.
"""


def extract_solution(solution_str):
    """从 solution 中提取 <answer>...</answer> 内的内容"""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if len(matches) < 1:
        return None
    return matches[-1].group(1).strip()


def normalize_answer(s):
    """标准化答案用于 EM 匹配"""
    import string
    
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    """EM 匹配检查"""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return True
    return False


def extract_json_from_response(response_text: str) -> Dict:
    """从响应中提取 JSON"""
    text = response_text.strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    
    json_match = re.search(r'\{[^{}]*"judgment"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def get_llm_judge_client():
    """获取或创建 LLM Judge 客户端"""
    global _llm_judge_client
    if _llm_judge_client is None:
        # 配置 httpx 连接池，与 LLM_JUDGE_CONCURRENCY 保持一致
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=LLM_JUDGE_CONCURRENCY,
                max_keepalive_connections=LLM_JUDGE_CONCURRENCY,
            ),
            timeout=httpx.Timeout(float(LLM_JUDGE_TIMEOUT), connect=10.0)
        )
        _llm_judge_client = AsyncOpenAI(
            api_key="EMPTY", 
            base_url=LLM_JUDGE_API_BASE, 
            http_client=http_client
        )
    return _llm_judge_client


async def judge_single(question: str, correct_answer: str, response: str, semaphore: asyncio.Semaphore) -> Dict:
    """单个样本的 LLM Judge"""
    client = get_llm_judge_client()
    
    async with semaphore:
        prompt = EVALUATION_PROMPT.format(question=question, correct_answer=correct_answer, response=response)
        
        for attempt in range(LLM_JUDGE_MAX_RETRIES):
            try:
                completion = await client.chat.completions.create(
                    model=LLM_JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512
                )
                
                result_text = completion.choices[0].message.content
                result_json = extract_json_from_response(result_text)
                
                if result_json and "judgment" in result_json:
                    is_correct = result_json["judgment"].lower() == "correct"
                    return {
                        "success": True,
                        "is_correct": is_correct,
                        "reasoning": result_json.get("reasoning", ""),
                        "raw_response": result_text
                    }
            except Exception as e:
                if attempt == LLM_JUDGE_MAX_RETRIES - 1:
                    return {"success": False, "error": str(e)}
                await asyncio.sleep(0.5 * (attempt + 1))
        
        return {"success": False, "error": "Max retries exceeded"}


async def judge_batch_async(batch_data: List[Dict]) -> List[Dict]:
    """批量 Judge"""
    semaphore = asyncio.Semaphore(LLM_JUDGE_CONCURRENCY)
    tasks = [judge_single(item["question"], item["correct_answer"], item["response"], semaphore) for item in batch_data]
    return await asyncio.gather(*tasks)


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """批量计算 reward score（参照 search_r1_like_qa_em.py）"""
    total_samples = len(solution_strs)
    need_judge_count = 0
    no_answer_label = 0
    llm_judge_count = 0
    em_fallback_count = 0
    correct_count = 0
    
    # 初始化结果列表，保持索引顺序
    results = [None] * total_samples
    batch_judge_data = []
    
    for i in range(total_samples):
        output_str = solution_strs[i]
        ground_truth = ground_truths[i]
        
        # 没有输出字符串
        if not output_str:
            no_answer_label += 1
            results[i] = {"score": 0.0, "correct": False, "method": "no_output", "pred_ans": "", "reason": "", "raw_response": ""}
            continue
        
        # 没有 ground truth
        if not ground_truth:
            no_answer_label += 1
            results[i] = {"score": 0.0, "correct": False, "method": "no_label", "pred_ans": "", "reason": "", "raw_response": ""}
            continue
        
        # 提取答案
        extracted_answer = extract_solution(output_str)
        if not extracted_answer:
            no_answer_label += 1
            results[i] = {"score": 0.0, "correct": False, "method": "no_extraction", "pred_ans": "", "reason": "", "raw_response": ""}
            continue
        
        # 提取 question（和 search_r1 一样，直接从 extra_infos 拿）
        question = extra_infos[i].get("question", "") if isinstance(extra_infos[i], dict) else ""

        # 处理 ground_truth 可能是字符串或字典的情况
        if isinstance(ground_truth, str):
            golden_answers = [ground_truth] if ground_truth else []
        else:
            golden_answers = ground_truth.get("target", []) if isinstance(ground_truth, dict) else []

        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        
        need_judge_count += 1
        batch_judge_data.append({
            "idx": i,
            "question": question,
            "correct_answer": ", ".join(golden_answers),
            "response": extracted_answer,
            "golden_answers": golden_answers
        })
    
    # 批量调用 LLM Judge
    if batch_judge_data:
        judge_results = asyncio.run(judge_batch_async(batch_judge_data))
        
        for item, judge_result in zip(batch_judge_data, judge_results):
            idx = item["idx"]
            
            if judge_result["success"]:
                llm_judge_count += 1
                is_correct = judge_result["is_correct"]
                if is_correct:
                    correct_count += 1
                
                results[idx] = {
                    "score": 1.0 if is_correct else 0.0,
                    "correct": is_correct,
                    "method": "llm_judge",
                    "pred_ans": item["response"],
                    "reason": judge_result.get("reasoning", ""),
                    "raw_response": judge_result.get("raw_response", "")
                }
            else:
                # EM 回退
                em_fallback_count += 1
                is_correct = em_check(item["response"], item["golden_answers"])
                if is_correct:
                    correct_count += 1
                
                results[idx] = {
                    "score": 1.0 if is_correct else 0.0,
                    "correct": is_correct,
                    "method": "em_fallback",
                    "pred_ans": item["response"],
                    "reason": f"LLM Judge failed: {judge_result.get('error', '')}",
                    "raw_response": ""
                }
    
    # 只在批量计算时打印统计（避免单样本 reward 计算时刷屏）
    if total_samples > 1:
        print(f"[LLM Judge] 📊 统计 | 总样本: {total_samples} | 需要Judge: {need_judge_count} | "
              f"无答案标签: {no_answer_label} | LLM Judge: {llm_judge_count} | EM回退: {em_fallback_count} | 正确: {correct_count}")
    
    # 保存数据
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    save_path = os.path.join(REWARD_LOG_DIR, f"reward_{timestamp}.jsonl")
    with open(save_path, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(results):
            item["index"] = idx
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return results
