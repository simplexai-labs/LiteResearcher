#!/usr/bin/env python3
"""
Quality filter for QA pairs from qa_outputs folder.
Filters based on: independence, answer precision, clarity, answerability, and avoiding essay questions.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import hashlib


# LLM Configuration. Values are set from CLI/env in main().
LLM_HOST = "127.0.0.1"
LLM_PORT = 8000
LLM_API_KEY = ""
LLM_MODEL = ""

FILTER_PROMPT = """Please evaluate the quality of the following question-answer pair. The QA pair must satisfy ALL of the following conditions to pass:

1. Question Independence: The question can be understood without any context
2. Answer Specificity & Verifiability: The answer must be VERIFIABLE and consist of specific, concrete information such as:
   - A specific number (e.g., "42", "1,500 members")
   - A particular name (e.g., "John Smith", "Eiffel Tower")
   - An exact model/designation (e.g., "Model T", "Boeing 747")
   - A precise date/year (e.g., "1945", "March 15, 1990")
   - A specific location (e.g., "Paris", "123 Main Street")
   The answer should NOT be a descriptive list, explanation, or multiple unverifiable items
3. Question Unambiguity: The question must be unambiguous with only one clear interpretation
4. Question Answerability: The question is clearly stated and it's obvious what is being asked
5. Avoid Open-ended Questions: Must allow for a definite, measurable short answer, not explanatory essay-type responses. Questions should NOT use words like "how", "why", "怎么", "如何", "为什么" that invite explanations
6. Avoid Oversimplicity: The question and answer must not be common sense; if you think it's too simple or you can easily give the answer directly, this QA pair should be filtered out
7. Time Specificity: The question must have a specific, concrete time constraint if it refers to time-dependent information. Questions with vague temporal references like "latest", "most recent", "as of now", "currently", "so far", "up to now", "到目前为止", "最新" are NOT allowed. However, questions with specific dates, years, or time periods (e.g., "in 2025", "after the 2024 Olympics", "by June 2023") are acceptable.

Question-Answer Pair:
Question: {question}
Answer: {answer}

Please provide your reasoning for each condition, then give your final answer in a box. Answer true if and only if the QA pair satisfies ALL 7 conditions above. Otherwise, answer false.

Here are some examples:

--------------------
Example 1:
Question: Flula Borg appeared in Season 2 Episode 3 "Like a Boss" of "Younger" on Apple TV in 2024. What important event were Liza and Kelsey preparing for in this episode?
Answer: In "Younger" Season 2 Episode 3 "Like a Boss", Liza and Kelsey were preparing for the launch of their new publishing imprint while facing massive online criticism.

Reasoning:
1. Question Independence: true - The question is self-contained
2. Answer Specificity: false - The answer is a descriptive explanation, not a specific fact
3. Question Unambiguity: true - Clear what is being asked
4. Question Answerability: true - The question is clearly stated
5. Avoid Open-ended Questions: false - Requires explanatory description, explanatory essay-type responses
6. Avoid Oversimplicity: true - Not common sense: I need to search to answer the question
7. Time Specificity: true - Contains specific year "2024"

There are multiple false conditions, therefore the answer is \\boxed{{false}}

Example 2:
Question: What was the gun configuration of the first production variant J22A (or J22 UBv) of the J22 fighter aircraft developed by the Swedish Royal Air Administration Aircraft Factory (FFVS) for the Swedish Air Force in 1940?
Answer: 2×8mm machine guns and 2×13.2mm machine guns

Reasoning:
1. Question Independence: true - Fully self-contained
2. Answer Specificity: true - Specific technical specification
3. Question Unambiguity: true - Clearly asks about a specific configuration
4. Question Answerability: true - Clear and answerable
5. Avoid Open-ended Questions: true - Requires a precise technical answer
6. Avoid Oversimplicity: true - I need to search historical knowledge to answer the question
7. Time Specificity: true - Contains specific year "1940"

All conditions are true, therefore the answer is \\boxed{{true}}

Example 3:
Question: What was one of the main German fighter aircraft models that the Swedish J22 fighter faced during its service in the 1940s?
Answer: FW 190

Reasoning:
1. Question Independence: true - Self-contained question
2. Answer Specificity: true - Specific aircraft model
3. Question Unambiguity: false - "one of" implies multiple correct answers, creating ambiguity
4. Question Answerability: true - Clear what is being asked
5. Avoid Open-ended Questions: true - Allows for a specific model name
6. Avoid Oversimplicity: true - I need to search historical knowledge to answer the question
7. Time Specificity: true - Contains specific decade "1940s"

There is a false condition, therefore the answer is \\boxed{{false}}

Example 4:
Question: In what year was the latest version of the annual report template file for the Cooperative Innovation High School (CIHS) in North Carolina released?
Answer: 2025

Reasoning:
1. Question Independence: true - Self-contained question
2. Answer Specificity: true - Specific year
3. Question Unambiguity: true - Clear what is being asked
4. Question Answerability: true - The question is clearly stated
5. Avoid Open-ended Questions: true - Requires a specific year
6. Avoid Oversimplicity: true - I need to search about CIHS to answer the question
7. Time Specificity: false - Contains "latest" which is a vague temporal reference that changes over time

There is a false condition, therefore the answer is \\boxed{{false}}

Example 5:
Question: In June 1937, in a collective school in Valencia, Spain, how did a teacher who had studied at Barcelona's "Nature School" (La Farigola) use the natural environment in teaching?
Answer: Organized students to visit vegetable gardens and orange trees, observe plant growth and draw what they saw, using nature as a learning object

Reasoning:
1. Question Independence: true - Self-contained question
2. Answer Specificity: false - Descriptive explanation rather than a specific fact
3. Question Unambiguity: true - Clear what is being asked
4. Question Answerability: true - The question is clearly stated
5. Avoid Open-ended Questions: false - Uses "how" which invites an explanatory response
6. Avoid Oversimplicity: true - I need to search to answer the question
7. Time Specificity: true - Contains specific time "June 1937"

There are multiple false conditions, therefore the answer is \\boxed{{false}}

Example 6:
Question: What was Frances Tiafoe's career tour-level finals record after his loss in the final of the 2025 Houston Men's Clay Court Championship?
Answer: 3 wins, 7 losses

Reasoning:
1. Question Independence: true - Self-contained question
2. Answer Specificity: true - Specific win-loss record
3. Question Unambiguity: true - Clear what is being asked
4. Question Answerability: true - The question is clearly stated
5. Avoid Open-ended Questions: true - Requires a specific record
6. Avoid Oversimplicity: true - I need to search to answer the question
7. Time Specificity: true - Contains specific year "2025" and specific event as temporal anchor

All conditions are true, therefore the answer is \\boxed{{true}}

Example 7:
Question: What is Frances Tiafoe's current career tour-level finals record?
Answer: 3 wins, 7 losses

Reasoning:
1. Question Independence: true - Self-contained question
2. Answer Specificity: true - Specific win-loss record
3. Question Unambiguity: true - Clear what is being asked
4. Question Answerability: true - The question is clearly stated
5. Avoid Open-ended Questions: true - Requires a specific record
6. Avoid Oversimplicity: true - I need to search to answer the question
7. Time Specificity: false - Contains "current" which is a vague temporal reference

There is a false condition, therefore the answer is \\boxed{{false}}

---------------------
IMPORTANT:
If you think any question is indeed a choise question or a yes/no question, please answer false.
e.g.
Question: 在2007年9月15日发表于《Biological Psychiatry》第62卷第6期的论文中，研究者在使用卡比多巴处理大鼠脑片后，3,4-亚甲二氧基甲基苯丙胺（MDMA）诱导的放电抑制和膜超极化现象是否消失？
Answer: 消失
This is a yes/no question, therefore the answer is \\boxed{{false}}


Now evaluate the following QA pair:
Question: {question}
Answer: {answer}

Reasoning:"""


def load_qa_pairs_from_json(json_path: str) -> List[Dict]:
    """Load QA pairs from a single JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = []
        url = data.get('url', '')  # Extract URL from the JSON
        if 'qa_pairs' in data:
            for qa in data['qa_pairs']:
                if 'question' in qa and 'answer' in qa:
                    qa_pairs.append({
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'url': url,
                        'source_file': os.path.basename(json_path)
                    })
        return qa_pairs
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []


def deduplicate_qa_pairs(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Deduplicate QA pairs by question.
    If same question has different answers, keep both as 'answer' and 'another_answer'.
    """
    question_dict = {}
    
    for qa in qa_pairs:
        question = qa['question']
        answer = qa['answer']
        url = qa.get('url', '')
        
        if question not in question_dict:
            question_dict[question] = {
                'question': question,
                'answer': answer,
                'url': url,
                'source_file': qa['source_file']
            }
        else:
            # Same question, check if different answer
            if answer != question_dict[question]['answer']:
                # Add as another_answer if not already present
                if 'another_answer' not in question_dict[question]:
                    question_dict[question]['another_answer'] = answer
                elif answer != question_dict[question].get('another_answer'):
                    # If there's already another_answer and this is a third different answer,
                    # we could extend to a list, but for now just skip
                    pass
    
    return list(question_dict.values())


def call_llm_filter(qa_pair: Dict) -> bool:
    """
    Call LLM to evaluate QA pair quality.
    Returns True if passes all criteria, False otherwise.
    """
    prompt = FILTER_PROMPT.format(
        question=qa_pair['question'],
        answer=qa_pair['answer']
    )
    
    url = f"http://{LLM_HOST}:{LLM_PORT}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=600)
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content'].strip()
            
            # Extract answer from \boxed{true} or \boxed{false}
            import re
            boxed_match = re.search(r'\\boxed\{(true|false)\}', content, re.IGNORECASE)
            if boxed_match:
                return boxed_match.group(1).lower() == 'true'
            
            # Fallback: check if 'true' appears in the last line or conclusion
            lines = content.lower().split('\n')
            for line in reversed(lines):
                if 'true' in line or 'false' in line:
                    return 'true' in line
            
            return False
        return False
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return False


def process_single_qa(qa_pair: Dict) -> Tuple[Dict, bool]:
    """Process a single QA pair through the filter."""
    passed = call_llm_filter(qa_pair)
    return qa_pair, passed


def format_output(qa_pair: Dict) -> Dict:
    """Format QA pair for output."""
    output = {
        "source": "direct_information_seeking_datagen",
        "url": qa_pair.get('url', ''),
        "question": qa_pair['question'],
        "answer": qa_pair['answer']
    }
    
    if 'another_answer' in qa_pair:
        output['another_answer'] = qa_pair['another_answer']
    
    return output


def load_checkpoint(ckpt_path: str) -> set:
    """Load processed files from checkpoint."""
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_checkpoint(ckpt_path: str, processed_files: set):
    """Save processed files to checkpoint."""
    with open(ckpt_path, 'w', encoding='utf-8') as f:
        for file in sorted(processed_files):
            f.write(f"{file}\n")


def append_to_jsonl(filepath: str, data: Dict):
    """Append a JSON object to a JSONL file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def main():
    global LLM_HOST, LLM_PORT, LLM_API_KEY, LLM_MODEL

    parser = argparse.ArgumentParser(description='Filter QA pairs for quality')
    parser.add_argument('--input_dir', type=str, default='qa_outputs',
                        help='Input directory containing JSON files')
    parser.add_argument('--output_dir', type=str, default='output_filtered',
                        help='Output directory for filtered results')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads')
    parser.add_argument('--ckpt', type=str, default='filter_ckpt.txt',
                        help='Checkpoint file path')
    parser.add_argument('--llm-host', type=str, default=os.getenv('LLM_HOST', LLM_HOST),
                        help='OpenAI-compatible LLM host')
    parser.add_argument('--llm-port', type=int, default=int(os.getenv('LLM_PORT', str(LLM_PORT))),
                        help='OpenAI-compatible LLM port')
    parser.add_argument('--llm-api-key', type=str, default=os.getenv('LLM_API_KEY', ''),
                        help='LLM API key. Defaults to LLM_API_KEY.')
    parser.add_argument('--llm-model', type=str, default=os.getenv('LLM_MODEL', ''),
                        help='LLM model name/path. Defaults to LLM_MODEL.')
    
    args = parser.parse_args()
    if not args.llm_api_key:
        parser.error('--llm-api-key is required, or set LLM_API_KEY.')
    if not args.llm_model:
        parser.error('--llm-model is required, or set LLM_MODEL.')

    LLM_HOST = args.llm_host
    LLM_PORT = args.llm_port
    LLM_API_KEY = args.llm_api_key
    LLM_MODEL = args.llm_model
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    positive_path = os.path.join(args.output_dir, 'positive.jsonl')
    negative_path = os.path.join(args.output_dir, 'negative.jsonl')
    
    # Load checkpoint
    processed_files = load_checkpoint(args.ckpt)
    print(f"Loaded checkpoint: {len(processed_files)} files already processed")
    
    # Get all JSON files
    json_files = list(Path(args.input_dir).glob('*.json'))
    json_files = [f for f in json_files if f.name not in processed_files]
    
    print(f"Found {len(json_files)} JSON files to process")
    
    if not json_files:
        print("No new files to process!")
        return
    
    # Step 1: Load all QA pairs
    print("\nStep 1: Loading QA pairs from JSON files...")
    all_qa_pairs = []
    for json_file in tqdm(json_files, desc="Loading files"):
        qa_pairs = load_qa_pairs_from_json(str(json_file))
        all_qa_pairs.extend(qa_pairs)
    
    print(f"Loaded {len(all_qa_pairs)} QA pairs")
    
    # Step 2: Deduplicate
    print("\nStep 2: Deduplicating QA pairs...")
    unique_qa_pairs = deduplicate_qa_pairs(all_qa_pairs)
    print(f"After deduplication: {len(unique_qa_pairs)} unique QA pairs")
    
    # Step 3: Pre-filter by answer length
    print("\nStep 3: Pre-filtering by answer length (max 30 chars)...")
    length_filtered_qa_pairs = [qa for qa in unique_qa_pairs if len(qa['answer']) <= 20]
    filtered_out_count = len(unique_qa_pairs) - len(length_filtered_qa_pairs)
    print(f"Filtered out {filtered_out_count} QA pairs with answer length > 20 chars")
    print(f"Remaining: {len(length_filtered_qa_pairs)} QA pairs")
    
    # Step 4: Filter with LLM
    print(f"\nStep 4: Filtering QA pairs with {args.workers} workers...")
    positive_count = 0
    negative_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_qa, qa): qa for qa in length_filtered_qa_pairs}
        
        with tqdm(total=len(length_filtered_qa_pairs), desc="Filtering") as pbar:
            for future in as_completed(futures):
                try:
                    qa_pair, passed = future.result()
                    output_data = format_output(qa_pair)
                    
                    if passed:
                        append_to_jsonl(positive_path, output_data)
                        positive_count += 1
                    else:
                        append_to_jsonl(negative_path, output_data)
                        negative_count += 1
                    
                except Exception as e:
                    print(f"\nError processing QA pair: {e}")
                
                pbar.update(1)
                pbar.set_postfix({'positive': positive_count, 'negative': negative_count})
    
    # Update checkpoint
    for json_file in json_files:
        processed_files.add(json_file.name)
    save_checkpoint(args.ckpt, processed_files)
    
    print(f"\n{'='*60}")
    print(f"Filtering complete!")
    print(f"Total loaded: {len(all_qa_pairs)}")
    print(f"After deduplication: {len(unique_qa_pairs)}")
    print(f"After length filtering: {len(length_filtered_qa_pairs)}")
    print(f"Passed LLM filter (positive): {positive_count}")
    print(f"Failed LLM filter (negative): {negative_count}")
    print(f"Positive rate (of length-filtered): {positive_count/len(length_filtered_qa_pairs)*100:.2f}%")
    print(f"{'='*60}")
    print(f"Results saved to:")
    print(f"  - {positive_path}")
    print(f"  - {negative_path}")
    print(f"Checkpoint saved to: {args.ckpt}")


if __name__ == '__main__':
    main()
