"""LiteResearcher ReAct Agent - multi-turn reasoning with search and visit tools."""

import ast
import json
import os
import re
import time
import asyncio
import random
import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import json5
except ImportError:
    json5 = None

import aiohttp
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer

from src.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_SIMPLE, JUDGE_PROMPT, JUDGE_PROMPT_XBENCH

# FastAPI service URLs
SEARCH_SERVER_URL = f"http://127.0.0.1:{os.environ.get('SEARCH_SERVER_PORT', '8001')}"
BROWSER_SERVER_URL = f"http://127.0.0.1:{os.environ.get('BROWSER_SERVER_PORT', '8002')}"
TOOL_SERVER_TIMEOUT = int(os.environ.get("TOOL_SERVER_TIMEOUT", 300))


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


class ReActAgent:
    """Multi-turn ReAct agent that calls search/visit tools via HTTP services."""

    def __init__(self, llm=None, **kwargs):
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_loads(payload: str) -> Optional[Dict]:
        if json5 is not None:
            try:
                return json5.loads(payload)
            except Exception:
                pass
        try:
            return json.loads(payload)
        except Exception:
            pass
        try:
            return ast.literal_eval(payload)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Answer formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_answer_markup(answer: str) -> str:
        if not isinstance(answer, str):
            return ""
        cleaned = answer.strip()
        if not cleaned:
            return ""
        lower = cleaned.lower()
        if lower.startswith("<answer") and "</answer>" in lower:
            return cleaned
        if "\n" in cleaned:
            return f"<answer>\n{cleaned}\n</answer>"
        return f"<answer>{cleaned}</answer>"

    @staticmethod
    def _extract_tool_interactions(messages: List[Dict]) -> List[Dict]:
        interactions: List[Dict] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if "<tool_call>" not in content or "</tool_call>" not in content:
                continue
            tool_call_raw = content.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
            parsed_payload = ReActAgent._safe_loads(tool_call_raw)
            tool_response = None
            if idx + 1 < len(messages):
                next_msg = messages[idx + 1]
                if (
                    next_msg.get("role") == "user"
                    and "<tool_response>" in next_msg.get("content", "")
                    and "</tool_response>" in next_msg.get("content", "")
                ):
                    tool_response = (
                        next_msg["content"].split("<tool_response>", 1)[1]
                        .split("</tool_response>", 1)[0]
                        .strip()
                    )
            interactions.append({
                "tool_call_raw": tool_call_raw,
                "tool_call": parsed_payload,
                "response": tool_response,
            })
        return interactions

    # ------------------------------------------------------------------
    # Judge
    # ------------------------------------------------------------------

    def judge_answer(self, question: str, reference: str, prediction: str, data_path: str = "") -> Dict[str, Any]:
        summary_enabled = os.environ.get("SUMMARY_ENABLE", "true").lower() != "false"
        is_xbench = "xbench" in data_path.lower() if data_path else False

        visit_api_base = os.environ.get("VISIT_API_BASE", "").strip()
        if visit_api_base:
            base_url = visit_api_base
        else:
            summary_ports_str = os.environ.get("SUMMARY_PORTS", "")
            if summary_ports_str:
                ports = []
                for token in summary_ports_str.replace(',', ' ').replace(';', ' ').split():
                    try:
                        ports.append(int(token))
                    except ValueError:
                        pass
                base_url = f"http://127.0.0.1:{ports[0]}/v1" if ports else ""
            else:
                base_url = os.environ.get("API_BASE", "").strip()

        model_name = os.environ.get("SUMMARY_MODEL_NAME") or os.environ.get("SUMMARY_MODEL_PATH", "")

        if not summary_enabled:
            return {"status": "skipped", "reason": "disabled", "correct": None,
                    "verdict": None, "reference": reference, "prediction": prediction}
        if not base_url or not model_name:
            return {"status": "skipped", "reason": "not configured", "correct": None,
                    "verdict": None, "reference": reference, "prediction": prediction}

        api_key = os.environ.get("API_KEY", "EMPTY")
        timeout = float(os.environ.get("SUMMARY_SERVER_TIMEOUT", 300))

        if is_xbench:
            prompt_text = JUDGE_PROMPT_XBENCH.format(
                question=str(question).strip(), reference=str(reference).strip(),
                prediction=str(prediction).strip())
        else:
            prompt_text = JUDGE_PROMPT.format(
                question=str(question).strip(), reference=str(reference).strip(),
                prediction=str(prediction).strip())

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        try:
            max_tokens = 512 if is_xbench else 128
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.0, top_p=1.0, max_tokens=max_tokens,
            )
            raw_content = response.choices[0].message.content or "" if response.choices else ""

            if is_xbench:
                normalized = raw_content.strip()
                has_error = "结论" in normalized and "错误" in normalized.split("结论")[-1]
                has_correct = "结论" in normalized and "正确" in normalized.split("结论")[-1]
                is_correct = (not has_error and has_correct) if (has_error or has_correct) else ("正确" in normalized and "错误" not in normalized)
            else:
                normalized = raw_content.strip().lower()
                is_correct = "correct" in normalized and "incorrect" not in normalized

            return {"status": "ok", "correct": is_correct, "verdict": "CORRECT" if is_correct else "INCORRECT",
                    "raw": raw_content, "reference": reference, "prediction": prediction}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "correct": None,
                    "verdict": None, "reference": reference, "prediction": prediction}

    # ------------------------------------------------------------------
    # LLM server interaction
    # ------------------------------------------------------------------

    def call_server(self, msgs, planning_port, max_tries=3):
        api_key = os.environ.get("SGLANG_API_KEY", "EMPTY")
        api_base = os.environ.get("SGLANG_API_BASE")
        if not api_base:
            api_base = f"http://127.0.0.1:{planning_port}/v1"

        client = OpenAI(api_key=api_key, base_url=api_base, timeout=600.0)
        model_max_length = int(os.environ.get("MAIN_MAX_MODEL_LEN", 90000))

        try:
            prompt_tokens = self.count_tokens(msgs)
        except Exception:
            prompt_tokens = sum(len(str(msg.get("content", ""))) for msg in msgs) // 4

        available_tokens = model_max_length - prompt_tokens - 1000
        max_tokens = max(512, min(available_tokens, 10000))

        last_error = None
        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    frequency_penalty=self.llm_generate_cfg.get('frequency_penalty', 0.0),
                    logprobs=True,
                    max_tokens=max_tokens,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1),
                    extra_body={
                        "top_k": self.llm_generate_cfg.get('top_k', 20),
                        "min_p": self.llm_generate_cfg.get('min_p', 0.0),
                        "repetition_penalty": self.llm_generate_cfg.get('repetition_penalty', 1.0),
                    }
                )
                content = chat_response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
            except APIError as e:
                last_error = str(e)
                if "400" in last_error or "Bad Request" in last_error:
                    return f"LENGTH_LIMIT_ERROR: prompt={prompt_tokens}, limit={model_max_length}"
            except (APIConnectionError, APITimeoutError) as e:
                last_error = str(e)
            except Exception as e:
                last_error = str(e)

            if attempt < max_tries - 1:
                time.sleep(min(1 * (2 ** attempt) + random.uniform(0, 1), 30))

        return f"SERVER_ERROR: failed after {max_tries} attempts: {last_error}"

    def count_tokens(self, messages):
        tokenizer_path = os.environ.get("TOKEN_COUNT_MODEL_PATH", self.llm_local_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(full_prompt, return_tensors="pt")
        return len(tokens["input_ids"][0])

    @staticmethod
    def count_messages_tokens_with_template(messages: List[Dict], tokenizer_path: str = None) -> Dict[str, int]:
        import warnings
        if tokenizer_path is None:
            tokenizer_path = os.environ.get("TOKEN_COUNT_MODEL_PATH", "")
        result = {"total_tokens": 0, "system_tokens": 0, "user_tokens": 0,
                  "assistant_tokens": 0, "tokenizer_path": tokenizer_path}
        if not tokenizer_path:
            result["error"] = "TOKEN_COUNT_MODEL_PATH not set"
            return result
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                tokens = tokenizer(full_prompt, return_tensors="pt")
                result["total_tokens"] = len(tokens["input_ids"][0])
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if content:
                    msg_tokens = len(tokenizer.encode(content, add_special_tokens=False))
                    if role == "system":
                        result["system_tokens"] += msg_tokens
                    elif role == "user":
                        result["user_tokens"] += msg_tokens
                    elif role == "assistant":
                        result["assistant_tokens"] += msg_tokens
        except Exception as e:
            result["error"] = str(e)
        return result

    # ------------------------------------------------------------------
    # Tool calling via HTTP
    # ------------------------------------------------------------------

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        return asyncio.run(self._async_call_tool(tool_name, tool_args))

    async def _async_call_tool(self, tool_name: str, tool_args: dict) -> str:
        timeout = aiohttp.ClientTimeout(total=TOOL_SERVER_TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if tool_name == "search":
                    payload = {"query": tool_args.get("query", [])}
                    async with session.post(f"{SEARCH_SERVER_URL}/search", json=payload) as resp:
                        data = await resp.json()
                        return data["result"] if data.get("success") else f"[Search] Error: {data.get('error')}"
                elif tool_name == "visit":
                    payload = {"url": tool_args.get("url", ""), "goal": tool_args.get("goal", "")}
                    async with session.post(f"{BROWSER_SERVER_URL}/browse", json=payload) as resp:
                        data = await resp.json()
                        return data["result"] if data.get("success") else f"[Visit] Error: {data.get('error')}"
                else:
                    return f"Error: Tool {tool_name} not found"
        except asyncio.TimeoutError:
            return f"Error: Tool {tool_name} timeout after {TOOL_SERVER_TIMEOUT}s"
        except Exception as e:
            return f"Error: Tool {tool_name} service error: {str(e)}"

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_result(self, question, answer, messages, termination,
                      prediction=None, question_id=None, data_path="",
                      total_time=None, turn_times=None) -> Dict[str, Any]:
        final_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                final_message = str(msg.get("content", ""))
                break

        if prediction is not None:
            final_answer = prediction.strip() if isinstance(prediction, str) else ""
        else:
            final_answer = final_message.strip() if final_message else ""

        answer_text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        markup = self._format_answer_markup(final_answer)

        result: Dict[str, Any] = {
            "question": question, "answer": answer, "messages": messages,
            "prediction": final_answer, "prediction_tagged": markup,
            "termination": termination, "final_answer": final_answer,
            "final_answer_markup": markup, "final_message": final_message,
            "question_id": question_id,
        }
        if total_time is not None:
            result["total_time"] = total_time
        if turn_times is not None:
            result["turn_times"] = turn_times

        result["token_stats"] = self.count_messages_tokens_with_template(messages)
        result["tool_interactions"] = self._extract_tool_interactions(messages)

        if final_answer:
            result["judge"] = self.judge_answer(question, answer_text, final_answer, data_path)
        else:
            result["judge"] = {"status": "no_answer", "correct": False, "verdict": "INCORRECT",
                               "reference": answer_text, "prediction": final_answer}
        return result

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def _run(self, data, model, **kwargs) -> Dict[str, Any]:
        self.model = model
        item_payload = data.get("item", {})
        question_id = data.get("question_id") or item_payload.get("question_id") or item_payload.get("id")
        data_path = data.get("data_path", "")

        try:
            question = item_payload["question"]
        except Exception:
            raw_msg = ""
            msgs = item_payload.get("messages") or []
            if len(msgs) > 1 and isinstance(msgs[1], dict):
                raw_msg = msgs[1].get("content", "")
            elif msgs and isinstance(msgs[0], dict):
                raw_msg = msgs[0].get("content", "")
            question = raw_msg.split("User:", 1)[1].strip() if "User:" in raw_msg else raw_msg
        if not isinstance(question, str):
            question = str(question)

        start_time = time.time()
        planning_port = data["planning_port"]
        answer = item_payload.get("answer", "")

        model_max_ctx = int(os.environ.get("MAIN_MAX_MODEL_LEN", 90000))
        max_timeout = int(os.environ.get("MAX_TIMEOUT_SECONDS", 9000))
        max_calls = int(os.environ.get("MAX_LLM_CALL_PER_RUN", 100))

        system_prompt = SYSTEM_PROMPT + today_date()
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]

        calls_left = max_calls
        round_idx = 0
        termination = ""
        turn_times: List[Dict[str, Any]] = []

        while calls_left > 0:
            # Timeout check
            if time.time() - start_time > max_timeout:
                return self._build_result(question, answer, messages,
                    f"timeout_after_{max_timeout}s", "No answer found (timeout)",
                    question_id, data_path, time.time() - start_time, turn_times)

            round_idx += 1
            calls_left -= 1
            turn_start = time.time()

            content = self.call_server(messages, planning_port)
            llm_time = time.time() - turn_start

            if '<tool_response>' in content:
                content = content[:content.find('<tool_response>')]
            messages.append({"role": "assistant", "content": content.strip()})

            norm = content.strip().lower()

            # Fatal errors → terminate
            if norm.startswith("length_limit_error"):
                turn_times.append({"turn": round_idx, "llm_time": llm_time, "tool_time": 0.0,
                                   "total_time": time.time() - turn_start, "action": "length_error"})
                return self._build_result(question, answer, messages, "length_limit_exceeded",
                    "Context length exceeded", question_id, data_path, time.time() - start_time, turn_times)

            if "server_error:" in norm:
                turn_times.append({"turn": round_idx, "llm_time": llm_time, "tool_time": 0.0,
                                   "total_time": time.time() - turn_start, "action": "server_error"})
                return self._build_result(question, answer, messages, "server_error",
                    "Server error", question_id, data_path, time.time() - start_time, turn_times)

            # Tool call
            tool_time = 0.0
            if '<tool_call>' in content and '</tool_call>' in content:
                tc_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
                tc_start = time.time()
                try:
                    tc = self._safe_loads(tc_str)
                    result = self.custom_call_tool(tc.get('name', ''), tc.get('arguments', {}))
                except Exception:
                    result = 'Error: Invalid tool call JSON.'
                tool_time = time.time() - tc_start
                messages.append({"role": "user", "content": "<tool_response>\n" + result + "\n</tool_response>"})

            # Answer found
            if '<answer>' in content and '</answer>' in content:
                turn_times.append({"turn": round_idx, "llm_time": llm_time, "tool_time": tool_time,
                                   "total_time": time.time() - turn_start, "action": "answer"})
                termination = "answer"
                break

            # Max calls exceeded
            if calls_left <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            turn_times.append({"turn": round_idx, "llm_time": llm_time, "tool_time": tool_time,
                               "total_time": time.time() - turn_start,
                               "action": "tool_call" if '<tool_call>' in content else "thinking"})

            # Context length check → send reminder
            try:
                token_count = self.count_tokens(messages)
            except Exception:
                token_count = 0

            if token_count > model_max_ctx:
                messages[-1]['content'] = (
                    "You have now reached the maximum context length. "
                    "Stop making tool calls and provide your best answer in this format: "
                    "<think>your final thinking</think>\n<answer>your answer</answer>"
                )
                final_start = time.time()
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})

                if '<answer>' in content and '</answer>' in content:
                    prediction = content.split('<answer>')[1].split('</answer>')[0]
                    termination = 'answer_at_context_limit'
                else:
                    prediction = content.strip()
                    termination = 'context_limit_no_format'

                turn_times.append({"turn": round_idx + 1, "llm_time": time.time() - final_start,
                                   "tool_time": 0.0, "total_time": time.time() - final_start,
                                   "action": "context_limit_reminder"})
                return self._build_result(question, answer, messages, termination,
                    prediction, question_id, data_path, time.time() - start_time, turn_times)

        # Post-loop: extract answer (supports partial responses)
        last = messages[-1]['content'] if messages else ""
        if '<answer>' in last:
            parts = last.split('<answer>')
            if len(parts) > 1:
                a = parts[1]
                prediction = a.split('</answer>')[0] if '</answer>' in a else a.strip()
                termination = termination or 'answer'
            else:
                prediction = 'No answer found.'
                termination = 'answer_not_found'
        else:
            prediction = 'No answer found.'
            termination = 'exceed_max_turns' if calls_left == 0 else 'answer_not_found'

        return self._build_result(question, answer, messages, termination,
            prediction, question_id, data_path, time.time() - start_time, turn_times)
