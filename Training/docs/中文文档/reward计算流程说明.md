# Reward 计算流程说明

## 一、整体架构

```
配置文件 (qwen3_agentloop.sh)
    ↓
主训练脚本 (main_ppo.py)
    ↓
加载 custom_reward_function (reward.py)
    ↓
创建 BatchRewardManager (batch.py)
    ↓
调用 compute_score_batch (llm_judge_async.py)
    ↓
返回 reward scores
```

## 二、详细流程

### 1. 配置加载 (qwen3_agentloop.sh)

```bash
custom_reward_function.path=verl/utils/reward_score/llm_judge_async.py
custom_reward_function.name=compute_score_batch
reward_model.reward_manager=batch
```

### 2. 函数加载 (verl/trainer/ppo/reward.py)

- `get_custom_reward_fn(config)` 动态加载外部reward函数
- 从配置的 path 和 name 导入 `compute_score_batch` 函数

### 3. Reward Manager 创建 (verl/trainer/ppo/reward.py)

```python
reward_manager_cls = get_reward_manager_cls("batch")  # BatchRewardManager
reward_manager = reward_manager_cls(
    tokenizer=tokenizer,
    num_examine=num_examine,
    compute_score=compute_score_batch,  # 你的自定义函数
    **reward_kwargs
)
```

### 4. 批量计算 Reward (verl/workers/reward_manager/batch.py)

**BatchRewardManager.__call__() 方法**:

```python
# 1. 解码所有响应
responses_str = [tokenizer.decode(response_ids[i][:valid_len]) for i in range(len(data))]

# 2. 收集所有必要信息
data_sources = data.non_tensor_batch["data_source"]
ground_truths = [item.non_tensor_batch["reward_model"]["ground_truth"] for item in data]
rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{}] * len(data))
extras = data.non_tensor_batch.get("extra_info", [{}] * len(data))

# 3. 将 rollout_reward_scores 添加到 extras
for i in range(len(data)):
    extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]

# 4. 调用你的 compute_score_batch
scores = self.compute_score(
    data_sources=data_sources,
    solution_strs=responses_str,
    ground_truths=ground_truths,
    extra_infos=extras,
    **self.reward_kwargs
)

# 5. 处理返回的 scores
for i, score in enumerate(scores):
    if isinstance(score, dict):
        reward = score["score"]
        # 收集额外信息（如 correct, method, reason 等）
        for key, value in score.items():
            reward_extra_info[key].append(value)
    else:
        reward = score
    
    reward_tensor[i, valid_response_length - 1] = reward
```

### 5. LLM Judge 计算 (verl/utils/reward_score/llm_judge_async.py)

**compute_score_batch() 函数流程**:

```python
def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """
    参数:
        data_sources: 数据来源列表
        solution_strs: 模型生成的完整响应（已解码）
        ground_truths: 标准答案字典列表，包含 {"target": [...]}
        extra_infos: 额外信息字典列表，包含 {"question": ..., "rollout_reward_scores": {...}}
        
    返回:
        List[Dict]: 每个样本返回一个字典
        {
            "score": 1.0 或 0.0,
            "correct": True/False,
            "method": "llm_judge" / "em_fallback" / "no_label" / "no_extraction",
            "pred_ans": 提取的答案,
            "reason": LLM判断的理由,
            "raw_response": LLM的原始响应
        }
    """
    
    # 步骤1: 从响应中提取答案
    extracted_answer = extract_solution(solution_str)  # 提取 <answer>...</answer>
    
    # 步骤2: 从 extra_infos 获取问题
    question = extra_infos[i].get("question", "")
    
    # 步骤3: 从 ground_truths 获取标准答案
    golden_answers = ground_truth.get("target", [])
    
    # 步骤4: 异步批量调用 LLM Judge
    judge_results = asyncio.run(judge_batch_async(batch_data))
    
    # 步骤5: 如果 LLM Judge 失败，回退到 EM (Exact Match) 匹配
    if not judge_result["success"]:
        is_correct = em_check(extracted_answer, golden_answers)
```

## 三、工具 Reward 的处理

### 工具不再计算 Reward

修改后的所有工具（google_search_tool.py, browse_tool.py 等）的 `calc_reward()` 方法都返回空列表：

```python
async def calc_reward(self, instance_id: str, **kwargs) -> list:
    # 不需要工具计算reward，返回空列表
    # 最终reward由 custom_reward_function (llm_judge_async.py) 计算
    return []
```

### Rollout 时收集工具 Rewards

在 `sglang_rollout.py` 中：

```python
# 收集所有工具的 reward（现在都是空列表）
tool_reward_scores = dict(tool_reward_scores)  # {"google_search": [], "browse": []}
all_rewards = {**tool_reward_scores, **{"user_turn_rewards": user_turn_rewards}}

# 这些会存储到 request.reward_scores 中
_req.finalize(self.processing_class, all_rewards, finish_reason_type)
```

### Reward Scores 的传递

```python
# rollout_reward_scores 被添加到 extra_infos
rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{}] * len(data))
for i in range(len(data)):
    extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]
    # 现在 rollout_reward_scores[i] = {"google_search": [], "browse": [], "user_turn_rewards": [...]}
```

## 四、最终 Reward 来源

**唯一的 Reward 来源**: `llm_judge_async.py` 的 `compute_score_batch()` 函数

- 基于模型生成的完整响应（包含所有工具调用和结果）
- 提取 `<answer>...</answer>` 标签内的最终答案
- 通过 LLM Judge 判断答案是否正确
- 失败时回退到精确匹配（EM）

工具的 `calc_reward()` 不再影响最终 reward，只是为了保持接口兼容性而存在。


