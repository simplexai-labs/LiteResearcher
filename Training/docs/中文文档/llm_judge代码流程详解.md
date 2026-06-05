# LLM Judge 代码流程详解

本文档详细解释 verl 中 LLM-as-Judge 的完整执行流程，包括异步机制的实现原理。

## 目录

1. [整体流程](#整体流程)
2. [Rollout阶段（生成+工具调用）](#rollout阶段生成工具调用)
3. [Reward计算阶段](#reward计算阶段)
4. [异步优化机制](#异步优化机制)
5. [时间线对比](#时间线对比)
6. [关键代码解析](#关键代码解析)

---

## 整体流程

verl 的训练循环包含以下主要阶段（基于 `verl/trainer/ppo/ray_trainer.py:fit()`）：

```
1. Generate (Rollout)    ← 生成+工具调用
2. Compute Reward        ← Judge评分（可异步）
3. Compute Old Log Prob  ← 重新计算训练策略的log_prob
4. Compute Ref Log Prob  ← 参考策略的log_prob（如果启用）
5. Compute Values        ← Critic值函数（如果启用）
6. Compute Advantage     ← 计算优势函数
7. Update Critic         ← 更新Critic网络
8. Update Actor          ← 更新Actor网络
```

**关键发现**：启用 `launch_reward_fn_async=True` 后，步骤2-5可以并行执行！

---

## Rollout阶段（生成+工具调用）

### 代码位置
`ray_trainer.py:1054-1061` → `sglang_rollout.py:_async_rollout_a_request()`

### 执行流程

```python
# ray_trainer.py 第1054行
with marked_timer("gen", timing_raw, color="red"):
    if not self.async_rollout_mode:
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
```

在 `generate_sequences()` 内部（`sglang_rollout.py`）：

#### 1. Multi-turn对话循环
```python
# sglang_rollout.py 第849行
while current_turns < self.config.multi_turn.max_assistant_turns:
    if _req.state == AsyncRolloutRequestStateEnum.PENDING:
        # 生成第一轮响应
        await self._handle_pending_state(_req)
        _req.state = AsyncRolloutRequestStateEnum.RUNNING

    elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
        # 执行工具调用（实时、同步）
        if _req.messages[-1].tool_calls is not None:
            parsed_tool_calls = _req.messages[-1].tool_calls
            # 并行执行所有工具调用
            tool_call_results = await asyncio.gather(
                *[self._tool_map[tool_call.function.name].execute(
                    _req.request_id,
                    tool_call.function.arguments,
                    **_req.tools_kwargs.get(tool_call.function.name, {}).get("execute_kwargs", {})
                ) for tool_call in parsed_tool_calls]
            )  # 第858-867行
            # 将工具结果添加到对话
            _req.add_tool_response_messages(self.processing_class, [resp for resp, _, _ in tool_call_results])
            _req.state = AsyncRolloutRequestStateEnum.RUNNING

    elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
        # LLM继续生成
        output = await self._handle_engine_call(_req, request_sampling_params, image_data=image_data)
        content = output["text"]

        # 检测是否需要调用工具
        if self._function_call_parser and self._function_call_parser.has_tool_call(content):
            _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
            # 解析工具调用
            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
            _req.add_assistant_message(self.processing_class, content=normed_content, tool_calls=parsed_tool_calls)
        else:
            # 生成结束
            _req.add_assistant_message(self.processing_class, content=content)
            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
            break
```

#### 2. 工具奖励计算（rollout内部）
```python
# sglang_rollout.py 第1018-1031行
# 计算每个工具的reward（例如是否成功调用）
async def calc_reward_and_release_fn(name: str, tool: BaseTool):
    reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
    await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
    return name, reward

tool_reward_tasks = []
for name in _req.tools_kwargs.keys():
    tool = self._tool_map[name]
    tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
tool_reward_scores = dict(tool_reward_scores)

# 工具reward存储在batch中，但这不是最终的训练reward
_req.finalize(self.processing_class, all_rewards, finish_reason_type)
```

**注意**：这里的 `tool.calc_reward()` 是工具本身的reward（例如搜索是否成功），**不是**最终答案的质量评分。

### Rollout返回结果

```python
# 返回的 gen_batch_output 包含：
{
    "prompts": [...],           # 原始用户问题
    "responses": [...],         # 完整响应（含工具调用结果）
    "input_ids": [...],         # 完整对话token ids
    "attention_mask": [...],    # 注意力掩码
    "response_mask": [...],     # 1=LLM生成，0=工具响应/padding
    "messages": [...],          # 完整对话历史
}
```

---

## Reward计算阶段

### 代码位置
`ray_trainer.py:1100-1109` 和 `1144-1149`

### 同步 vs 异步模式

#### 模式1：同步模式（`launch_reward_fn_async=False`）

```python
# ray_trainer.py 第1100-1109行
with marked_timer("reward", timing_raw, color="yellow"):
    # 计算reward model score（如果启用RM）
    if self.use_rm and "rm_scores" not in batch.batch.keys():
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)

    # 同步调用reward函数（阻塞直到完成）
    if self.config.reward_model.launch_reward_fn_async:
        future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
    else:
        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
        # ↑ 主进程直接计算，阻塞约1-5秒（取决于batch size和judge模型）
```

**流程**：
```
[Rollout完成] → [等待Judge计算完成] → [继续后续计算]
                     ↑ 阻塞在这里
```

#### 模式2：异步模式（`launch_reward_fn_async=True`）

```python
# 第1107行：立即启动Ray remote task，不阻塞
future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
# ↑ 返回一个future对象，Judge计算在Ray worker上异步进行

# 第1112-1142行：主进程继续执行其他计算（与Judge并行）
with marked_timer("old_log_prob", timing_raw, color="blue"):
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)  # 计算策略log_prob
    batch = batch.union(old_log_prob)

if self.use_reference_policy:
    with marked_timer("ref", timing_raw, color="olive"):
        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)  # 计算参考策略log_prob
        batch = batch.union(ref_log_prob)

if self.use_critic:
    with marked_timer("values", timing_raw, color="cyan"):
        values = self.critic_wg.compute_values(batch)  # 计算值函数
        batch = batch.union(values)

# 第1144-1149行：在计算advantage之前，等待Judge结果
with marked_timer("adv", timing_raw, color="brown"):
    if self.config.reward_model.launch_reward_fn_async:
        # 阻塞直到Judge计算完成
        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
        # ↑ 如果Judge已经完成，立即返回；否则等待
    batch.batch["token_level_scores"] = reward_tensor

    # 继续计算advantage（需要reward结果）
    batch = compute_advantage(batch, ...)
```

**流程**：
```
[Rollout完成]
    ├─→ [启动Judge（异步）] ──┐
    └─→ [计算log_prob]        │
        [计算ref_log_prob]    │  并行执行
        [计算values]          │
        [等待Judge完成] ←─────┘
        [计算advantage]
```

### compute_reward_async 实现

```python
# verl/trainer/ppo/reward.py 第176-192行
@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
    """
    在单独的Ray worker上运行reward计算
    - 使用1个CPU（不占用GPU）
    - 运行在独立进程中，不阻塞主trainer
    """
    if reward_fn is None:
        # 旧版API，已废弃
        reward_fn = load_reward_manager(config, tokenizer, ...)

    # 调用实际的reward计算函数
    return compute_reward(data, reward_fn)
```

### compute_reward 的实际执行

```python
# verl/trainer/ppo/reward.py 第153-173行
def compute_reward(data: DataProto, reward_fn):
    """
    对batch中的每个样本计算reward
    """
    try:
        # 调用自定义reward函数（例如 llm_judge_vllm.py 中的 compute_score）
        reward_result = reward_fn(data)

        if isinstance(reward_result, dict):
            reward_tensor = reward_result["reward"]
            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
        else:
            reward_tensor = reward_result
            reward_extra_infos_dict = {}
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict
```

#### LLM Judge 具体实现（`llm_judge_vllm.py`）

```python
# verl/utils/reward_score/llm_judge_vllm.py
def compute_score(data: DataProto):
    """
    使用LLM Judge评估答案质量
    """
    # 1. 从data中提取信息
    solution_str = data.non_tensor_batch.get("solution_str", ...)
    ground_truth = data.non_tensor_batch.get("ground_truth", ...)
    question = data.non_tensor_batch.get("extra_info", {}).get("question", ...)

    # 2. 初始化Judge客户端（使用vLLM OpenAI API）
    judge_client = OpenAI(
        base_url=os.environ.get("VLLM_JUDGE_API_BASE", "http://localhost:8000"),
        api_key="EMPTY",
    )

    # 3. 批量调用Judge模型
    scores = []
    for i in range(len(solution_str)):
        # 构造prompt
        prompt = f"""Given a Question and its Golden Answer, verify whether the Predicted Answer is correct.
Question: {question[i]}
Golden Answer: {ground_truth[i]}
Predicted Answer: {solution_str[i]}
Answer (True/False):"""

        # 调用Judge API
        response = judge_client.chat.completions.create(
            model=os.environ.get("VLLM_JUDGE_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )

        # 解析Judge输出
        judge_output = response.choices[0].message.content.strip().lower()
        score = 1.0 if "true" in judge_output else 0.0
        scores.append(score)

    # 4. 转换为token-level reward tensor
    reward_tensor = torch.tensor(scores, dtype=torch.float32)
    reward_tensor = reward_tensor.unsqueeze(-1).expand(-1, max_response_length)

    return {"reward": reward_tensor, "reward_extra_info": {"judge_outputs": judge_outputs}}
```

---

## 异步优化机制

### 为什么需要异步？

在大batch训练时，reward计算是瓶颈：

| 操作 | 时间消耗（batch_size=512） |
|------|--------------------------|
| Rollout（生成+工具调用） | ~30-60秒 |
| **Judge reward计算** | **~5-15秒** ← 瓶颈 |
| Compute log_prob | ~3-5秒 |
| Compute ref_log_prob | ~2-4秒 |
| Compute values | ~2-3秒 |
| Compute advantage | ~0.1秒 |
| Update actor | ~10-20秒 |

**问题**：如果同步执行，Judge计算时CPU/GPU空闲等待。

### 异步优化效果

#### 同步模式时间线

```
Time →
0s          30s         35s       40s     43s    46s    47s         67s
├───────────┼───────────┼─────────┼───────┼──────┼──────┼───────────┤
│  Rollout  │  Judge    │ log_prob│  ref  │ vals │ adv  │  Update   │
│           │  (wait)   │         │       │      │      │           │
└───────────┴───────────┴─────────┴───────┴──────┴──────┴───────────┘
总耗时：67秒
```

#### 异步模式时间线

```
Time →
0s          30s                    40s     43s    46s    47s         67s
├───────────┼──────────────────────┼───────┼──────┼──────┼───────────┤
│  Rollout  │  log_prob + ref + vals│ adv  │      │      │  Update   │
│           │  (并行)              │      │      │      │           │
└───────────┤                      ├──────┴──────┴──────┴───────────┘
            │  Judge (Ray worker)  │
            │  (并行执行)          │
            └──────────────────────┘
总耗时：57秒 (节省10秒，约15%提速)
```

**关键点**：
- Judge在Ray worker上异步执行（使用1个CPU）
- 主进程同时计算log_prob、ref_log_prob、values
- 在需要reward时才等待Judge完成（第1148行）
- 如果log_prob等计算时间 > Judge计算时间，Judge结果已ready，无需等待

### 配置参数

```yaml
# 在配置文件中启用异步
reward_model:
  enable: False
  custom_reward_function:
    path: verl/utils/reward_score/llm_judge_vllm.py
    name: compute_score
  launch_reward_fn_async: True  # ← 关键配置
```

```bash
# 或命令行覆盖
python3 -m verl.trainer.main_ppo \
    reward_model.custom_reward_function.path=verl/utils/reward_score/llm_judge_vllm.py \
    reward_model.launch_reward_fn_async=True
```

---

## 时间线对比

### 完整训练步骤时间分解

假设 batch_size=512, rollout.n=4（每个prompt采样4次）

| 步骤 | 同步模式 | 异步模式 | 说明 |
|-----|---------|---------|------|
| 1. Generate | 30s | 30s | 生成+工具调用 |
| 2. Reward (Judge) | **10s** | **0s** | 异步模式立即返回future |
| 3. Old log_prob | 4s | 4s | 重新计算策略log_prob |
| 4. Ref log_prob | 3s | 3s | 参考策略log_prob |
| 5. Values | 2s | 2s | Critic值函数 |
| 6. Wait Judge | 0s | **1s** | 异步模式：前面计算9s，Judge需10s，等1s |
| 7. Advantage | 0.1s | 0.1s | 计算优势函数 |
| 8. Update Critic | 5s | 5s | 更新Critic网络 |
| 9. Update Actor | 15s | 15s | 更新Actor网络 |
| **总计** | **69.1s** | **60.1s** | **节省9秒（13%提速）** |

### 什么时候异步最有效？

**最佳场景**：Judge时间 ≈ (log_prob + ref_log_prob + values)时间
- 如果Judge很慢（>15s），异步优势有限
- 如果Judge很快（<5s），异步优势明显

**建议**：
- 使用较小的Judge模型（Qwen2.5-3B而非7B）
- 增加Judge服务的并发能力
- 对于小batch（<128），同步模式可能更简单

---

## 关键代码解析

### 1. Rollout中的工具调用

**文件**：`verl/workers/rollout/sglang_rollout/sglang_rollout.py`

```python
# 第858-867行：并行执行工具调用
tool_call_results = await asyncio.gather(
    *[
        self._tool_map[tool_call.function.name].execute(
            _req.request_id,
            tool_call.function.arguments,
            **_req.tools_kwargs.get(tool_call.function.name, {}).get("execute_kwargs", {})
        )
        for tool_call in parsed_tool_calls
    ]
)
```

**工具实现示例**（`verl/tools/search_tool.py`）：

```python
async def execute(self, request_id: str, arguments: str, **kwargs) -> tuple[str, float, dict]:
    """
    执行搜索工具

    Returns:
        (content, reward, metrics):
            - content: 工具返回的文本结果
            - reward: 工具执行的reward（0=失败，1=成功）
            - metrics: 额外指标
    """
    try:
        # 解析参数
        args = json.loads(arguments)
        query = args.get("query", "")

        # 调用搜索API
        results = await self._call_search_api(query)

        # 格式化结果
        content = self._format_search_results(results)
        reward = 1.0  # 成功调用
        metrics = {"search_results_count": len(results)}

        return content, reward, metrics
    except Exception as e:
        # 工具调用失败
        return f"Error: {str(e)}", 0.0, {"error": str(e)}
```

### 2. 异步Reward计算的触发点

**文件**：`verl/trainer/ppo/ray_trainer.py`

```python
# 第1106-1109行：触发异步计算
if self.config.reward_model.launch_reward_fn_async:
    # 启动Ray remote task，立即返回future
    future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
    # ↑ 不阻塞，继续执行后续代码
else:
    # 同步计算，阻塞直到完成
    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
```

### 3. 异步结果的等待点

```python
# 第1144-1149行：等待Judge结果
with marked_timer("adv", timing_raw, color="brown"):
    if self.config.reward_model.launch_reward_fn_async:
        # 阻塞直到future完成（如果尚未完成）
        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
        # ↑ ray.get() 是同步等待，类似 await
    batch.batch["token_level_scores"] = reward_tensor

    # 必须有reward结果才能计算advantage
    batch = compute_advantage(
        batch,
        adv_estimator=self.config.algorithm.adv_estimator,
        gamma=self.config.algorithm.gamma,
        lam=self.config.algorithm.lam,
        num_repeat=self.config.actor_rollout_ref.rollout.n,
        ...
    )
```

**为什么在这里等待？**
- Advantage计算需要reward值
- 之前的log_prob、ref_log_prob、values计算不需要reward
- 这样可以最大化并行时间

### 4. Ray Remote的工作原理

```python
# verl/trainer/ppo/reward.py 第176-192行
@ray.remote(num_cpus=1)  # ← 声明Ray actor，使用1个CPU
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
    """
    这个函数会在Ray集群的某个worker上执行
    - 使用单独的Python进程
    - 不占用主训练进程的资源
    - 可以并行执行多个reward计算任务
    """
    return compute_reward(data, reward_fn)

# 调用方式：
future = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
# ↑ .remote() 表示异步调用，返回ObjectRef（类似future）

# 等待结果：
result = ray.get(future)
# ↑ ray.get() 阻塞直到结果ready
```

---

## 异步机制的关键问题解答

### Q1: 启用 `launch_reward_fn_async=True` 后，judge和后续流程是异步的吗？

**A: 是部分异步的。**

- **异步部分**：Judge计算与 old_log_prob、ref_log_prob、values 计算是并行的
- **同步等待点**：在计算advantage之前（第1148行），必须等待Judge完成

**时序图**：
```
主进程                          Ray Worker (Judge)
│                                    │
├─ future_reward = ...remote()  ────┤ 启动Judge计算
│  (立即返回，不等待)                 │
│                                    ├─ decode response
├─ compute_log_prob()                ├─ extract answer
│  (与Judge并行)                     ├─ call vLLM API
│                                    │  (HTTP请求)
├─ compute_ref_log_prob()            │
│  (与Judge并行)                     ├─ parse True/False
│                                    │
├─ compute_values()                  ├─ construct reward tensor
│  (与Judge并行)                     │
│                                    └─ return result
├─ ray.get(future_reward) ───────────┐
│  (阻塞等待)                        │
│  ← reward结果 ←─────────────────────┘
│
├─ compute_advantage()
│  (需要reward，所以必须等待)
```

### Q2: 什么时候会 await reward 计算？

**A: 在计算advantage之前（`ray_trainer.py:1148`）**

```python
# 第1144行：进入advantage计算阶段
with marked_timer("adv", timing_raw, color="brown"):
    # 第1147-1148行：等待Judge完成
    if self.config.reward_model.launch_reward_fn_async:
        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
        # ↑ 这里是唯一的阻塞等待点
        # 如果Judge已经完成，立即返回（耗时<1ms）
        # 如果Judge还在计算，等待完成（耗时取决于剩余时间）

    # 第1149行：使用reward结果
    batch.batch["token_level_scores"] = reward_tensor

    # 第1175行：计算advantage（必须有reward）
    batch = compute_advantage(batch, ...)
```

**为什么不在更早的地方等待？**
- log_prob、ref_log_prob、values 的计算都**不依赖**reward
- 推迟等待可以最大化并行时间
- Advantage计算是第一个**需要**reward的地方

### Q3: Judge计算是真正的异步吗？

**A: 是的，使用Ray的异步机制。**

```python
# ray.remote 装饰器让函数在单独的进程中执行
@ray.remote(num_cpus=1)
def compute_reward_async(data, reward_fn):
    return compute_reward(data, reward_fn)

# 调用方式
future = compute_reward_async.remote(...)  # 立即返回ObjectRef
result = ray.get(future)                    # 阻塞等待结果
```

**Ray的工作原理**：
1. `.remote()` 将任务提交到Ray调度器
2. Ray在某个worker进程中执行任务
3. 结果存储在Ray的对象存储中
4. `ray.get()` 从对象存储获取结果（如果未ready则等待）

---

## 调试和监控

### 查看时间消耗

训练时会输出timing信息：

```
Step 1:
  gen: 32.45s
  reward: 0.02s (async trigger)
  old_log_prob: 4.12s
  ref: 3.01s
  values: 2.34s
  adv: 1.23s (包含等待Judge的时间)
  update_critic: 5.67s
  update_actor: 15.89s
Total: 64.73s
```

**关键指标**：
- `reward` 很小（<0.1s）：异步启动成功
- `adv` 包含等待时间：如果Judge很慢，这里会增加

### 检查Judge服务状态

```bash
# 测试Judge API
curl http://localhost:8000/v1/models

# 查看Judge服务日志
# 应该看到周期性的请求（每个training step一次）
```

### 常见问题排查

#### 问题1：异步模式下reward全是0

**原因**：Judge服务未启动或配置错误

**排查**：
```python
# 在主程序中测试Judge调用
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("VLLM_JUDGE_API_BASE"),
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model=os.environ.get("VLLM_JUDGE_MODEL"),
    messages=[{"role": "user", "content": "Is 3 equal to 3? Answer True or False."}],
    temperature=0.0,
    max_tokens=10,
)
print(response.choices[0].message.content)  # 应该输出 "True"
```

#### 问题2：训练速度没有提升

**原因**：Judge计算时间远大于log_prob等计算时间

**解决**：
- 使用更小的Judge模型
- 增加Judge服务的GPU/TP并行度
- 优化Judge prompt长度

#### 问题3：Ray worker OOM

**原因**：batch太大，序列化的data对象占用内存过多

**解决**：
- 减小batch_size
- 减小max_response_length
- 增加Ray worker的内存限制

---

## 总结

### 核心要点

1. **工具调用在Rollout内部完成**
   - 与LLM生成交织进行
   - 使用asyncio并行执行多个工具
   - 返回完整对话（含工具响应）

2. **Reward计算是独立阶段**
   - 在Rollout完成后执行
   - 评估最终答案质量
   - 可以异步执行以节省时间

3. **异步机制利用Ray**
   - 使用 `@ray.remote` 在独立进程执行Judge
   - 主进程继续计算log_prob等
   - 在计算advantage前等待Judge结果

4. **性能优化建议**
   - 启用 `launch_reward_fn_async=True`
   - 使用较小的Judge模型（3B）
   - 确保Judge服务稳定运行
   - 监控timing指标调优

### 配置checklist

```bash
# 1. 启动Judge服务
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8000

# 2. 设置环境变量
export VLLM_JUDGE_API_BASE="http://localhost:8000"
export VLLM_JUDGE_MODEL="Qwen/Qwen2.5-3B-Instruct"

# 3. 训练配置
python3 -m verl.trainer.main_ppo \
    reward_model.custom_reward_function.path=verl/utils/reward_score/llm_judge_vllm.py \
    reward_model.launch_reward_fn_async=True \
    # ... 其他配置
```

---

**相关文档**：
- [llm_judge使用指南.md](./llm_judge使用指南.md) - 用户使用指南
- [reward奖励计算说明.md](./reward奖励计算说明.md) - Reward系统概述
