# LLM-as-Judge Reward 使用指南

本文档介绍如何在verl中使用基于vLLM后端的LLM-as-Judge进行reward计算。

## 概述

LLM-as-Judge reward函数使用一个独立的LLM模型来评判生成的答案是否正确。相比规则based的Exact Match，LLM judge可以：
- 理解语义等价性（例如"3个"和"三"）
- 判断答案是否包含关键信息
- 处理多种表达方式

## verl的两种Reward计算方式

### 方式1: 默认配置（基于data_source路由）

这是verl的**默认方式**，也是searchR1等benchmark使用的方式。

**配置**：
```yaml
reward_model:
  enable: False  # 不使用模型based reward（如RM打分）
  custom_reward_function:
    path: null  # 默认为null，使用verl内置的default_compute_score
    name: compute_score
  launch_reward_fn_async: False  # 不启用异步
```

**工作原理**：
1. 当 `path: null` 时，verl使用 `verl/utils/reward_score/__init__.py` 中的 `default_compute_score()`
2. `default_compute_score()` 根据数据的 `data_source` 字段自动路由到不同的reward函数：
   - `data_source="searchR1_nq"` → Exact Match reward (`search_r1_like_qa_em.compute_score()`)
   - `data_source="gsm8k"` → 数学验证 (`gsm8k.compute_score()`)
   - `data_source="GAIA"` → Exact Match reward
   - 其他data_source → 对应的reward函数

**代码示例**：
```python
# verl/utils/reward_score/__init__.py 中的逻辑
def default_compute_score(data_source, solution_str, ground_truth, ...):
    if data_source in ["searchR1_nq", "searchR1_triviaqa", "GAIA", "GPQA", ...]:
        from . import search_r1_like_qa_em
        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    elif data_source == "gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function not implemented for {data_source}")
    return res
```

**使用场景**：
- 不同benchmark需要不同的reward计算逻辑
- 使用规则based reward（如Exact Match、代码执行、数学验证）
- 无需额外GPU资源
- 追求最快的训练速度

### 方式2: LLM-as-Judge（统一使用Judge模型评分）

**配置**：
```yaml
reward_model:
  enable: False  # 不使用模型based reward
  custom_reward_function:
    path: verl/utils/reward_score/llm_judge_vllm.py  # 指定judge文件
    name: compute_score
  launch_reward_fn_async: True  # 启用异步调用（推荐）
```

**工作原理**：
1. 当 `path` 不为 `null` 时，verl加载指定的Python文件
2. 调用该文件中的 `compute_score` 函数
3. **完全bypass** `default_compute_score()`，不再根据data_source路由
4. 所有数据统一使用LLM judge进行评分

**使用场景**：
- 需要语义理解的答案评判
- 答案格式多样，难以用规则匹配
- 有额外GPU资源运行judge模型
- 可以接受较慢的训练速度换取更准确的reward

### 两种方式的对比

| 特性 | 默认（data_source路由） | LLM-as-Judge |
|------|------------------------|--------------|
| **配置** | `path: null` | `path: verl/utils/reward_score/llm_judge_vllm.py` |
| **异步调用** | `False` | `True`（推荐） |
| **函数路由** | 根据data_source自动选择 | 全部使用judge |
| **速度** | 极快（~1000 samples/sec） | 较慢（~1-10 samples/sec） |
| **成本** | 无额外成本 | 需要额外GPU运行judge模型 |
| **准确性** | 严格匹配，可能误判 | 语义理解，更灵活 |
| **鲁棒性** | 对格式敏感 | 对格式鲁棒 |
| **GPU资源** | 只需训练GPU | 需要训练GPU + judge GPU |

### 如何选择

**使用默认方式（推荐大多数场景）**：
```bash
# 不需要额外配置，直接训练
python3 -m verl.trainer.main_ppo \
    --config-name search_multiturn_grpo \
    data.train_files=data/benchmarks_processed/all_benchmarks_test.parquet
    # 默认会根据data_source使用对应的reward函数
```

**切换到LLM-as-Judge**：
```bash
# 1. 先启动judge服务（见下文）
# 2. 配置环境变量
export VLLM_JUDGE_API_BASE="http://localhost:8000"
export VLLM_JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"

# 3. 训练时指定使用judge
python3 -m verl.trainer.main_ppo \
    --config-name search_multiturn_grpo \
    reward_model.custom_reward_function.path=verl/utils/reward_score/llm_judge_vllm.py \
    reward_model.launch_reward_fn_async=True \
    data.train_files=data/benchmarks_processed/all_benchmarks_test.parquet
```

**建议**：
- 如果答案格式固定（如数字、短语），优先使用**默认方式**（Exact Match等）
- 如果答案是长文本、需要语义判断，使用**LLM-as-Judge**
- 可以先用默认方式快速训练，再用LLM-as-Judge精调

## 架构

```
Training Process          Judge Backend
     │                         │
     ├─ Generate Answer        │
     ├─ Extract <answer>       │
     ├─ Call Judge ───────────→ vLLM Server
     │                         │  (Judge Model)
     ├─ Get Score ←────────────┘
     └─ Compute Reward
```

## 部署Judge后端

### 1. 启动vLLM服务器

首先需要启动一个独立的vLLM服务器运行judge模型：

```bash
# 在单独的GPU上启动judge服务
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.5

# 测试服务是否正常
curl http://localhost:8000/v1/models
```

**重要提示**：
- Judge模型应该运行在**不同于训练的GPU**上
- 如果GPU资源有限，可以使用较小的模型（如Qwen2.5-3B）
- 建议使用较低的`gpu_memory_utilization`以留出空间给训练

### 2. 配置环境变量

在训练脚本中设置judge后端信息：

```bash
# vLLM judge服务器配置
export VLLM_JUDGE_API_BASE="http://localhost:8000"  # vLLM服务地址
export VLLM_JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # Judge模型名称
export VLLM_JUDGE_MAX_RETRIES="3"                   # 最大重试次数
export VLLM_JUDGE_TIMEOUT="30"                      # 请求超时(秒)
```

如果judge服务在远程机器上：
```bash
export VLLM_JUDGE_API_BASE="http://192.168.1.100:8000"
```

## 配置训练使用LLM Judge

### 方法1: 修改配置文件

在你的训练配置YAML中添加：

```yaml
reward_model:
  enable: False  # 不使用模型based reward
  custom_reward_function:
    path: verl/utils/reward_score/llm_judge_vllm.py
    name: compute_score
  launch_reward_fn_async: True  # 启用异步调用（推荐）

# 如果数据是特定benchmark，可以设置data_source使用judge
data:
  train_files: data/benchmarks_processed/GAIA_test.parquet
  val_files: null
  # 确保data中包含question字段和ground_truth
```

### 方法2: 命令行覆盖

```bash
python3 -m verl.trainer.main_ppo \
    --config-path examples/sglang_multiturn/config \
    --config-name search_multiturn_grpo \
    reward_model.enable=False \
    reward_model.custom_reward_function.path=verl/utils/reward_score/llm_judge_vllm.py \
    reward_model.custom_reward_function.name=compute_score \
    reward_model.launch_reward_fn_async=True \
    data.train_files=data/benchmarks_processed/all_benchmarks_test.parquet
```

## Judge Prompt格式

实现中使用的prompt格式：

```
Given a Question and its Golden Answer, verify whether the Predicted Answer is correct.
The prediction is correct if it fully aligns with the meaning and key information of the
Golden Answer. Respond with True if the prediction is correct and False otherwise.

Question: {question}

Golden Answer: {golden_answer}

Predicted Answer: {predicted_answer}

Answer (True/False):
```

## 完整训练示例

### 示例1: 使用LLM Judge训练GRPO

```bash
#!/bin/bash

# 1. 启动judge服务（在单独的终端）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 2048

# 2. 设置judge配置
export VLLM_JUDGE_API_BASE="http://localhost:8000"
export VLLM_JUDGE_MODEL="Qwen/Qwen2.5-3B-Instruct"
export VLLM_JUDGE_MAX_RETRIES="3"
export VLLM_JUDGE_TIMEOUT="30"

# 3. 激活环境
conda activate /share/project/wanli/env/verl-v060

# 4. 运行训练（使用剩余的GPU）
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
    --config-path examples/sglang_multiturn/config \
    --config-name search_multiturn_grpo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.rollout.n=4 \
    reward_model.enable=False \
    reward_model.custom_reward_function.path=verl/utils/reward_score/llm_judge_vllm.py \
    reward_model.custom_reward_function.name=compute_score \
    reward_model.launch_reward_fn_async=True \
    data.train_files=data/benchmarks_processed/all_benchmarks_test.parquet \
    data.train_batch_size=256 \
    trainer.total_epochs=3 \
    trainer.logger='["console","wandb"]'
```

### 示例2: 仅推理（不训练）

如果只想用judge评估模型输出而不训练：

```bash
#!/bin/bash

# 设置judge配置
export VLLM_JUDGE_API_BASE="http://localhost:8000"
export VLLM_JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"

# 运行推理评估脚本（需要自己实现或使用verl的eval模式）
python evaluate_with_judge.py \
    --model-path checkpoints/my_model \
    --test-data data/benchmarks_processed/GAIA_test.parquet \
    --use-llm-judge \
    --output results/judge_scores.json
```

## 数据格式要求

使用LLM judge时，数据必须包含以下字段：

```python
{
    "data_source": "GAIA",  # benchmark名称
    "extra_info": {
        "question": "黑龙江、吉林、辽宁，共有多少个地市级行政单位与外国接壤？",  # 必需
        # 其他metadata...
    },
    "reward_model": {
        "ground_truth": {
            "target": ["3"]  # 或 np.array(["3"])，必需
        },
        "style": "rule"
    },
    # 其他字段...
}
```

## 性能考虑

### 吞吐量

- **串行调用**: ~1-2 samples/sec（取决于judge模型大小）
- **并行调用**: 建议使用`launch_reward_fn_async=True`，可以达到更高吞吐
- **批处理**: 代码支持批处理，但需要在reward_manager层面启用

### 成本优化

1. **使用较小的judge模型**
   ```bash
   # 3B模型通常足够，速度快且显存占用小
   --model Qwen/Qwen2.5-3B-Instruct
   ```

2. **减少max_tokens**
   - Judge只需要输出True/False，已设置为`max_tokens=10`

3. **缓存结果**（可选增强）
   - 对相同的(question, golden_answer, predicted_answer)组合缓存结果
   - 可以添加Redis或本地缓存

### 延迟优化

- **启用异步**: `launch_reward_fn_async=True`
- **增加重试**: `VLLM_JUDGE_MAX_RETRIES=3`
- **调整超时**: 如果judge模型快，可以降低timeout

## 故障排查

### 问题1: 连接judge服务失败

**错误信息**: `Connection refused` 或 `Timeout`

**解决方案**:
```bash
# 检查judge服务是否运行
curl http://localhost:8000/v1/models

# 检查端口是否正确
netstat -tuln | grep 8000

# 检查防火墙（如果是远程服务）
telnet 192.168.1.100 8000
```

### 问题2: Judge返回格式错误

**错误信息**: `Unable to parse judge response`

**原因**: Judge模型输出不是True/False

**解决方案**:
- 使用instruction-tuned模型（如Qwen2.5-Instruct）
- 调整temperature=0.0确保确定性输出
- 检查judge模型是否理解英文prompt（可以改用中文）

### 问题3: Reward全是0

**原因**: 可能是judge服务未启动或配置错误

**检查步骤**:
```python
# 测试judge函数
from verl.utils.reward_score.llm_judge_vllm import compute_score

result = compute_score(
    data_source="test",
    solution_str="<answer>3</answer>",
    ground_truth={"target": ["3"]},
    extra_info={"question": "1+2=?"}
)
print(result)  # 应该看到 {'score': 1.0, ...}
```

### 问题4: 训练过程中judge调用太慢

**优化方案**:
1. 使用更快的judge模型（3B而不是7B）
2. 增加`tensor_parallel_size`（如果有多GPU）
3. 确保`launch_reward_fn_async=True`已启用
4. 检查网络延迟（如果judge在远程）

## 进阶: 自定义Judge Prompt

如果需要修改judge prompt，编辑 `llm_judge_vllm.py` 中的 `_create_judge_prompt` 方法：

```python
def _create_judge_prompt(self, question: str, golden_answer: str, predicted_answer: str) -> str:
    """自定义prompt模板"""
    # 示例：中文prompt
    prompt = f"""请判断预测答案是否正确。如果预测答案与标准答案的含义和关键信息完全一致，请回答"正确"，否则回答"错误"。

问题：{question}

标准答案：{golden_answer}

预测答案：{predicted_answer}

判断结果（正确/错误）："""
    return prompt
```

然后相应修改 `_parse_judge_response` 来解析新格式。

## 参考

- verl reward manager文档: `docs/中文文档/reward奖励计算说明.md`
- vLLM部署文档: https://docs.vllm.ai/
- Async reward配置: `verl/trainer/config/reward_model/`
