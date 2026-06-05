# Reward计算机制说明

## ✅ 问题已解决

已将6个新benchmark的data_source添加到`default_compute_score`函数中，使其使用与SearchR1相同的exact match reward计算方法。

## Reward计算流程

### 1. 配置层（Config）

**文件**: `verl/trainer/config/data/legacy_data.yaml`
```yaml
reward_fn_key: data_source  # 使用data_source字段来选择reward函数
```

**说明**:
- `reward_fn_key` 指定从数据中提取哪个字段来选择reward函数
- 默认值是 `data_source`
- 这个字段的值会传递给 `default_compute_score()` 的第一个参数

### 2. 数据层（Data）

每条数据包含：
```python
{
    "data_source": "GAIA",  # 用于选择reward函数
    "prompt": [...],
    "reward_model": {
        "ground_truth": {"target": ["答案"]},
        "style": "rule"
    },
    ...
}
```

### 3. Reward函数选择（Function Selection）

**文件**: `verl/utils/reward_score/__init__.py`

```python
def default_compute_score(data_source, solution_str, ground_truth, ...):
    """根据data_source选择不同的reward计算函数"""

    # 对于SearchR1和新的benchmarks，使用相同的exact match
    if data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
        # 新增的benchmarks
        "GAIA",
        "GPQA",
        "HLE",
        "WebWalkerQA",
        "Browsecomp",
        "Xbench",
    ]:
        from . import search_r1_like_qa_em
        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    # 对于GSM8K，使用不同的函数
    elif data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)

    # 对于MATH数据集
    elif data_source in ["lighteval/MATH", ...]:
        from . import math_reward
        res = math_reward.compute_score(solution_str, ground_truth)

    # 对于代码数据集
    elif data_source in ["codecontests", "apps", ...]:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
```

### 4. 调用链路（Call Chain）

```
数据加载 (DataLoader)
    ↓
DataProto.batch["data_source"] = ["GAIA", "GPQA", ...]
    ↓
训练循环 (ray_trainer.py:1107)
    ↓
compute_reward(batch, reward_fn)
    ↓
Reward Manager (reward_manager/naive.py)
    ↓
对每个样本循环:
    data_source = batch["data_source"][i]  # 例如: "GAIA"
    solution = batch["responses"][i]       # 模型输出
    ground_truth = batch["reward_model"][i]["ground_truth"]
    ↓
    score = default_compute_score(
        data_source=data_source,
        solution_str=solution,
        ground_truth=ground_truth
    )
    ↓
根据data_source=="GAIA"，调用:
    search_r1_like_qa_em.compute_score(solution, ground_truth)
    ↓
返回: {"score": 1.0, "pred_ans": "提取的答案"}
```

## Exact Match Reward详解

**文件**: `verl/utils/reward_score/search_r1_like_qa_em.py`

### 评分逻辑

```python
def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """
    Exact Match (EM) scoring function

    Args:
        solution_str: 模型输出的完整响应
        ground_truth: {"target": ["正确答案1", "正确答案2", ...]}
        method: "strict" - 严格提取<answer>标签
        format_score: 格式正确但答案错误的得分 (默认0.0)
        score: 答案正确的得分 (默认1.0)

    Returns:
        dict: {"score": float, "pred_ans": str}
    """

    # 1. 从<answer>标签中提取答案
    answer = extract_solution(solution_str)
    # 例如: "<think>...</think><answer>34689</answer>" -> "34689"

    # 2. 如果没有提取到答案
    if answer is None:
        return {"score": 0, "pred_ans": ""}

    # 3. Exact Match检查（带归一化）
    if em_check(answer, ground_truth["target"]):
        return {"score": 1.0, "pred_ans": answer}
    else:
        return {"score": 0.0, "pred_ans": answer}
```

### 答案归一化（Normalization）

```python
def normalize_answer(s):
    """归一化答案，用于EM比较"""
    # 1. 转小写: "Beijing" -> "beijing"
    # 2. 移除冠词: "the answer" -> "answer"
    # 3. 移除标点: "1,000" -> "1000"
    # 4. 去除多余空格: "a  b" -> "a b"

    return white_space_fix(remove_articles(remove_punc(lower(s))))
```

**例子**:
```python
# 模型输出
solution = "<answer>The Beijing</answer>"
# Ground truth
ground_truth = {"target": ["beijing", "Beijing, China"]}

# 提取: "The Beijing"
# 归一化: "beijing"
# 匹配第一个ground truth的归一化结果: "beijing"
# 结果: score=1.0 ✓
```

## 测试结果

### 单元测试
```bash
python3 test_reward_calculation.py
```

**结果**:
- ✅ GAIA: 正确答案得1分，错误答案得0分
- ✅ GPQA: 正确答案得1分，错误答案得0分
- ✅ HLE: 正确答案得1分，错误答案得0分
- ✅ WebWalkerQA: 正确答案得1分，错误答案得0分
- ✅ Browsecomp: 正确答案得1分，错误答案得0分
- ✅ Xbench: 正确答案得1分，错误答案得0分
- ✅ searchR1_nq: 正确答案得1分，错误答案得0分

### 真实数据测试
使用每个benchmark的第一条真实数据测试，所有测试通过 ✓

## 自定义Reward函数（可选）

如果不想修改 `default_compute_score`，也可以在配置文件中指定自定义reward函数：

```yaml
# 在 google_search_browse_multiturn_grpo.yaml 中添加
custom_reward_function:
  path: verl/utils/reward_score/search_r1_like_qa_em.py
  name: compute_score
  reward_kwargs:
    format_score: 0.0  # 格式正确但答案错误的分数
    score: 1.0         # 答案正确的分数
```

**优点**:
- 不修改源代码
- 可以为不同实验配置不同的reward函数
- 可以传递自定义参数

**缺点**:
- 需要在每个配置文件中重复配置
- 绕过了data_source的自动选择机制

## 总结

| 项目 | 说明 |
|------|------|
| **配置字段** | `data.reward_fn_key = "data_source"` |
| **数据字段** | 每条数据的 `data_source` 字段 |
| **选择函数** | `default_compute_score()` |
| **具体实现** | `search_r1_like_qa_em.compute_score()` |
| **评分方法** | Exact Match (EM) with normalization |
| **支持的data_source** | searchR1系列 + 6个新benchmark |
| **返回格式** | `{"score": float, "pred_ans": str}` |

### 关键优势

1. ✅ **统一评分标准**: 所有benchmark使用相同的EM评分
2. ✅ **自动选择**: 根据data_source自动选择正确的reward函数
3. ✅ **可扩展**: 添加新benchmark只需在列表中加一行
4. ✅ **已测试**: 所有benchmark都经过单元测试和真实数据测试
5. ✅ **不区分来源**: 合并的数据集可以直接使用，无需额外配置

现在可以直接运行推理了！🎉
