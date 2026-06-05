# VERL 验证流程完全指南

## 一、概述

本文档详细解释 VERL 框架中的验证（Validation）流程，包括：
- `_validate()` 方法的完整工作流程
- `process_validation_metrics()` 函数的统计机制
- Majority Voting 的实现原理
- 常见错误及解决方案

---

## 二、验证触发与配置

### 2.1 何时触发验证？

```python
# 在训练配置中设置
trainer.test_freq=50          # 每 50 步验证一次
trainer.val_before_train=False # 训练前不验证
```

### 2.2 验证数据采样策略

```python
actor_rollout_ref.rollout.val_kwargs.n=5        # 每个问题生成 5 个答案
actor_rollout_ref.rollout.val_kwargs.do_sample=False  # 使用贪心解码（确定性）
```

**关键理解**：每个验证问题会被重复 N 次（n=5），生成 N 个不同答案，用于统计 Best-of-N、Worst-of-N、Majority Voting 等指标。

---

## 三、`_validate()` 方法详解

### 3.1 整体流程

```
加载验证数据 → 重复采样(×5) → 生成响应 → 计算奖励 → 统计指标 → 返回结果
```

### 3.2 关键步骤

#### Step 1: 初始化数据收集器

```python
reward_extra_infos_dict: dict[str, list] = defaultdict(list)  # 存储所有额外信息

# 收集样本用于日志和导出
sample_inputs = []      # 输入问题
sample_outputs = []     # 模型生成的回答
sample_scores = []      # 奖励分数
sample_uids = []        # 唯一标识符（用于分组）
```

#### Step 2: 重复采样（核心机制）

```python
test_batch = test_batch.repeat(
    repeat_times=5,  # n=5
    interleave=True
)
```

**示例**：
```
原始: [问题1, 问题2, 问题3]  (3个问题)
重复后: [问题1, 问题1, 问题1, 问题1, 问题1,  (共15个样本)
       问题2, 问题2, 问题2, 问题2, 问题2,
       问题3, 问题3, 问题3, 问题3, 问题3]
```

**UID 机制**：同一问题的 5 个副本共享同一个 `uid`，后续按 `uid` 分组统计。

#### Step 3: 生成响应

```python
test_gen_batch.meta_info = {
    "do_sample": False,      # 贪心解码
    "validate": True,
    "global_steps": self.global_steps,
}

# 调用 SGLang/vLLM 生成
test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
```

#### Step 4: 计算奖励（关键！）

```python
# 调用 reward function
result = self.val_reward_fn(test_batch, return_dict=True)
reward_tensor = result["reward_tensor"]
scores = reward_tensor.sum(-1).cpu().tolist()

# 收集主要分数
reward_extra_infos_dict["reward"].extend(scores)

# 收集额外信息（如 pred, score 等）
if "reward_extra_info" in result:
    for key, lst in result["reward_extra_info"].items():
        reward_extra_infos_dict[key].extend(lst)
```

**数据结构示例**（修复后）：
```python
reward_extra_infos_dict = {
    "reward": [0.0, 0.0, 1.0, 0.0, 0.0, ...],  # 所有样本的总分数
    "score": [0.0, 0.0, 1.0, 0.0, 0.0, ...],   # 详细分数
    "pred": ["错误", "", "巴黎", "", "错误", ...]  # 提取的答案（空字符串代替 None）
}
```

#### Step 5: 调用统计函数

```python
data_sources = np.concatenate(data_source_lst, axis=0)  # 数据源标签

data_src2var2metric2val = process_validation_metrics(
    data_sources,           # ["searchR1_nq", "searchR1_nq", ...]
    sample_uids,            # ["uid-1", "uid-1", "uid-1", ...]
    reward_extra_infos_dict # {"reward": [...], "pred": [...], "score": [...]}
)
```

#### Step 6: 格式化输出

```python
metric_dict = {}
for data_source, var2metric2val in data_src2var2metric2val.items():
    for var_name, metric2val in var2metric2val.items():
        for metric_name, metric_val in metric2val.items():
            # 构建指标名称
            if var_name == "reward" and metric_name.startswith(("mean", "best", "maj")):
                prefix = "val-core"  # 核心指标
            else:
                prefix = "val-aux"   # 辅助指标

            full_name = f"{prefix}/{data_source}/{var_name}/{metric_name}"
            metric_dict[full_name] = metric_val
```

**输出示例**：
```python
{
    "val-core/searchR1_nq/reward/mean@5": 0.4593,
    "val-core/searchR1_nq/reward/best@5/mean": 0.6234,
    "val-core/searchR1_nq/reward/maj@5/mean": 0.5120,
    "val-aux/searchR1_nq/reward/std@5": 0.3421,
    ...
}
```

---

## 四、`process_validation_metrics()` 函数详解

### 4.1 函数签名

```python
def process_validation_metrics(
    data_sources: list[str],      # 每个样本的数据源标签
    sample_uids: list[str],        # 每个样本的唯一 ID
    infos_dict: dict[str, list],   # 包含多个变量的字典
    seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]
```

### 4.2 输入数据示例

假设有 2 个问题，每个生成 5 个答案（n=5），共 10 个样本：

```python
data_sources = [
    "searchR1_nq", "searchR1_nq", "searchR1_nq", "searchR1_nq", "searchR1_nq",  # 问题1
    "searchR1_nq", "searchR1_nq", "searchR1_nq", "searchR1_nq", "searchR1_nq",  # 问题2
]

sample_uids = [
    "uid-123", "uid-123", "uid-123", "uid-123", "uid-123",  # 问题1的5个答案共享 uid
    "uid-456", "uid-456", "uid-456", "uid-456", "uid-456",  # 问题2的5个答案共享 uid
]

infos_dict = {
    "reward": [0.0, 0.0, 1.0, 0.0, 0.0,  # 问题1：第3个答案正确
               0.0, 1.0, 1.0, 0.0, 0.0], # 问题2：第2、3个答案正确
    "pred": ["错误1", "", "巴黎", "", "错误2",     # 问题1（空字符串表示未提取）
             "", "巴黎", "Paris", "伦敦", ""],      # 问题2
    "score": [0.0, 0.0, 1.0, 0.0, 0.0,
              0.0, 1.0, 1.0, 0.0, 0.0]
}
```

### 4.3 处理流程

#### 阶段 1：按 (data_source, uid, variable) 分组

```python
data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for sample_idx, data_source in enumerate(data_sources):
    uid = sample_uids[sample_idx]
    for var_name, var_vals in infos_dict.items():
        data_src2uid2var2vals[data_source][uid][var_name].append(var_vals[sample_idx])
```

**结果结构**：
```python
{
    "searchR1_nq": {
        "uid-123": {
            "reward": [0.0, 0.0, 1.0, 0.0, 0.0],
            "pred": ["错误1", "", "巴黎", "", "错误2"],
            "score": [0.0, 0.0, 1.0, 0.0, 0.0]
        },
        "uid-456": {
            "reward": [0.0, 1.0, 1.0, 0.0, 0.0],
            "pred": ["", "巴黎", "Paris", "伦敦", ""],
            "score": [0.0, 1.0, 1.0, 0.0, 0.0]
        }
    }
}
```

#### 阶段 2：为每个 (data_source, uid, variable) 计算统计量

```python
for data_source, uid2var2vals in data_src2uid2var2vals.items():
    for uid, var2vals in uid2var2vals.items():
        for var_name, var_vals in var2vals.items():
            # 跳过字符串变量
            if isinstance(var_vals[0], str):
                continue  # pred 字段会被跳过

            # 计算数值统计量
            metric = {}
            n_resps = len(var_vals)  # 5
            metric[f"mean@{n_resps}"] = np.mean(var_vals)
            metric[f"std@{n_resps}"] = np.std(var_vals)
```

**对 `"uid-123"` 的 `"reward"` 计算结果**：
```python
{
    "mean@5": 0.2,   # (0+0+1+0+0)/5 = 0.2
    "std@5": 0.4,    # 标准差
    ...
}
```

#### 阶段 2.5：计算 Best-of-N 和 Worst-of-N（Bootstrap Sampling）

```python
if n_resps > 1:
    ns = [2, 4, 5]  # 采样数量：2^1, 2^2, n_resps

    for n in ns:
        # Bootstrap 采样
        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
            data=var_vals,
            subset_size=n,
            reduce_fns=[np.max, np.min],
            seed=seed
        )
        metric[f"best@{n}/mean"] = bon_mean   # Best-of-N 的均值
        metric[f"best@{n}/std"] = bon_std
        metric[f"worst@{n}/mean"] = won_mean  # Worst-of-N 的均值
        metric[f"worst@{n}/std"] = won_std
```

**Best-of-N 示例**（best@2）：
- 从 `[0.0, 0.0, 1.0, 0.0, 0.0]` 中随机抽 2 个，取最大值，重复 1000 次：
  - 第1次：`[0.0, 1.0]` → max = 1.0
  - 第2次：`[0.0, 0.0]` → max = 0.0
  - 第3次：`[1.0, 0.0]` → max = 1.0
  - ...
- 计算这 1000 个最大值的均值：`mean ≈ 0.4`

#### 阶段 2.6：计算 Majority Voting（如果有 `pred` 字段）

```python
if var2vals.get("pred", None) is not None:
    # 将 reward 和 pred 配对
    vote_data = [
        {"val": val, "pred": pred}
        for val, pred in zip(var_vals, var2vals["pred"])
    ]

    [(maj_n_mean, maj_n_std)] = bootstrap_metric(
        data=vote_data,
        subset_size=n,
        reduce_fns=[calc_maj_val],
        seed=seed
    )
    metric[f"maj@{n}/mean"] = maj_n_mean
    metric[f"maj@{n}/std"] = maj_n_std
```

**Majority Voting 原理**（`calc_maj_val` 函数）：

```python
# 输入数据（问题2的5个答案）
vote_data = [
    {"pred": "",      "val": 0.0},
    {"pred": "巴黎",  "val": 1.0},
    {"pred": "Paris", "val": 1.0},
    {"pred": "伦敦",  "val": 0.0},
    {"pred": "",      "val": 0.0}
]

# Step 1: 按 pred 分组
vote2vals = {
    "":      [0.0, 0.0],  # 出现2次
    "巴黎":  [1.0],        # 出现1次
    "Paris": [1.0],        # 出现1次
    "伦敦":  [0.0]         # 出现1次
}

# Step 2: 统计出现次数
vote2cnt = {"": 2, "巴黎": 1, "Paris": 1, "伦敦": 1}

# Step 3: 找出多数答案
maj_vote = ""  # 空字符串出现最多（2次）

# Step 4: 返回该答案第一次出现时的分数
maj_val = 0.0  # 返回第一个空字符串对应的分数

# 注意：如果答案规范化（巴黎 = Paris），需要预处理
```

**重要**：当前的 majority voting 是基于**字符串精确匹配**，如果需要语义匹配（"巴黎" = "Paris"），需要在 `pred` 中预先规范化答案。

#### 阶段 3：跨问题（uid）聚合统计量

```python
# 收集所有 uid 的统计量
data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for data_source, uid2var2metric in data_src2uid2var2metric.items():
    for uid, var2metric in uid2var2metric.items():
        for var_name, metric in var2metric.items():
            for metric_name, metric_val in metric.items():
                data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

# 计算平均值
data_src2var2metric2val = {}
for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
    for var_name, metric2uid_vals in var2metric2uid_vals.items():
        for metric_name, uid_vals in metric2uid_vals.items():
            data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)
```

**示例**：
- `"uid-123"` 的 `mean@5` = 0.2
- `"uid-456"` 的 `mean@5` = 0.4
- 最终输出：`mean@5` = 0.3（所有问题的平均值）

### 4.4 输出结构

```python
{
    "searchR1_nq": {
        "reward": {
            "mean@5": 0.3,
            "std@5": 0.35,
            "best@2/mean": 0.5,
            "best@5/mean": 0.85,
            "maj@5/mean": 0.4,
            ...
        },
        "score": {
            "mean@5": 0.3,
            ...
        }
        # 注意：pred 是字符串，被跳过了
    }
}
```

---

## 五、关键概念详解

### 5.1 Best-of-N (BoN) Sampling

**定义**：从 N 个生成的答案中选择分数最高的那个。

**应用场景**：
- 代码生成：允许模型多次尝试，选择最佳结果
- 复杂推理：增加找到正确答案的概率

**Bootstrap 模拟**：
```
问题: "法国首都？"
5个答案及分数: [错误(0), 错误(0), 正确(1), 错误(0), 错误(0)]

Best@5: 1.0 (从全部5个中选最好的)
Best@2: 通过 bootstrap 估计（随机抽2个，取最大，重复1000次）
  → 约 0.4 (因为有 1/5 概率抽到正确答案)
```

### 5.2 Majority Voting

**定义**：从多个答案中选择**出现次数最多**的答案。

**应用场景**：
- 消除随机性错误
- 提高鲁棒性（多数答案一致则更可信）

**与 Best-of-N 的区别**：
- **Best-of-N**：基于**分数**选择（需要有评分标准）
- **Majority Voting**：基于**答案文本**的频率选择

**示例**：
```
5个答案及分数:
  "巴黎"(1.0), "柏林"(0.0), "巴黎"(1.0), "巴黎"(1.0), "伦敦"(0.0)

Best@5: 1.0 (任意一个"巴黎")
Majority: "巴黎"出现3次，返回第一个"巴黎"的分数 1.0
```

### 5.3 UID 分组机制

**目的**：区分"同一问题的多个答案"和"不同问题"。

```
uid-123: [答案1, 答案2, 答案3, 答案4, 答案5]  → 计算统计量 → metric1
uid-456: [答案1, 答案2, 答案3, 答案4, 答案5]  → 计算统计量 → metric2
...
最终指标 = mean(metric1, metric2, ...)  # 跨问题聚合
```

### 5.4 数据源（Data Source）

不同数据集分别统计，便于分析模型在不同任务上的表现：

| 数据源 | 类型 | 特点 |
|-------|------|------|
| `searchR1_nq` | Natural Questions | 事实性问答 |
| `searchR1_triviaqa` | TriviaQA | 琐事问答 |
| `searchR1_hotpotqa` | HotpotQA | 多跳推理 |
| `searchR1_2wikimultihopqa` | 2WikiMultihopQA | 维基百科多跳 |
| `searchR1_musique` | MuSiQue | 音乐知识 |

---

## 六、数据类型处理规则

### 6.1 支持的数据类型

| 类型 | 示例字段 | 处理方式 |
|------|---------|---------|
| **数值** | `reward`, `score`, `acc` | 计算 mean, std, best@N, worst@N, maj@N |
| **字符串** | `pred` | 跳过单独统计，仅用于 majority voting |

### 6.2 不支持的数据类型

| 类型 | 问题 | 解决方案 |
|------|------|---------|
| `None` | 无法计算数值统计 | 转为空字符串 `""` 或跳过 |
| 混合类型 | `[0.9, "text", None]` | 统一类型 |

### 6.3 字段命名规范

| 字段名 | 用途 | 必须是字符串？ |
|-------|------|---------------|
| `reward` | 主要奖励分数 | ❌ 数值 |
| `score` | 详细分数 | ❌ 数值 |
| `pred` | 预测答案文本（用于 majority voting） | ✅ 字符串 |
| ~~`pred_ans`~~ | ❌ 错误命名（框架期望 `pred`） | - |

---

## 七、常见错误与解决方案

### 7.1 错误：`TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'`

**原因**：`reward_extra_infos_dict` 中包含 `None` 值。

**错误代码**：
```python
# 旧版本
return {
    "pred_ans": answer,  # answer 可能是 None
    "score": 0
}
```

**修复后**：
```python
# 新版本
pred_text = answer if answer is not None else ""
return {
    "pred": pred_text,  # 空字符串代替 None
    "score": 0
}
```

### 7.2 错误：Majority Voting 不生效

**原因 1**：字段名错误（使用了 `pred_ans` 而不是 `pred`）

**修复**：确保字段名为 `"pred"`：
```python
result = {
    "pred": pred_text,  # ✅ 正确
    # "pred_ans": pred_text,  # ❌ 错误
}
```

**原因 2**：`pred` 字段包含 `None` 值

**修复**：将 `None` 转为空字符串。

### 7.3 训练时不报错，验证时报错

**原因**：训练和验证对 `reward_extra_infos_dict` 的处理方式不同。

| 阶段 | 处理方式 | 是否调用统计函数？ |
|------|---------|------------------|
| **训练** | 转为 numpy array，存储到 `batch.non_tensor_batch` | ❌ 否 |
| **验证** | 调用 `process_validation_metrics` 计算统计量 | ✅ 是 |

**解决方案**：确保 `reward_extra_infos_dict` 中没有 `None` 值。

---

## 八、实际案例：agentloop_baseline.sh 对比

### 8.1 配置对比

| 配置项 | agentloop_baseline.sh | agentloop_search_browse.sh（修复前）|
|-------|---------------------|---------------------------|
| `rollout.n` | 5 | 5 |
| `max_assistant_turns` | 2 | 10 |
| `compute_score 返回值` | **float** | **dict（包含 pred）** |
| `pred 字段` | ❌ 无 | ✅ 有（修复后） |

### 8.2 验证输出对比

**Baseline 输出**：
```python
reward_extra_infos_dict = {
    "reward": [0.0, 0.0, 1.0, 0.0, 0.0, ...]  # 只有 reward
}

# 最终指标
{
    "val-core/searchR1_nq/reward/mean@5": 0.4593,
    "val-core/searchR1_nq/reward/best@5/mean": 0.6234,
    # 没有 maj@N（因为没有 pred 字段）
}
```

**修复后的版本**：
```python
reward_extra_infos_dict = {
    "reward": [0.0, 0.0, 1.0, 0.0, 0.0, ...],
    "score": [0.0, 0.0, 1.0, 0.0, 0.0, ...],
    "pred": ["", "", "巴黎", "", "", ...]  # 支持 majority voting
}

# 最终指标
{
    "val-core/searchR1_nq/reward/mean@5": 0.4593,
    "val-core/searchR1_nq/reward/best@5/mean": 0.6234,
    "val-core/searchR1_nq/reward/maj@5/mean": 0.5120,  # ✅ 新增
    "val-aux/searchR1_nq/score/mean@5": 0.4593,
    ...
}
```

---

## 九、调试技巧

### 9.1 查看验证输出文件

```bash
# 查看导出的验证结果
ls -lh validation_trajectory/*/
cat validation_trajectory/*/50.jsonl | jq '.' | head -20
```

**JSONL 格式**：
```json
{
  "input": "问题文本",
  "output": "模型生成的完整对话",
  "gt": {"target": ["正确答案1", "正确答案2"]},
  "score": 1.0,
  "reward": 1.0,
  "pred": "巴黎",
  "uid": "uuid-..."
}
```

### 9.2 检查 None 比例

```bash
grep "Extracted answer: None" logs/*.log | wc -l
grep "Extracted answer is not None" logs/*.log | wc -l
```

如果 None 比例 >80%，说明模型需要更多训练或 `max_assistant_turns` 设置不足。

### 9.3 添加调试日志

在 `metric_utils.py:438` 添加：
```python
for var_name, var_vals in var2vals.items():
    print(f"[DEBUG] Processing {var_name}: first={var_vals[0]}, type={type(var_vals[0])}, len={len(var_vals)}")
    if isinstance(var_vals[0], str):
        print(f"[DEBUG] Skipping {var_name} (string)")
        continue
```

---

## 十、常见问题 FAQ

### Q1: 为什么验证时使用贪心解码（do_sample=False）？

**A**: 验证需要确定性结果，便于比较不同 checkpoint 的性能。训练时可以采样增加多样性。

### Q2: Best@5 和 Mean@5 哪个更重要？

**A**:
- **Mean@5**: 反映模型的平均性能
- **Best@5**: 反映允许多次尝试时的最佳性能（更贴近实际应用）

在代码生成等任务中，Best@N 更实用；在对话任务中，Mean@N 更重要。

### Q3: Majority Voting 与 Best-of-N 有何区别？

**A**:
- **Best-of-N**: 需要评分标准，选分数最高的
- **Majority Voting**: 基于答案文本频率，选出现最多的

Majority Voting 更适合多个等价答案的场景（如"巴黎"和"Paris"）。

### Q4: 验证需要多长时间？

**A**: 取决于验证集大小和生成长度：
- 51,713 问题 × 5 = 258,565 样本
- 假设每样本 2 秒 → 理论 143 小时
- 实际有并行 → 约 1-2 小时（取决于 GPU 数量）

### Q5: 可以跳过验证吗？

**A**: 可以设置 `trainer.test_freq=-1` 禁用，但会失去性能监控。建议降低验证频率（如 `test_freq=200`）而不是完全禁用。

### Q6: 如何支持语义等价的 Majority Voting？

**A**: 在 `compute_score` 中规范化答案：
```python
# 规范化预测答案
def normalize_answer_for_voting(answer):
    mapping = {"Paris": "巴黎", "Beijing": "北京", ...}
    return mapping.get(answer, answer)

pred_text = normalize_answer_for_voting(answer) if answer else ""
```

---

## 十一、总结

### 11.1 核心要点

1. **验证流程**：加载 → 重复(×5) → 生成 → 奖励 → 统计 → 输出
2. **UID 机制**：同一问题的多个答案共享 uid，先按 uid 分组统计，再跨 uid 聚合
3. **数据类型规则**：
   - 数值变量（reward, score）→ 计算统计量
   - 字符串变量（pred）→ 跳过单独统计，仅用于 majority voting
   - **不允许 None 值**

### 11.2 修复检查清单

- [x] `compute_score` 返回 `"pred"` 字段（不是 `"pred_ans"`）
- [x] `None` 值转为空字符串 `""`
- [x] 确保 `pred` 是字符串类型
- [x] 验证 majority voting 是否生效（检查日志中的 `maj@N` 指标）

### 11.3 最佳实践

1. **字段命名**：使用 `"pred"` 而不是 `"pred_ans"`
2. **处理 None**：转为空字符串或特殊标记 `"NO_ANSWER"`
3. **答案规范化**：如需语义等价的 majority voting，预先规范化答案
4. **调试验证**：定期查看导出的 JSONL 文件，检查 `pred` 字段是否正确

---

**文档版本**: 2.0
**最后更新**: 2025-11-03
**适用版本**: VERL v0.6.0
