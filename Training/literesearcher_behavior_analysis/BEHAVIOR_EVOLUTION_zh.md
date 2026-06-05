# RL 两阶段训练的行为演化分析
*Qwen3-4B Deep Research Agent 在纯 on-policy GRPO 下的轨迹级行为变迁*

> 🇬🇧 **English version**: [`BEHAVIOR_EVOLUTION.md`](BEHAVIOR_EVOLUTION.md)

> **Paper 章节稿 · 李万里 · 2026-06-03**
>
> **数据来源**
> - Sim rollout（训练时采样轨迹，128 prompts × 8 GRPO rollout per step）
>   - S1: `verl/rollout_trajectory/qwen3_deepresearch_tis_rl_onpolicy_bs128_local_rag_only_token-mean-seq-mean-temp_0.7_length_32k_nokl/`
>   - S2: `verl/rollout_trajectory/qwen3_deepresearch_tis_rl_stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k/`
> - Online benchmark（GAIA pass@4，每个 step 412 条记录）
>   - S1: `DeepResearch/bench_results/qwen3-4B-RL/onpolicy_bs128_local_rag_only_token-mean-seq-mean-temp_0.7_length_32k_nokl/`
>   - S2: `DeepResearch/bench_results/qwen3-4B-RL/stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k/`
> - 覆盖范围：32 个 checkpoint（S1=19, S2=13）；每个 checkpoint 抽 300 条 rollout × 30 个行为指标。
> - 6 张图（`fig1` – `fig6`）+ 10 条代表性 trajectory 实例（`trajectory_examples.json`）随附本文档。

---

## 0 · 一段话 paper-style TL;DR

纯 on-policy GRPO 在 **无 KL、无 entropy 正则** 的设置下，模型的行为轨迹呈现一条 **非单调的五阶段曲线**：(A) **base 模型的探索式行为**，token / turn / 搜索预算都很高；(B) **压缩阶段**，模型收敛到一个「短而准」的策略，长度、turns、工具调用全部减半，但准确率却在上升；(C) **静默衰退阶段**，sim reward 继续在涨，但 quoted-query、`<think>` 块这些关键先验在悄悄被消除；(D) **终末崩塌**，由两个 reward 信号盲区（32k 上限 + `<answer>` 标签要求）联合触发，`no_extraction` 比例在一步内从 1 % 跳到 38 %；(E) **二阶段救援**，通过 *warm-start 自最后一个健康 checkpoint（ckpt 220）* + 上下文窗口扩到 48k + 加入更难的数据 mix，*反转了行为分布*，进入长 reasoning、多 turn、自我反思的真正能泛化到 GAIA 的模式（pass@1 55 → 68 %）。本文档用 30+ 指标 × 32 个 checkpoint + 10 条代表性 trajectory 对每个 regime 进行量化与举例。

---

## 1 · 五阶段行为分类表

| 阶段 | Stage / steps | 定义性行为 | Sim reward | GAIA acc | 触发原因 |
|---|---|---|---|---|---|
| **A. Base 探索期** | S1 0–80 | 17k tok, 28 turns, 7.8 search/traj, 16.8 queries, 46 % 引号查询 | 0.50 | n/a | 继承自 Qwen3-4B-Thinking-2507 |
| **B. 压缩期** | S1 80–250 | 10k tok, 19 turns, 4.0 search, 9 queries, 16 % 引号查询 | 0.66 → 0.70 | 56 → 59 % | GRPO advantage 偏好「短而准」样本 |
| **C. 静默衰退** | S1 250–490 | 长度/turns 震荡，但 quoted-query 跌到 8 %、`<think>` 跌到 9 块 | 0.66 → 0.76 | 62 → 63 % | 在 reward-flat manifold 上漂移 |
| **D. 终末崩塌** | S1 490–540 | `no_extraction` 1 % → 38 %, turns 21 → 10, search 3 → 1.3, `<think>` 9 → 3.7 | 0.72 → 0.51 | 63 → 51 % | 32k 截断 + `<answer>` 强约束联合触发 reward 失明 |
| **E. 二阶段救援** | S2 0–500（warm from S1 ckpt 220） | 15–21k tok, 32–45 turns, hedge ×3 vs S1, 16 think blocks, 引号 12 % | 0.41 → 0.85 | 60 → 68 % | 48k 上限 + 难数据 mix 重新打开长 reasoning manifold |

后续章节对每个阶段进行细化，包括具体 trajectory 证据与图表依据。

---

## 2 · 实验配置与指标

### 2.1 两阶段训练管道

```
Qwen3-4B-Thinking-2507  ── S1 GRPO (~787 步) ──► （最终崩塌）
                              │
                              └─► step 220 ckpt ────► S2 GRPO (~570 步) ──► 最终模型
```

- **S1** 配置：`loss_agg_mode = seq-mean-token-mean`，`train_batch = mini_batch = 128`（纯 on-policy ⇒ `ratio ≡ 1`，787 步全程 `pg_clipfrac ≡ 0`），无 KL，entropy_coeff = 0，lr = 1e-6 constant，temp 0.7，`max_response_length = 32k`，`max_assistant_turns = 40`，reward = LLM-judge {0,1}，若 `extract_solution` 找不到合法 `</answer>` 则强制 score = 0。
- **S2** 配置：warm-start 自 S1 ckpt 220，`max_response_length = 48k`，temp 1.0，数据扩为 `wiki 16k-32k + mqa_subgraph6 + science`，对多跳推理要求更高。

### 2.2 30+ 行为指标（per checkpoint）

对每条 trajectory 解析其 `<tool_call>{…}</tool_call>` JSON 块、`<think>…</think>` 段、`<answer>…</answer>` 标签，导出：

- **输出预算**：`output_tokens`、`total_turns`、`truncated_frac`（≥ 31.5k tokens）、`no_extraction_frac`（无 `<answer>` 标签）
- **工具使用**：各工具调用次数（search / visit / python）、总 query 数、每次 search 的 query 数（search 支持 query list）、平均 query 长度、含 `"…"` 引号的 query 比例
- **推理风格**：`<think>` 块的数量与字符总长、`<think>` 内中文字符比例、hedge 词数（`wait`、`actually`、`let me reconsider`、重新、等等…）
- **结果**：`score`、`correct`、`method`（`llm_judge` / `no_extraction`）
- **Online**：交叉对照 GAIA `summary.json` 的 pass@1 / pass@4 + 用同样 30 个指标分析每条 `result_*.json`

代码：`extract_behaviors.py`、`run_all.py`、`mine_examples.py`、`make_figs.py`。原始时间轴数据：`behavior_timeline.json`。Trajectory 实例：`trajectory_examples.json`。

---

## 3 · 阶段 A —— Base 探索期（S1 0–80 步）

Base 模型 Qwen3-4B-Thinking-2507 本身已经是一个「agent」—— 它会输出 `<think>` 块和 `<tool_call>` JSON。但它对搜索成本毫无校准。Step 1 的 rollout 显示模型在 **暴搜问题**：单题打了 25 次 tool call、45 个 query，撞到 32k 上限，超时后从未输出 `<answer>`。

**Step 1 的代表性 rollout**

| 字段 | 值 |
|---|---|
| 问题 | *"Which airline loyalty program, known for eliminating mileage expiration starting in 2011, absorbed a major U.S. carrier's frequent flyer program on October 1, 2009…"* |
| Output tokens | **32 768**（截断）|
| Total turns | **51** |
| Tool calls | 25（search + visit 交替）|
| Total queries | 45 |
| 前 3 个 query | `"\"eliminating mileage expiration\" 2011 airline loyalty program"`, `"\"mileage expiration\" eliminated 2011 airline"`, `"\"October 1, 2009\" \"absorbed\" \"frequent flyer program\""` |
| 最终 `<answer>` | **（无 —— `no_extraction`，reward = 0）** |
| 最后一个 `<think>` | *"Search results not helpful. Let's search for 'Diamond Medallion' and 'private jet card'."* |

这条 trajectory 集中体现了阶段 A 的所有特征：激进的引号精确匹配（base 模型的真实强项）、反复改写 query、在预算压力下完全 *无法 commit* 答案。该阶段所有 rollout 的引号 query 占比：**46 %** —— 全两阶段训练中最高的 checkpoint。

阶段 A 的聚合指标：

| 指标 | Step 1 | Step 80（外推边界）|
|---|---|---|
| 平均 output tokens | 17 398 | ≈ 11 000 |
| 平均 turns | 28.3 | ≈ 19 |
| 平均 tool calls | 7.8 search + 0.9 visit | 4.0 + 0.5 |
| 平均 queries | 16.8 | 9 |
| 引号 query 占比 | **46.0 %** | 24 % |
| 截断率 | 23 % | 4 % |
| Sim reward | 0.497 | 0.59 |

此时 reward 信号被一个 failure mode（截断 → 无 answer 标签 → score 0）主导。GRPO advantage 因而压倒性地偏好 **能足够早收尾以输出 `<answer>` 的 trajectory**，这正是阶段 B 的入场条件。

---

## 4 · 阶段 B —— 压缩期（S1 80–250 步）

这就是 *paper 的 headline checkpoint 诞生的阶段* —— S2 warm-start 用的就是这里的 step **220**。经验上，在 step 80 到 step 250 之间，policy 把 token 预算砍半，准确率却涨了 17 pp。

**Step 220 的代表性正确 rollout**

| 字段 | 值 |
|---|---|
| 问题 | *"Gamma Aquariids Meteor Shower 所属的星座是哪一个？"* |
| Output tokens | **1 957** |
| Total turns | 6 |
| Tool calls | `[search, visit]`（各 1 次）|
| Queries | `["Gamma Aquariids Meteor Shower 星座", "…属于哪个星座", "…constellation"]`（一次 search 里 3 个 query）|
| 首个 `<think>` | *"搜索结果包含一个指向 'Gamma Aquariids Meteor Shower' 的链接。我们打开它。"* |
| 末个 `<think>` | *"该页面显示星座是水瓶座 (Aquarius)。所以答案是：水瓶座 (Aquarius)。"* |
| `<answer>` | `Aquarius` ✓ |

这是教科书式的 **search → visit → answer** 模式。注意 RL 学到的三个性质：
1. **多 query 批处理**：模型在 *一次* `search` 调用里塞 3 个 query（`queries_per_search_call ≈ 3.06`）。Base 模型也会，但不那么稳定。
2. **双语 query**：故意混用中文和英文 query，覆盖两个 index 空间 —— 注意中文 query、双语改写、英文兜底的组合。
3. **One-shot commit**：单次 `visit` + 单次 `<answer>`，不再二次怀疑。

阶段 B 的聚合指标：

| 指标 | Step 100 | Step 200 | Step 220（ckpt）| Step 250 |
|---|---|---|---|---|
| Sim acc | 0.593 | 0.643 | **0.700** | 0.653 |
| GAIA pass@1 | — | 0.561 | **0.592** | — |
| 平均 tokens | 10 983 | 9 607 | 10 805 | 10 838 |
| 平均 turns | 18.8 | 17.2 | 19.2 | 19.8 |
| 平均 search 调用 | 4.25 | 3.49 | 3.98 | 3.94 |
| Queries / search | 2.90 | 3.13 | **3.06** | 3.05 |
| 引号 query 占比 | 24.3 % | 16.3 % | **15.7 %** | 12.7 % |
| 截断率 | 4.3 % | 0.7 % | 2.3 % | 3.7 % |
| no_extraction | 4.3 % | 0.7 % | **1.7 %** | 4.7 % |

step 220 时已经能看到两个早期预警信号，但只看 reward 看不出来：
- 引号 query 占比从 46 % 跌到 16 % —— base 模型 *精确匹配先验已被腐蚀 65 %*。
- `no_extraction` 处于极低的 1.7 %，但这是 *因为模型学会了提前收尾*，而不是 *学会了 commit 更好的答案*。这一点会在阶段 D 反咬一口。

**为什么 ckpt 220 是 S2 的正确 warm-start 点**（一个 paper-grade 的论断）：step 220 是 *第一个* 同时满足 (sim_acc ≥ 0.70) ∧ (no_extraction ≤ 2 %) ∧ (中文 query 能力健在) 的 checkpoint。更早的 checkpoint 准确率不够；更晚的 checkpoint 已经在丢失 S2 需要的行为先验。


---

## 5 · 阶段 C —— 静默衰退（S1 250–490 步）

这是最微妙、也是教学价值最高的阶段。Sim reward 一直在涨（0.66 → 0.76），GAIA pass@1 也跟着小幅上扬（62 % → 63 %）。在工程师能看到的每一个 dashboard 上，S1 看起来都很健康。但 policy 内部正在 **悄悄消除 reward 函数看不见的行为先验**：

| 指标 | Step 250 | Step 350 | Step 450 | Step 480 |
|---|---|---|---|---|
| Sim acc | 0.653 | 0.720 | **0.760** | 0.720 |
| GAIA pass@1 | — | — | — | 0.629 |
| 平均 tokens | 10 838 | 11 621 | 11 477 | 10 301 |
| 平均 turns | 19.8 | 23.7 | 23.3 | 21.1 |
| 引号 query 占比 | 12.7 % | 16.0 % | **7.6 %** | 9.1 % |
| 截断率 | 3.7 % | 3.0 % | 1.7 % | **0.7 %** |
| no_extraction | 4.7 % | 5.0 % | 1.7 % | **1.0 %** |

这种模式是 *学会规避 reward 的盲区* 而非 *学会做任务*。三个观察：

1. **截断率掉到 0.7 %**（step 480）—— 模型 *学会了* 一旦撞 32k 就铁定得 0，于是反射性地在到达 32k 前提前关闭轨迹。这个技能本身是有用的，但 **正是这个技能会在阶段 D 引爆崩塌**：一旦「短即安全」被内化，任何进一步缩短的趋势都会被奖励。
2. **引号 query 占比崩到 7.6 %**（step 450）。Base 模型「从问题里抓实体短语、加引号搜」的强先验 —— 在 entity-heavy 检索任务上极有用 —— 正在被遗忘，因为在很多简单的训练问题里不加引号的 query 也能成功。
3. **Reward 继续上涨**，因为这些被悄悄腐蚀的先验在相对容易的训练分布（`rag_direct`, `local_rag_only`）上不会 *直接* 损害聚合准确率。它们会在 GAIA 上造成伤害 —— 但 step 480 的 GAIA 结果（0.629）还没拉响警报。

**Step 480 的 trajectory（与 step 220 同一道题对比）**：

| 字段 | Step 220 | Step 480 |
|---|---|---|
| Output tokens | 1 957 | 1 915（基本一致）|
| Total turns | 6 | 6 |
| Search calls | 1 | 1 |
| Tool 序列 | `[search, visit]` | `[search, visit]` |
| 首个 `<think>` | *"…我们打开它。"* | *"…我们打开 universeguide.com 上的页面。"* |
| `<answer>` | `Aquarius` ✓ | `Aquarius` ✓ |

在这道简单题上，两个 checkpoint 行为上无法区分。衰退在 *单个* 易题样本上是 *不可见的*；它只在 *困难* 样本的行为分布上有种群水平的位移 —— 而 GAIA 暴露的正是这个分布。

**为什么这个阶段会以崩塌而非平台收尾**：损失聚合 `seq-mean-token-mean`（∑ᵢ (1/Nᵢ) ∑ₜ loss_{i,t}，再 mean over batch）让每条 *trajectory* 等权重，与长度无关。再加上无 KL、无 entropy floor，没有任何力量在反对进一步缩短。Policy 沿着 reward-flat manifold 一路滑向更短更短的轨迹，直到某一步让足够多的轨迹掉进「单 turn 瞎猜」区。

---

## 6 · 阶段 D —— 终末崩塌（S1 490–540 步）

Step 480 到 step 510 之间，policy 穿过了一条相变线。所有 *输出预算* 轴上的指标同步崩溃：

| 指标 | Step 480 | Step 510 | Δ |
|---|---|---|---|
| Sim acc | 0.720 | 0.510 | **−21 pp** |
| 平均 output tokens | 10 301 | 9 301 | −10 %（有迷惑性 —— 见下）|
| 平均 turns | **21.1** | **10.2** | **−52 %** |
| 平均 search 调用 | 3.08 | 1.29 | −58 % |
| `<think>` 块 | 9.5 | 3.7 | −61 % |
| 截断率 | 0.7 % | 6.3 % | ×9 |
| **no_extraction** | **1.0 %** | **38.3 %** | **×38** |
| Hedge tokens | 0.9 | 1.4 | （噪声）|

**真正的 smoking gun 是 `no_extraction`**。一个训练步之内，无 `<answer>` 标签的 rollout 比例从 1 % 跳到 38 %。读 `verl/utils/reward_score/llm_judge_async.py` 可以确认：任何 `extract_solution` 失败的 trajectory 都被强制赋 `score = 0.0`、`method = "no_extraction"`，*与答错完全不可区分*。GRPO 因此拿到一个「你 40 % 的 trajectory 是错的」的 reward 信号，但完全不知道是 *内容错* 还是 *格式错*。

**Step 510 的崩塌型 trajectory**：

| 字段 | 值 |
|---|---|
| 问题 | *"斯洛博达耶希瓦在1929年阿拉伯暴乱后迁往耶路撒冷之前，其原名是什么，以及迁往耶路撒冷后更名为什么？"* |
| Output tokens | **32 768**（截断）|
| Turns | 14 |
| Tool calls | 6（`[search, visit, visit, search, visit, visit]`）|
| 最后一个 `<think>` 末尾 80 字符 | *"…我们查一下希伯伦耶希瓦维基百科 (Hebron Yeshiva Wikipedia)。"* |
| `<answer>` | **（无 —— 始终未输出）** |
| Reward | 0（no_extraction）|

对比阶段 A 的 step-1 trajectory：failure mode 一模一样（在 commit 之前耗尽预算）。区别在于阶段 A 时模型在 *探索*；阶段 D 时模型已经 *停止探索*（turns 28 → 10），但同时也不再 *commit*（因为 policy 已经被推过了「会收尾一个多跳问题」的边界）。它现在被卡在两个差 attractor 之间。

崩塌窗口内：
- 引号 query 占比崩到 1 %（step 510），到 step 750 进一步降到 **0.1 %**。
- `<think>` 块降到 3.7（step 510）；每个 `<think>` 现在是单句话而非规划块。
- Search 调用降到 1.3 /traj：模型已停止迭代。

**虚假恢复（step 540–787）**：sim reward 缓慢爬回 0.80（step 660），但 GAIA pass@1 一直被钉在 0.51–0.58。恢复在 *训练分布* 上是真实的（模型重新学会输出 *某个* `<answer>` 标签，即使是瞎猜），但在 GAIA 上不真实 —— 见第 8 节的 sim-real gap 图。


---

## 7 · 阶段 E —— 二阶段救援（S2 0–500 步，warm from S1 ckpt 220）

S2 不是 S1 的延续，而是 *regime reset*。Warm-start 自 S1 ckpt 220 保留了阶段 B 残存的行为先验；新的上下文窗口（32k → 48k）消除了驱动阶段 C 衰退的 reward 盲区；更难的数据 mix 强迫多跳推理 *无法被走捷径*。这三个变化共同翻转了行为 attractor。

S2 全程的 **宏观 signature**：

| 指标 | S2 step 1 | S2 step 100 | S2 step 300 | S2 step 500 | S2 step 570 |
|---|---|---|---|---|---|
| Sim acc | 0.410 | 0.610 | 0.717 | **0.847** | 0.727 |
| GAIA pass@1 | — | 0.600 | 0.650 | 0.663 | **0.682** |
| GAIA pass@4 | — | 80.6 | 83.5 | 79.6 | 85.4 |
| 平均 output tokens | 18 497 | 15 731 | 16 379 | 15 771 | **20 933** |
| 平均 turns | **36.8** | 35.9 | 38.8 | 32.9 | **44.6** |
| 平均 `<think>` 块 | 17.4 | 16.9 | 18.4 | 15.5 | **21.3** |
| 平均 hedge tokens | 2.6 | 3.9 | 4.4 | 2.9 | **5.0** |
| 引号 query 占比 | 19.3 % | 15.6 % | 16.9 % | 12.5 % | 12.1 % |
| no_extraction | 9.7 % | 6.7 % | 4.3 % | **2.3 %** | 8.0 % |

对比 S1 *最强* checkpoint（step 480）：平均 turns 21.1、`<think>` 9.5、hedge 0.9。**S2 产出 2 × 的 turns、1.7 × 的 `<think>` 块、3–5 × 的 hedge tokens**，并且 GAIA pass@1 单调上升。

**Step 400 的长 reasoning trajectory（正确，难多跳）**：

| 字段 | 值 |
|---|---|
| 问题 | *"香川县选区在2023年1月31日时，现任参议员中属于2022年当选类别且为第三届任期的是谁？"* |
| Output tokens | **44 781** |
| Total turns | **114** |
| Tool calls | 56（大部分是 `visit`，逐页扫维基百科）|
| Total queries | 21 |
| 首个 `<think>` | *"我们有多个相关的维基百科页面。让我们打开关于'香川县选区'的维基百科页面。"* |
| 末个 `<think>` | *"现在我们有了答案。因此，答案是：Yoshihiko Isozaki。"* |
| `<answer>` | `Yoshihiko Isozaki` ✓ |

这条 trajectory **在 S1 不可能存在** —— 114 turns × ~400 tokens 早在第 80 turn 就撞 32k 上限了。S2 的 48k 上限（外加模型 *学到了愿意用满*）创造了一个全新的行为模式。

**Step 570 的长 reasoning trajectory（最难题）**：

| 字段 | 值 |
|---|---|
| 问题 | *"What is the name of the exhibition expected to travel to several art museums in California starting in 2024 and accompanied by the book Julia Morgan: The Road to San Simeon?"* |
| Output tokens | **47 409**（接近 48k 上限）|
| Turns | **120** |
| Tool calls | 59 |
| Queries | 27 |
| 末个 `<think>` | *"Thus, difficult. Nevertheless, I'll answer: The exhibition is the forthcoming traveling museum exhibition…"* |
| `<answer>` | （长描述性答案 —— judge 判为正确）✓ |

两个重要的行为性质：
1. **模型知道自己不确定**，频繁输出 `Thus, difficult`、`Nevertheless, I'll answer` 之类的 hedge。S1 step 450+ 的 hedge_count = 1.2；S2 step 570 = 5.0。
2. **模型愿意 commit 一个长的描述性答案** 而不是什么都不输出。这与 S1 的崩塌模式（用「短」换「不 commit」）正相反。

### 7.1 S2 成功而 S1 失败的三个机制

1. **上下文窗口不匹配的解除**。训练 reward 是基于 `max_response_length` 截断后的生成计算的。S1 这个上限是 32k —— 与 GAIA 评测一致，所以 reward landscape *被截断系统性地扭曲*。S2 上限是 48k，远超任何 GAIA 任务所需，截断率降到 25 % 以下，policy gradient 重新看见内容质量这个维度。
2. **数据难度下限**。S2 加入了 `wiki_16k-32k`、`mqa_subgraph6`、science 数据 —— 一次 search-and-answer 在这些题上注定赢不了。S1 滑过的那条 reward-flat manifold（阶段 C）在 S2 不存在，因为在新数据分布上更短的 trajectory *严格更差*。**压缩力在数据层面被消除**。
3. **Warm-start 点的选择**。从 ckpt 220（最后一个健康点）而非 ckpt 480（看似 reward 最高但已经深陷阶段 C 衰退）出发，意味着 S2 继承的模型还保留 16 % 引号 query、8.6 个 `<think>` 块、19 turns / traj。这些是 S2 放大的 *底料*。如果从 ckpt 480 出发，同样的训练几乎肯定 *无法重建* 已经丢失的行为多样性。


---

## 8 · 图

所有图共用两个视觉约定：S1 step **220** 处的灰色虚线（S2 起点）+ S1 step **490–540** 的橙色带（崩塌窗口）。

### 图 1 —— Accuracy 与回答预算

![fig1](fig1_acc_length.png)

- (a) Sim train acc：S1 在 step 450 见顶（0.76），step 510 崩到 0.51。S2 在 step 500 见顶（0.85），无崩塌。
- (b) GAIA pass@1：S1 在 step 480 见顶（0.63），step 600 跌到 0.51。S2 在 step 400 & 570 达到 0.68。
- (c)(d) Response length：S1 从 17k 跌到 7k tokens；S2 全程 13–21k（真正的 reasoning）。
- (e)(f) Turns：S1 27 → 10，S2 全程 32–45 —— 长 reasoning 模式。

### 图 2 —— 工具使用 & query 形状

![fig2](fig2_tools.png)

- (a) Search 调用：S1 7.8 → 1.3（崩塌后），S2 9.9 → 3.8（健康）。
- (b) Visit 调用：S1 在崩塌后 visit ≈ 0；S2 全程维持 visit（多跳难题必需）。
- (d) 每次 search 的 query 数：S1 健康期稳定在 3.0，崩塌后跳到 4.7（一次 search 塞一大坨 query，"fire-and-forget" 退化症状）；S2 稳定在 2.5–3.6。
- (f) 引号 query 占比 —— **最具诊断性的单一指标**：S1 46 % → 7.6 % → **0.1 %**（绝迹）；S2 11–19 %（保留）。

### 图 3 —— 截断与 answer-tag 失败

![fig3](fig3_truncation.png)

- (a) S1 sim 截断率跌到 0.7 %（step 480 —— 模型学会规避），后续略反弹。S2 sim 截断率 11–25 %（模型用满 48k headroom）。
- (b) **S1 sim `no_extraction` 单步从 1.0 % 跳到 38.3 %（480 → 510）**。S2 全程 2–10 %。
- (c)(d) GAIA inference（评测时无截断上限），两个 stage 的 answer-tag 比例都稳定 —— 问题在于 *训练时 reward 信号被扭曲*，不在模型本身的 answer-format 能力。

### 图 4 —— 推理风格：thinking、语言、自我反思

![fig4](fig4_thinking.png)

- (a)(b) `<think>` 块数：S1 13 → 9 → 4（崩塌）→ 6（恢复）；S2 15–21。
- (c) 中文字符占比：两个 stage 都稳定在 3.5–5.5 % —— base 模型的双语习惯被保留（且对跨语言检索覆盖率有用）。
- (d) Hedge tokens：S1 最低 0.9；S2 2.8–5.0。S2 模型公开权衡多个选项，*这正是多跳自我修正的底料*。

### 图 5 —— Sim-to-Real 一致性

![fig5](fig5_sim2real.png)

- (a) Sim (rollout) vs Real (GAIA pass@1) —— S1 在 step 510 后两条线大幅分叉；S2 两条线全程同步。
- (b) Sim − Real gap：S1 健康期 ≈ 0.08–0.13；**崩塌后 0.17–0.27**（reward 测的是 sim 模式的过拟合）；S2 全程 0.05–0.18。

> **这是 paper "sim-to-real" 贡献的量化论断**：Stage 2 的训练 reward 全程是 GAIA pass@1 的 *忠实预测器*；Stage 1 在 step 500 之后失去这个性质，训练信号与真实任务表现反相关。

### 图 6 —— GAIA pass@1 / pass@4

![fig6](fig6_passk.png)

- S1 pass@4 见顶 84.5（step 220–480）；S2 pass@4 见顶 **86.4（step 60）** —— warm-start 后仅 60 步就达到 paper 最高水准。
- pass@4 − pass@1 gap：S1 ≈ 22 pp，S2 ≈ 15 pp —— S2 更 *确定*：rollout 收敛到正确答案而非发散。


---

## 9 · 完整指标表（每个 checkpoint 采样 300 条 rollout）

| stage | step | sim_acc | bench_acc | resp_tok | turns | n_search | queries | qPer | quoted% | trunc% | noAns% | thinks | hedge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 1   | 0.497 |  —    | 17398 | 28.3 | 7.77 | 16.78 | 2.70 | 46.0 | 23.0 | 23.0 | 13.0 | 2.6 |
| S1 | 30  | 0.530 |  —    | 15473 | 25.7 | 6.90 | 14.68 | 2.75 | 40.7 | 15.7 | 15.0 | 11.8 | 2.5 |
| S1 | 60  | 0.517 |  —    | 12549 | 20.4 | 5.16 | 10.73 | 2.74 | 35.0 |  9.0 |  8.7 |  9.2 | 2.2 |
| S1 | 100 | 0.593 |  —    | 10983 | 18.8 | 4.25 |  9.46 | 2.90 | 24.3 |  4.3 |  4.3 |  8.4 | 1.5 |
| S1 | 150 | 0.663 |  —    | 10782 | 19.8 | 4.67 |  9.08 | 2.74 | 22.0 |  6.0 |  6.3 |  8.9 | 2.1 |
| S1 | 200 | 0.643 | 0.561 |  9607 | 17.2 | 3.49 |  8.44 | 3.13 | 16.3 |  0.7 |  0.7 |  7.6 | 1.8 |
| S1 | **220** | **0.700** | **0.592** | 10805 | 19.2 | 3.98 | 9.61 | 3.06 | 15.7 | 2.3 | 1.7 | 8.6 | 1.5 |
| S1 | 250 | 0.653 |  —    | 10838 | 19.8 | 3.94 |  9.07 | 3.05 | 12.7 |  3.7 |  4.7 |  8.9 | 1.0 |
| S1 | 300 | 0.627 | 0.624 | 12527 | 25.6 | 5.08 |  9.08 | 2.26 | 19.3 |  3.3 |  3.0 | 11.8 | 1.4 |
| S1 | 350 | 0.720 |  —    | 11621 | 23.7 | 5.02 |  8.88 | 2.26 | 16.0 |  3.0 |  5.0 | 10.8 | 2.8 |
| S1 | 400 | 0.707 |  —    | 12191 | 25.0 | 4.78 |  9.23 | 2.49 |  9.2 |  3.0 |  3.7 | 11.5 | 1.4 |
| S1 | 450 | 0.760 |  —    | 11477 | 23.3 | 4.11 |  8.71 | 2.68 |  7.6 |  1.7 |  1.7 | 10.7 | 1.2 |
| S1 | 480 | 0.720 | 0.629 | 10301 | 21.1 | 3.08 |  7.36 | 2.94 |  9.1 |  0.7 |  1.0 |  9.5 | 0.9 |
| S1 | **510** | **0.510** | — | 9301 | **10.2** | **1.29** | 5.73 | 4.70 | 1.0 | 6.3 | **38.3** | **3.7** | 1.4 |
| S1 | 550 | 0.600 |  —    |  6576 |  8.4 | 1.20 |  5.74 | 4.99 |  0.5 |  5.7 | 12.0 |  3.1 | 1.0 |
| S1 | 600 | 0.677 | 0.507 |  7011 | 10.6 | 1.55 |  6.15 | 4.44 |  0.8 |  4.0 |  6.7 |  4.2 | 0.8 |
| S1 | 660 | 0.803 | 0.583 |  9361 | 15.6 | 2.06 |  7.70 | 4.22 |  0.8 |  5.0 |  6.7 |  6.8 | 1.3 |
| S1 | 700 | 0.767 | 0.532 | 11164 | 18.2 | 2.83 |  8.88 | 3.64 |  0.7 |  8.0 |  9.7 |  8.0 | 1.1 |
| S1 | 750 | 0.803 |  —    |  9691 | 13.7 | 2.01 |  8.21 | 4.38 |  0.1 |  6.3 |  8.7 |  5.8 | 1.8 |
| S2 | 1   | 0.410 |  —    | 18497 | 36.8 | 9.97 | 17.59 | 2.59 | 19.3 | 17.0 |  9.7 | 17.4 | 2.6 |
| S2 | 30  | 0.553 |  —    | 17871 | 36.3 | 9.96 | 17.70 | 2.51 | 18.6 | 14.3 |  8.0 | 17.1 | 2.3 |
| S2 | 60  | 0.550 | 0.633 | 16159 | 33.7 | 7.55 | 15.26 | 3.01 | 17.9 | 12.3 |  9.0 | 15.8 | 2.8 |
| S2 | 100 | 0.610 | 0.600 | 15731 | 35.9 | 6.79 | 13.33 | 2.66 | 15.6 | 11.0 |  6.7 | 16.9 | 3.9 |
| S2 | 150 | 0.647 |  —    | 15745 | 35.1 | 5.81 | 12.00 | 2.99 | 13.1 | 12.0 |  7.0 | 16.6 | 3.9 |
| S2 | 200 | 0.623 | 0.617 | 14790 | 34.0 | 5.66 | 10.65 | 2.86 | 13.3 | 13.0 |  6.3 | 16.0 | 3.9 |
| S2 | 250 | 0.710 |  —    | 13205 | 32.1 | 4.72 |  8.97 | 2.86 | 11.6 | 10.0 |  8.3 | 15.1 | 2.8 |
| S2 | 300 | 0.717 | 0.650 | 16379 | 38.8 | 5.81 | 10.54 | 2.62 | 16.9 | 11.3 |  4.3 | 18.4 | 4.4 |
| S2 | 350 | 0.727 |  —    | 17404 | 38.2 | 5.51 | 11.78 | 3.08 | 11.0 | 15.7 |  6.7 | 18.1 | 3.3 |
| S2 | 400 | 0.747 | 0.682 | 16343 | 34.8 | 4.85 | 10.78 | 3.01 | 14.3 | 14.3 |  4.3 | 16.4 | 3.0 |
| S2 | 450 | 0.773 |  —    | 17220 | 34.7 | 4.84 | 12.47 | 3.43 | 11.4 | 14.3 |  4.0 | 16.3 | 3.2 |
| S2 | **500** | **0.847** | 0.663 | 15771 | 32.9 | 3.76 | 10.81 | 3.64 | 12.5 | 12.0 | **2.3** | 15.5 | 2.9 |
| S2 | 570 | 0.727 | **0.682** | 20933 | **44.6** | 4.97 | 13.66 | 3.58 | 12.1 | 25.0 |  8.0 | **21.3** | **5.0** |

字段说明：`resp_tok` = 平均 output tokens；`turns` = 每题 agent turns；`n_search` = 平均 search 工具调用次数；`queries` = 总 query 数（search 支持 query list）；`qPer` = 每次 search 的 query 数；`quoted%` = 含 `"…"` 引号的 query 占比；`trunc%` = 撞 32k / 48k 上限的 trajectory 占比；`noAns%` = 缺 `<answer>` 标签的占比（强制 score = 0）；`thinks` = 每 traj 的 `<think>` 块数；`hedge` = 自我反思 token 数（`wait`/`actually`/重新/等等）。

---

## 10 · 对 paper 的论断意义

1. **为什么 two-stage curriculum 是必要而非可选**。Stage 1 不能无限训下去 —— 它有一个确定性的崩塌机制（第 6 节），增加 step 数无法规避。Two-stage 设计因此不只是个 convenience，而是 *对「GRPO + seq-mean-token-mean + no-KL + 有界上下文」这个 setting 中无界压缩力的必要校正*。

2. **为什么 checkpoint 选择比 lr 更重要**。选 ckpt 220（而非 reward 更高的 ckpt 480）是整次训练的 *最重要单一超参*。按 reward 选更晚的 checkpoint 会让 S2 warm-start 进入一个已经丢失了 hard 多跳所需先验的行为盆地。

3. **为什么要监控行为指标而非 reward 曲线**。引号 query 占比、`<think>` 块数、`no_extraction` 比例三者联合能在崩塌 *之前* 就抓住阶段 C 衰退。Sim reward 和 GAIA pass@1 都没抓住（两者一直涨到 step 480）。

4. **为什么 sim-to-real gap 是正确的成功指标**。Paper 的贡献不是「我们在 GAIA 上拿到 71 %」—— 很多论文都能。而是「我们训出了一个 sim reward 与 GAIA pass@1 全程相关的 RL policy」—— 这是 agentic RL 真正的难题。图 5 是这个论断的量化版本。

5. **完整 ablation 表的样子**。完整图景需要 ablate (a) 48k 上限（S2 回 32k）、(b) 数据 mix（S2 只用 S1 数据）、(c) warm-start 点（S2 从 ckpt 480）、(d) loss 聚合（用 `token-mean` 替 `seq-mean-token-mean`）。每行应在 **(sim reward 轨迹形状, sim-real gap, GAIA pass@1)** 这三元组上 benchmark —— 上面的行为分析预言了哪些行会崩塌、怎么崩塌。

---

## 11 · 复现指令

```bash
cd <本目录>

# 1. 抽取 30+ 行为指标 per checkpoint（约 1 分钟）
python run_all.py          # 产出 behavior_timeline.json

# 2. 挖 10 条代表性 trajectory（约 5 秒）
python mine_examples.py    # 产出 trajectory_examples.json

# 3. 渲染 6 张图（约 5 秒）
python make_figs.py        # 产出 fig1..6.png
```

抽取器输入是本文档开头列出的 rollout JSONL 文件与 per-task GAIA result JSON。

— 完 —

---

## 12 · 完整训练路径上的行为演化  （S1 step 0–220 ▸ S2 step 0–570，在 step 220 处拼接）

Stage 2 直接从 S1 ckpt-220 warm-start（被废弃的 S1 ≥ 240 分支与最终模型无关），所以从信息量上讲，最好把两阶段画在 **同一条连续的 global-step 坐标轴上**：S1 的 0–220 步后接 S2 的 0–570 步（重新索引为 global step 220–790）。下面 12 个行为指标都同时在 *训练 on-policy rollout*（每步 300 条轨迹）和 *held-out GAIA pass@1 评测*（每步 412 个任务）上计算。

![continuous full](fig_continuous_full.png)

### 12.1 paper 应该重点讲的 12 个趋势

| # | 指标 | S1 0 → 220 (rollout) | S2 0 → 570 (rollout) | 解读 |
|---|---|---|---|---|
| 1 | **任务正确率** | 0.50 → **0.70** | 0.41 → **0.85** | S2 从 0.41（数据更难）出发，最终比整个 S1 峰值高 21 pp。|
| 2 | **平均输出长度**（tokens）| 17.4 k → **10.8 k** | 18.5 k → **20.9 k** | S1 在 32 k 上限下 *压缩* 输出；S2 在 48 k 上限下 *扩展* 输出。|
| 3 | **平均 agent turn 数** | 28.3 → **19.2** | 36.8 → **44.6** | S1 缩短轨迹；S2 把 turn 数翻倍 —— 多跳推理变可行。|
| 4 | **每轨 search 调用数** | 7.8 → **4.0** | 10.0 → **5.0** | 搜索变 *更高效*（调用变少），然后稳定下来。|
| 5 | **每轨 browse / visit 调用数** ↑ | 5.5 → 4.6（基本持平）| 7.5 → **16.4** | S2 学会 *读页面*，而不仅仅是搜页面。**这是最关键的行为变化之一**。|
| 6 | **Browse 占比**（visit / (search+visit)）↑ | 0.46 → 0.56 | 0.50 → **0.76** | 训练末期模型每 4 次 tool call 有 3 次是在读内容。|
| 7 | **Search 并发度**（每次 search 的 query 数）↑ | 2.70 → **3.06** | 2.59 → **3.58** | 每次 search 请求批量打更多互异 query —— 习得的 cost-amortization。|
| 8 | **每轨总 query 数** | 16.8 → 9.6 | 17.6 → 13.7 | 总 query 数下降，但 **每个 query 的信息密度** 上升（见 #5–7）。|
| 9 | **引号精确匹配 query 占比** | 0.46 → 0.16 | 0.19 → 0.12 | 引号 query 先验缩小；S2 通过 page-reading 通道部分替代它。|
| 10 | **每轨 `<think>` 块数** | 13.0 → 8.6 | 17.4 → **21.3** | S2 末期每轨 reasoning 次数比 S2 起点多 17 %，比 S1 ckpt-220 多约 2.5×。|
| 11 | **思考密度**（`<think>` 字符 / 总字符）| 0.092 → 0.089（持平）| 0.102 → **0.142** | S2 末期 reasoning 文本占整段输出的比重显著上升。|
| 12 | **自我反思 hedge 词**（`wait / actually / 重新…`）| 2.58 → 1.47 | 2.64 → **5.03** | S2 模型 *给自己纠错* 的频率比 S1 ckpt-220 多 3.4×。|

### 12.2 Headline 图 —— 4 条主要趋势

![continuous headline](fig_continuous_headline.png)

这张图设计为可直接 drop-in paper。把 sim train 和 GAIA eval 画在同一 x 轴上，对应 4 个最能概括 policy 演化的指标：

- (a) 拼接后 sim 与 real 准确率收敛，最终 GAIA pass@1 ≈ 0.68。
- (b) Browse / visit 调用数沿 S2 近线性增长 —— 训练分布和 GAIA 评测上 *同步增长*，说明这是 **可迁移的行为**，不是 sim reward 过拟合。
- (c) Search 并发度同步增长，说明模型在 *并行* 检索而非串行。
- (d) `<think>` 块数在拼接处翻倍，并继续增长 —— 长程 planning 成为主导 reasoning 模式。

### 12.3 训练 vs 评测准确率单图

![continuous acc](fig_continuous_acc.png)

适合 intro 用的单 panel 图：train rollout reward（蓝）vs GAIA pass@1（红）vs GAIA pass@4（绿）沿连续训练轴。Sim 和 real 曲线在几乎整个训练全程同步；唯一系统性背离出现在 S2 末期（step 700+），train reward 飙高而 GAIA 稳在 ≈ 0.68 —— 提示模型已接近该 benchmark 的自然上限。

### 12.4 图所支撑的 paper 论断

这种单轴视角支持三个清晰的 paper 论断：

1. **行为转变沿 global training step 单调推进** —— 12 个指标里有 8 个是单调的（browse 调用、browse 占比、search 并发、`<think>` 块数、思考密度、hedge 次数、准确率、turn 数）。剩下 4 个（response length、总 query 数、引号 query 占比、原始 search 调用数）遵循「效率压缩」的预期模式。
2. **对于能同时测量的行为，sim 与 real 方向完全一致**。即使绝对数值有差距（如 response length，因为 eval 没有 length cap），**方向永远一致**。
3. **最关键的变化发生在 *S2 训练过程中*，而不在拼接点本身**。拼接处确实让 length、turns、`<think>` 数出现一个 step jump（数据 mix + context window 改变），但 browse 占比（0.50 → 0.76）和 `<think>` 密度（0.10 → 0.14）的缓慢爬升是 *S2 内部的持续学习效应* —— 这是 S2 在做真实 RL 而非仅靠继承一个更好起点的直接证据。

