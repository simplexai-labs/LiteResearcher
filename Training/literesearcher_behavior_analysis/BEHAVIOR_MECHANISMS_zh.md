# Sim2Real 收益背后的行为机制

> 我们 Qwen3-4B 深度研究 agent 两阶段 RL 训练的、可直接写入 paper 的行为分析。
> English companion: [BEHAVIOR_MECHANISMS.md](./BEHAVIOR_MECHANISMS.md).
> 详细的失败模式分类与轨迹样本请见上游长文档
> [BEHAVIOR_EVOLUTION_zh.md](../assets/behavior_evolution/BEHAVIOR_EVOLUTION_zh.md)。

---

## 0 · 为什么需要这份文档

两阶段训练方案把我们的深度研究 agent 从
**GAIA pass@1 = 0.55 提升到 0.68**（pass@4 从 0.78 提到 0.85）。paper 里
我们必须回答一个具体问题：

> **模型的*行为*发生了什么变化，导致了这个收益？**

我们严格剔除"非模型行为"的因素（context 长度、数据配比、断点选择）。
这些是必要的脚手架，本文档专注讲 *policy 本身* 学会了做什么不一样的事。

分析基于一条连续训练时间轴：41 个均匀采样的 checkpoint（S1 step 0–220
取 12 个，S2 step 0–570 取 29 个），每个 checkpoint 同时配有它的训练
rollout 批次（300 条 / step）与一次 GAIA 离线评测（412 题 / step，pass@4
采样）。每个 checkpoint 我们从原始 assistant transcript 中抽 30+ 个行为指标。

把所有有噪声、或者本身是别人副产物的指标剔掉之后，**剩下四个机制是独立
有数据支撑、并且和 accuracy 曲线同步走的**。paper 应该 commit 的就是
这四个。

---

## 1 · 机制总览

| 编号 | 机制 | 单一指标 | S1 起 → S2 末 | 强度 |
|---:|------|----------|:----:|:--:|
| **M1** | 每步 tool 选择漂移：visit 比 search 更可能 | `browse_ratio` = visit/(visit+search) | 0.46 → **0.76** | ★★★ |
| **M2** | 轨迹长度的再分配：先压缩、后扩张 | `total_turns` | 28 → 19 → **45** | ★★ |
| **M3** | 单个 `<think>` 块更长（更深的推理） | `chars / <think>` | 387 → **626** | ★★ |
| **M4** | 自我纠错 token 总量上升 | `hedge_count` | 1.5 → **5.0** | ★★（频率驱动，见 §6）|

这四个机制驱动的最终结果（图 1）：

| 指标 | S1 起 | S1 220 | S2 末 |
|------|---:|---:|---:|
| Train rollout reward | 0.50 | 0.70 | 0.73 |
| **GAIA pass@1** | 0.55 | 0.59 | **0.68** |
| **GAIA pass@4** | 0.78 | 0.82 | **0.85** |

![](figures/fig1_accuracy_sim2real.png)

**图 1.** 连续训练路径上的 accuracy。pass@1 和 pass@4 都在 Stage-2 才真正
往上爬。Stage-1 step 220 的训练 reward 看起来很健康（0.70），但 GAIA
pass@1 已经卡在 0.59 — rollout reward 在 overfit，行为已经在悄悄退化。
Stage-2 即使在更换数据后训练 reward 有一段短暂的下降，最终也把真实
任务的 pass@1 又拉高了 **+9 pt**。

---

## 2 · 四机制总图

![](figures/fig2_mechanisms.png)

**图 2.** 四个行为机制画在同一条连续训练时间轴上。垂直虚线 = ckpt-220
splice 点；浅蓝 / 浅红背景 = S1 / S2 区域。

* **(a) M1 · Tool 选择漂移**：`browse_ratio` 从 0.46 单调上升到 0.76，
  rollout 曲线和 GAIA 曲线紧紧跟着。S2 末期，4 次 tool call 里有 3 次
  是深度页面阅读，不是 top-k 搜索。
* **(b) M2 · 轨迹再分配**：S1 把 turn 从 28 压到 19 —— 是个 *假经济*，
  和 pass@1 静默卡住高度相关。S2 把 turn 重新扩张到 ~45 ，并且
  *把多出来的预算花在 browse 上*。
* **(c) M3 · 单 think 推理深度**：每个 turn 的 `<think>` 数量基本不变
  （~0.46），但 *每个 think 块的字符数* 从 ~390 涨到 ~620（S2 后段）。
  单次推理写得更多。
* **(d) M4 · 自我纠错总量**：每条轨迹的 hedge 词
  （`wait` / `actually` / `let me re-check` / `but ` / `however`
  / `等等` / `不对`）从 ~1.5 涨到 ~5.0。

这四个机制不是完全独立的 —— M3、M4 部分由 M1、M2 驱动（turn 多了所以
think 多；think 多了所以 hedge 多）。但它们对应四个 *可分别测量* 的
现象，组合起来构成一个连贯的因果故事（§7）。

---

## 3 · 机制 1 · Tool 选择漂移（search → browse）

### 测量方式

每个 assistant turn 要么不发起 tool call，要么发起：
`<tool_call>{"name": "search", "arguments": …}</tool_call>` 或
`<tool_call>{"name": "visit", …}</tool_call>`（极少量的 `google_scholar`
和 `python` 归到 "other" 桶丢掉）。

定义
```
browse_ratio  =  num_visit  /  (num_visit + num_search)
```
也就是 "一次 tool call 选择 visit 而非 search 的经验概率"。同时报告归一化
的 `visits_per_turn` 和 `searches_per_turn`（图 3）。

### 观察到的变化

| 指标 | S1 step 1 | S1 step 220 | S2 step 1 | S2 step 570 |
|---|---:|---:|---:|---:|
| `num_search` | 7.77 | 3.98 | 9.97 | 4.97 |
| `num_visit` | 5.46 | 4.62 | 7.45 | **16.36** |
| `browse_ratio` | 0.463 | 0.556 | 0.500 | **0.763** |
| `visits / turn` | 0.193 | 0.241 | 0.203 | **0.367** |
| `searches / turn` | 0.275 | 0.207 | 0.271 | **0.111** |

![](figures/fig3_normalized_rates.png)

**图 3.** 归一化的 per-turn 比率。(a) 每 turn browse 概率近似翻倍；
(b) 每 turn search 概率近似减半。它们是同一个 "per-step 选择概率漂移"
的两个互补面。

### 为什么这件事重要

深度研究任务的瓶颈很少是 "找不到候选 URL"，而是 "页面里到底写了什么"
（表格、脚注、限定句）。Stage-1 的 policy 在不确定的时候倾向 *"再发一条
搜索请求"*；Stage-2 的 policy 学到的是 *"再打开一个页面读读"*。
这是 tool 使用 *哲学层面* 的质变，不是数值的微调。

这也是我们手上 *最干净、最单调* 的信号 —— GAIA 评测曲线在 panel (a)
里几乎和 rollout 曲线重合，行为在分布外能稳定迁移。

---

## 4 · 机制 2 · 轨迹再分配（先压缩、后扩张）

### 观察到的变化

| 阶段 | 平均 `total_turns` | 备注 |
|---|---:|---|
| S1 step 1（warm start） | 28.3 | base model 自带的探索 |
| S1 step 220（S1 最佳） | 19.2 | **被压缩** —— 减少 32% |
| S2 step 1（splice 后） | 36.8 | 更长 context 下 base model 的扩张 |
| S2 step 570（终点） | 44.6 | **再扩张** 21% |

形状是一个干净的 **U**：S1 压缩、S2 扩张。

### 为什么 S1 的压缩是 *坏事*

S1 的压缩看起来像在赚 reward —— 输出短了、当下 rollout 分数稍高。但
GAIA pass@1（图 1）暴露了代价：大约从 step 200 起，模型在通过
*截短证据搜索* 来赢训练奖励 —— 调用更少 tool、更早放弃、对手上有限的
证据 over-commit。step 530–750 的最终崩溃是这个趋势的延续（详细失败
模式见上游 `BEHAVIOR_EVOLUTION_zh.md` §3）。

### 为什么 S2 的扩张是 *好事*

S2 把 turn 翻倍，但 *多出来的 turn 全都是 browse*（M1）。S2 step 570
平均一条轨迹 16 次 visit、5 次 search ；对比 S1 step 220 的 4.6 visit、
4.0 search。多出来的预算被花在 **读更多页面** 上，而不是发更多请求。

这就是"再分配"的含义：Stage-2 不只是把轨迹拉长，而是 *改变了多出来的
turn 用来干什么*。

---

## 5 · 机制 3 · 单 think 块推理深度增长

### 测量方式

每条轨迹我们数 `<think>` 块的个数 `num_think` 和总字符数 `think_chars`。
派生指标 `chars / <think>` = 单个推理块的平均长度。

| 指标 | S1 step 1 | S1 step 220 | S2 step 570 |
|---|---:|---:|---:|
| `num_think` | 13.00 | 8.59 | **21.26** |
| `think_chars` | 5033 | 3413 | **13316** |
| `chars / <think>` | 387 | 397 | **626**（+58%）|
| `think_density` = think_chars / output_chars | 0.092 | 0.089 | **0.142** |

### 一个微妙但关键的区分

直觉上的说法 "模型更频繁地 think 了" —— 是错的。
`<think> / turn` 在整个训练过程中几乎不变（0.46 → 0.48）。模型从一开始
就在几乎每次 tool call 前面插 think 块。

真正变化的是 *每个 think 块的长度*。S2 后期的 `<think>` 块里能明显看到
更多内容：列举假设、衡量证据、规划下一次 browse 的目标、对之前的承诺
做修正。

结合 M2（更多 turn），整体效果是 *每条轨迹的总推理输出量* 涨到约 **4×**
（3.4k → 13.3k 字符），且推理在全部输出里的占比从 9% 涨到 14%。

### 解释

`<think>` 是 free-form 推理通道：更长的块意味着模型在
*每一份证据上花更多算力*。S2 policy 把每个抓回来的页面都当作要仔细
分析的东西，而不是一段需要赶紧过完的输入。


---

## 6 · 机制 4 · 自我纠错总量增长

### 测量方式

我们在 `<think>` 块内统计 "hedge / 修正" 词：`wait`、`actually`、
`let me re-check`、`hmm`、`but `、`however`、`reconsider`、
`on second thought`，以及它们的中文等价物（`等等`、`不对`、`重新` 等）。
这些是 **commit-and-revise** 推理的显式标记：模型提出一个答案或一步操作，
立刻退回去质疑它。

| 指标 | S1 step 1 | S1 step 220 | S2 step 570 | 比例 |
|---|---:|---:|---:|---:|
| `hedge_count`（每轨迹）| 2.58 | 1.47 | **5.03** | **3.4×** |
| `hedge / <think>` | 0.199 | 0.171 | 0.236 | 1.4× |
| `hedge / 1k think 字符` | 0.513 | 0.430 | 0.377 | 0.7× |

### 诚实的解读

两件事同时成立：

1. **自我纠错 token 的绝对量大幅增长（3.4×）。** S2 末期的典型轨迹有
   ~5 个显式修正点。读 transcript 的人会明显感到模型 "更频繁地质疑
   自己"。
2. **每个 think 块每千字符的 hedge 密度其实大体稳定**（甚至略降）。
   所以 M4 主要是 *频率侧* 现象 —— 它跟着 M2 × M3 联合放大的总推理量
   一起扩大，并不是 policy 学到的新行为。

我们仍然把 M4 列为机制，原因是它是 S2 后期轨迹的 *读者直观体验* —— 模型
产生的可见自我纠错量大幅增加，正是这一点让 trajectory 读起来更
"深思熟虑"，即使 per-char 密度并没漂移。诚实的 paper 表述应该是：

> 自我纠错 token 在 Stage-2 末期每条轨迹被发出的频次约为 3 倍，
> 这一现象由总推理量（M2 + M3）的联合扩张驱动；单个 think 块内的
> hedging 比率本身基本保持稳定。

这才是数据支撑的说法。我们刻意 *不* 声称模型学到了新的 "反思技能" ——
数据没有支持这种更强的 claim。

---

## 7 · 这四个机制如何共同产生 accuracy 收益

四个机制组合成一个单一的因果故事：

```
  M2: 更多 turn
        ×
  M1: 每个 turn 更可能是 browse
        ↓
  每条轨迹拿到更多证据（4.6 → 16.4 次 visit）
        ↓
  M3: 每个 <think> 块更长
        ↓
  每份证据被更深地综合
        ↓
  M4: 更多显式的 commit / 修正循环
        ↓
  Train rollout reward  0.50 → 0.73
  GAIA pass@1            0.55 → 0.68
  GAIA pass@4            0.78 → 0.85
```

一句话：**多读页面，每页都仔细想。** 整个文档其余部分都是对这八个字
的细颗粒度量化。

按时间序列里的信号强度排序，主导次序是
**M1 > M2 > M3 > M4**。

---

## 8 · 被我们考虑过但剔除的机制

为了透明，下面列出我们测试过、但最终决定不写进 paper 的候选机制：

| 被剔除的 claim | 为什么不成立 |
|----------------|--------------|
| "并发搜索增加（每 call 更多 query）" | `queries_per_search_call` 只从 3.06 涨到 3.58，并且非单调；不够独立为一条机制。 |
| "模型学会每 turn 更频繁思考" | `<think> / turn` 基本不变（0.45 → 0.48）。推理块插入频率从 S1 step 1 起就已经饱和。 |
| "中段 turn 学会了思考" | 两个 stage 中段 turn 都已经 ~100% 有 think。唯一的 per-turn `<think>` 差异在前 ~5 个 turn（head bucket: S1 50% → S2 78%），比 M3 要弱。 |
| "带引号的精确短语 query 变多" | `quoted_query_frac` 0.46 → 0.50，基本持平。 |
| "每 think 的纠错密度增长" | `hedge / 1k think 字符` 平甚至略降。只有绝对量增长（见 M4 注脚）。|
| "模型发明新工具" | `num_other_tools` 和 `num_python` 全程为 0。工具集是 *写死的*，不是学出来的。 |

正是这些 ablation 决定了 headline 图（图 2）恰好是四个 panel，不是
八个。

---

## 9 · 复现说明

### 数据来源

* **训练 rollout（S1）：**
  `/share/project/wanli/Search_Agent/verl/rollout_trajectory/qwen3_deepresearch_tis_rl_onpolicy_bs128_local_rag_only_token-mean-seq-mean-temp_0.7_length_32k_nokl/<时间戳>/<step>.jsonl`
* **训练 rollout（S2）：**
  `/share/project/wanli/Search_Agent/verl/rollout_trajectory/qwen3_deepresearch_tis_rl_stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k/<时间戳>/<step>.jsonl`
* **GAIA bench（S1）：**
  `/share/project/wanli/Search_Agent/DeepResearch/bench_results/qwen3-4B-RL/onpolicy_bs128_local_rag_only_token-mean-seq-mean-temp_0.7_length_32k_nokl/global_step_<N>/`
* **GAIA bench（S2）：**
  `/share/project/wanli/Search_Agent/DeepResearch/bench_results/qwen3-4B-RL/stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k/global_step_<N>/`

### 指标抽取

所有指标都通过对 `<think>…</think>` / `<tool_call>…</tool_call>` /
`<tool_response>…</tool_response>` / `<answer>…</answer>` 做简单的正则
解析得到。完整抽取器在
`../behavior_analysis/extract_behaviors.py`，每个 checkpoint 的输出缓存
在 `data/behavior_timeline.json`（41 个 checkpoint × 30+ 指标）。

### 图复现

```bash
python3 make_paper_figures.py
```

生成所有四张图的 PNG（160 dpi）和 PDF。

### GAIA-eval 的一个限制

GAIA 的 bench result 文件把 assistant 回复存成 `{"role": "assistant",
"content": …}`，其中 `<think>` 块 **被剥掉**（只保留 `<answer>` 和
`<tool_call>`）。所以 `<think>` 衍生的指标（M3 chars/think、M4
hedge_count、think_density）只能在训练 rollout 侧报。Tool 使用相关的
指标（M1、M2）两侧都有，并且它们之间高度同步（图 2 panel (a)、图 3
panel (a)(b)） —— 这本身就是一个 sanity check，说明仅有训练侧的指标
不是 simulator artifact。

---

## 10 · paper 可直接复用的段落

3–4 句、可直接放进 experiments / analysis section 的描述：

> 我们在两阶段训练 41 个均匀采样的 checkpoint 上跟踪 30 个行为指标，
> 发现 Sim2Real 的收益由 4 个 *与 GAIA pass@1 曲线协变* 的行为机制
> 解释。**(M1)** Per-step 的 tool 选择从 `search` 漂移到 `visit`：
> 经验 browse ratio 从 0.46 单调上升到 0.76。**(M2)** 轨迹长度的再
> 分配：Stage-1 在 pass@1 静默 plateau 的同时把 turn 从 28 压缩到 19，
> Stage-2 又把它扩张到 45，*并把多出来的预算用在页面阅读上而非更多
> 查询*。**(M3)** 单 `<think>` 块的深度增长：每个 think 从 ~390 字
> 增长到 ~626 字，而插入频率几乎不变，使得推理在输出中的占比从 9%
> 涨到 14%。**(M4)** 每条轨迹的显式自我纠错 token 增长 3.4 倍
> （`wait` / `actually` / `let me re-check` / `等等` …），这一现象由
> M2 与 M3 的联合扩张驱动，而非来自每 think 的 hedging 比率提升。
> 这四个机制共同产生了 GAIA 上 +9 pt pass@1 / +3 pt pass@4 的可观察
> 提升，其中 M1 和 M2 是主导因素。

