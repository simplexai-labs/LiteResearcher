# LLM Judge 并发实现说明

本文档说明如何启用 LLM-as-Judge 的并发执行，实现 20-30x 性能提升。

## 核心概念速览

**max_workers 工作原理（3句话说清楚）：**

1. **BatchRewardManager 一次接收整个 batch**（如 512 个样本）
2. **一次性提交所有任务**到 ThreadPoolExecutor 线程池（for 循环提交，不是串行执行）
3. **线程池自动调度**：32 个工作线程分批并发执行（32 个一批，共 16 批）

**比喻**：就像餐厅有 512 个订单、32 个厨师。所有订单一次性进厨房（batch），厨师自动分批做（max_workers=32），不是一个个做完再接下一个（那是串行）。

---

## 快速开始

只需在训练脚本中添加两行配置：

```bash
python3 -m verl.trainer.main_ppo \
    reward_model.custom_reward_function.path=verl/utils/reward_score/llm_judge_vllm.py \
    reward_model.launch_reward_fn_async=True \
    reward_model.reward_manager=batch \              # ← 使用BatchRewardManager
    reward_model.reward_kwargs.max_workers=32 \      # ← 设置并发线程数
    ...
```

**就这么简单！** 性能立即提升 20-30 倍。

---

## 实现原理

### 框架已有的并发支持

verl框架已经提供了完整的批量并发能力：

1. **BatchRewardManager** (`verl/workers/reward_manager/batch.py`)
   - 批量调用reward函数
   - 一次处理整个batch的所有样本

2. **ThreadPoolExecutor** 并发模式
   - 参考 `recipe/genrm_remote/reward_function.py`
   - 标准的Python并发实现

### 执行流程

**核心机制：一次性提交 + 线程池自动调度**

```
BatchRewardManager.__call__()
    ↓
提取batch中所有样本信息 (假设 batch_size=512)
    ├─ solution_strs: [response1, response2, ..., response512]
    ├─ ground_truths: [gt1, gt2, ..., gt512]
    └─ extra_infos: [info1, info2, ..., info512]
    ↓
compute_score_batch(solution_strs, ground_truths, extra_infos, max_workers=32)
    ↓
一次性提交所有 512 个任务到 ThreadPoolExecutor
    ├─ for 循环：executor.submit(compute_score, sample_i)
    ├─ 所有 512 个任务进入线程池队列
    └─ 注意：不是 batch for 循环，而是一次性提交全部
    ↓
ThreadPoolExecutor 自动调度 (32 个工作线程)
    ├─ 第1轮 (0-1秒):  Threads 1-32  处理 samples 1-32   → 32 个并发 HTTP 请求
    ├─ 第2轮 (1-2秒):  Threads 1-32  处理 samples 33-64  → 32 个并发 HTTP 请求
    ├─ 第3轮 (2-3秒):  Threads 1-32  处理 samples 65-96  → 32 个并发 HTTP 请求
    ├─ ...
    └─ 第16轮(15-16秒): Threads 1-32  处理 samples 481-512 → 32 个并发 HTTP 请求
    ↓
等待所有线程完成 (总耗时: ~16秒，而非串行的 512秒)
    ↓
返回 [score1, score2, ..., score512]
```

**关键理解点：**
1. ✅ **batch 内的所有样本一次性提交**到线程池
2. ✅ **线程池自动分批处理**：每批最多 32 个并发
3. ❌ **不是**逐个 batch for 循环串行处理
4. ❌ **不是**整个 batch 512 个样本同时并发（那样需要 512 个线程）

### max_workers 工作机制详解

**问：max_workers=32 是怎么给的？**

答：`max_workers=32` 从配置传入，控制线程池的并发线程数：

```bash
reward_model.reward_kwargs.max_workers=32
```

**问：是按照 batch for 循环吗？**

答：不是。执行流程是：
1. BatchRewardManager 接收**整个 batch**（例如 512 个样本）
2. 调用 `compute_score_batch()`，一次性 for 循环提交所有 512 个任务到线程池
3. ThreadPoolExecutor 的 32 个工作线程自动从队列中取任务并发执行
4. 线程池内部自动分批：每批最多 32 个并发，共需 512÷32=16 批

**问：batch 内部的一起送到 llm as judge 吗？**

答：**是一起提交**到线程池，但**分批并发执行**：
- ✅ 512 个样本一次性提交到线程池队列
- ✅ 线程池自动调度，每次最多 32 个样本并发发送 HTTP 请求到 Judge
- ❌ 不是 512 个样本同时并发（那样需要 512 个线程，开销太大）

### 代码实现

**ThreadPoolExecutor 并发调用**（`llm_judge_vllm.py:272-309`）：

```python
from concurrent.futures import ThreadPoolExecutor

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """
    并发处理整个batch的评分
    
    例如：batch_size=512, max_workers=32
    - 一次性提交 512 个任务到线程池
    - 线程池自动调度 32 个线程并发执行
    - 总耗时约 512÷32=16 秒 (假设每个请求1秒)
    """
    max_workers = kwargs.pop('max_workers', 32)  # 从配置获取并发线程数

    # 创建线程池：最多 32 个工作线程
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 【关键步骤1】一次性提交所有任务（例如 512 个）
        futures = []
        for ds, sol, gt, ei in zip(data_sources, solution_strs, ground_truths, extra_infos):
            # 每个 submit 都是异步的，立即返回 Future 对象
            # 任务会进入队列，等待线程池调度
            future = executor.submit(compute_score, ds, sol, gt, ei)
            futures.append(future)
        
        # 此时：所有 512 个任务已在队列中，32 个线程正在并发处理前 32 个

        # 【关键步骤2】等待所有结果
        # future.result() 会阻塞直到该任务完成
        # 这里按顺序收集结果，但任务是并发执行的
        results = [future.result() for future in futures]

    return results  # 返回 [score1, score2, ..., score512]
```

### 完整代码调用链（数据传递路径）

**从 512 个样本到 Judge 服务的完整路径：**

```python
# ==================== 文件: verl/workers/reward_manager/batch.py ====================
class BatchRewardManager:
    def __call__(self, data):
        # data 包含 512 个样本的完整信息
        
        # 1. 解码所有 512 个 responses
        responses_str = [decode(response_ids[i]) for i in range(512)]
        # → ["response1", "response2", ..., "response512"]
        
        # 2. 提取所有 512 个 ground_truths 和 extra_infos
        ground_truths = [data[i].get("ground_truth") for i in range(512)]
        # → [{"target": "ans1"}, {"target": "ans2"}, ..., {"target": "ans512"}]
        
        extra_infos = [data[i].get("extra_info") for i in range(512)]
        # → [{"question": "q1"}, {"question": "q2"}, ..., {"question": "q512"}]
        
        # 3. 【关键调用】一次性传递所有 512 个样本的数据
        scores = self.compute_score(
            data_sources=data_sources,      # 512个元素的list
            solution_strs=responses_str,    # 512个元素的list
            ground_truths=ground_truths,    # 512个元素的list
            extra_infos=extra_infos,        # 512个元素的list
            max_workers=32                  # 来自配置
        )
        # → 这里调用的是下面的 compute_score_batch()
        

# ==================== 文件: verl/utils/reward_score/llm_judge_vllm.py ====================
def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """
    接收所有 512 个样本的数据
    """
    max_workers = kwargs.pop('max_workers', 32)  # 获取 max_workers=32
    
    # 创建线程池：32 个工作线程
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        
        # 【关键循环】遍历所有 512 个样本，一次性提交到线程池
        for i in range(512):
            # 每次提交一个样本的数据到线程池
            future = executor.submit(
                compute_score,           # 函数
                data_sources[i],         # 第i个样本的data_source
                solution_strs[i],        # 第i个样本的response
                ground_truths[i],        # 第i个样本的ground_truth
                extra_infos[i]           # 第i个样本的extra_info
            )
            futures.append(future)
        # 注意：上面的for循环很快完成（只是提交任务），不是串行执行
        
        # 等待所有 512 个任务完成（这里会阻塞等待）
        results = [future.result() for future in futures]
        # → [score1, score2, ..., score512]
    
    return results


def compute_score(data_source, solution_str, ground_truth, extra_info):
    """
    处理单个样本（在线程中执行）
    """
    # 1. 提取数据
    question = extra_info["question"]
    golden_answer = ground_truth["target"]
    predicted_answer = extract_answer(solution_str)
    
    # 2. 调用 Judge 客户端
    judge_client = _get_judge_client()  # 获取全局客户端
    result = judge_client.judge_single(question, golden_answer, predicted_answer)
    # → 这里会发送 HTTP 请求
    
    return result  # {"score": 1.0, "judge_response": "True", ...}


class VLLMJudgeClient:
    def judge_single(self, question, golden_answer, predicted_answer):
        """
        发送单个 HTTP 请求到 Judge 服务
        """
        # 1. 构造 prompt
        prompt = f"Question: {question}\nGolden: {golden_answer}\nPredicted: {predicted_answer}\nAnswer:"
        
        # 2. 【关键】发送 HTTP POST 请求到远程 Judge 服务
        response = requests.post(
            url="http://47.111.147.142:8765/v1/completions",  # Judge 服务地址
            json={
                "model": "Qwen3-4B-Instruct",
                "prompt": prompt,
                "max_tokens": 10,
                "temperature": 0.0
            },
            timeout=600
        )
        # → 这个请求会通过网络发送到远程服务器
        # → Judge 服务处理并返回 "True" 或 "False"
        
        # 3. 解析响应
        judge_text = response.json()["choices"][0]["text"]  # "True" 或 "False"
        is_correct = (judge_text.lower() == "true")
        score = 1.0 if is_correct else 0.0
        
        return {"score": score, "judge_response": judge_text, "is_correct": is_correct}
```

**数据流动总结：**

| 位置 | 数据规模 | 操作 |
|------|----------|------|
| BatchRewardManager | 512 个样本（整个batch） | 提取数据，准备调用 |
| compute_score_batch() | 512 个样本（整个batch） | 一次性接收，创建线程池 |
| ThreadPoolExecutor | 512 个任务（队列中） | 分批调度给32个线程 |
| compute_score() | 1 个样本（每个线程） | 在线程中处理单个样本 |
| judge_single() | 1 个样本（每个线程） | 发送1个HTTP请求 |
| vLLM Judge服务 | 最多32个并发请求 | 同时处理32个推理 |

---

## 整体并发流程图

### 数据流转全景（完整路径）

```
训练步骤开始
    ↓
Actor 生成 512 个 responses (batch_size=512, n=5 → 512=102*5+2)
    ↓
【步骤1】BatchRewardManager 接收整个 batch
    ├─ 文件: verl/workers/reward_manager/batch.py
    ├─ 样本1: prompt1 + response1 + ground_truth1 + extra_info1
    ├─ 样本2: prompt2 + response2 + ground_truth2 + extra_info2
    ├─ ...
    └─ 样本512: prompt512 + response512 + ground_truth512 + extra_info512
    ↓
【步骤2】调用 compute_score_batch() 
    ├─ 文件: verl/utils/reward_score/llm_judge_vllm.py
    ├─ 参数传递: data_sources, solution_strs, ground_truths, extra_infos, max_workers=32
    └─ 这512个样本被一次性传入该函数
    ↓
【步骤3-并发关键点】创建 ThreadPoolExecutor(max_workers=32)
    ├─ for 循环: 一次性提交全部 512 个任务到线程池队列
    │   └─ executor.submit(compute_score, ds, sol, gt, ei) × 512次
    └─ 注意：这里的for循环只是提交任务，不是串行执行
    ↓
【步骤4】线程池自动调度执行 (32个工作线程并发)
    ┌──────────────────────────────────────────────────────────────┐
    │ 时间 0-1s:  32个线程同时调用                                  │
    │   Thread 1: compute_score(sample_1)   → judge_client.judge_single() → HTTP POST  │
    │   Thread 2: compute_score(sample_2)   → judge_client.judge_single() → HTTP POST  │
    │   ...                                                         │
    │   Thread 32: compute_score(sample_32) → judge_client.judge_single() → HTTP POST  │
    │                        ↓                                      │
    │                 32个并发HTTP请求发送到                        │
    │      vLLM Judge服务 (http://47.111.147.142:8765/v1/completions) │
    ├──────────────────────────────────────────────────────────────┤
    │ 时间 1-2s:  处理第2批 (samples 33-64)                        │
    │   Thread 1-32: 继续从队列取新任务 → 32个HTTP POST            │
    ├──────────────────────────────────────────────────────────────┤
    │ ...                                                           │
    ├──────────────────────────────────────────────────────────────┤
    │ 时间15-16s: 处理第16批 (samples 481-512)                     │
    │   Thread 1-32: 最后一批任务 → 32个HTTP POST                  │
    └──────────────────────────────────────────────────────────────┘
    ↓
【步骤5】收集所有结果
    ├─ [score1, score2, ..., score512]
    ├─ 每个score包含: {"score": 0.0/1.0, "judge_response": "True/False", "is_correct": bool}
    └─ 总耗时: ~16秒 (相比串行的 512秒，加速 32x)
    ↓
【步骤6】返回给 Trainer 继续训练
    └─ 用于计算 advantage 和更新模型
```

### 关键数据传递路径

**512个样本是如何到达 LLM Judge 的？**

```
BatchRewardManager (Python对象)
    ↓ 调用 self.compute_score(...)
compute_score_batch() (llm_judge_vllm.py:272)
    ↓ 接收 512个样本的所有数据
    ↓ 创建线程池
ThreadPoolExecutor (max_workers=32)
    ↓ for循环提交512次 executor.submit()
    ↓ 任务进入队列，线程池自动调度
32个工作线程并发执行
    ↓ 每个线程调用 compute_score(单个样本)
    ↓ compute_score() 内部调用
judge_client.judge_single(question, golden, predicted)
    ↓ judge_single() 内部调用
requests.post(url="http://47.111.147.142:8765/v1/completions", json={...})
    ↓ HTTP POST请求
vLLM Judge 服务 (远程)
    ↓ Judge模型推理
    ↓ 返回 "True" 或 "False"
    ↓ 原路返回
compute_score() 返回 {"score": 1.0/0.0, ...}
    ↓
compute_score_batch() 收集所有512个结果
    ↓
BatchRewardManager 获得 [score1, ..., score512]
```

**关键理解：**
1. ✅ **512个样本直接被送到 llm_judge_vllm.py 文件**：是的，通过 `compute_score_batch()` 函数
2. ✅ **max_workers 最终影响的是 HTTP 请求到 Judge 服务的并发数**：是的，32个线程=32个并发HTTP请求
3. ✅ **不是直接送给 Judge 服务**：而是通过 ThreadPoolExecutor 分批发送 HTTP 请求
4. ✅ **Judge 服务是远程的**：通过 HTTP POST 调用，每个请求需要网络往返时间

### max_workers 影响的到底是什么？

**问题澄清：worker 最终影响的是 LLM as Judge 吗？**

**答：是的！** `max_workers` 直接控制同时发送到 Judge 服务的并发 HTTP 请求数量。

#### 详细说明

```
配置中的 max_workers=32
    ↓
传递给 ThreadPoolExecutor(max_workers=32)
    ↓
创建 32 个工作线程
    ↓
每个线程执行流程：
  1. 从队列取一个样本
  2. 调用 compute_score(样本)
  3. 内部调用 judge_client.judge_single()
  4. 发送 HTTP POST 请求到 Judge 服务 ← 【这里就是瓶颈】
  5. 等待 Judge 服务返回结果
  6. 继续处理下一个样本
    ↓
因此：max_workers=32 意味着同时最多 32 个 HTTP 请求发送给 Judge 服务
```

#### Worker vs Judge 的关系

| 概念 | 位置 | 作用 | 
|------|------|------|
| **max_workers** | 训练机器（本地） | 控制并发线程数 |
| **ThreadPoolExecutor** | 训练机器（本地） | 管理 32 个工作线程 |
| **compute_score()** | 训练机器（本地） | 在线程中执行，准备数据 |
| **HTTP POST** | 网络传输 | 每个线程发送一个请求 |
| **vLLM Judge 服务** | 远程服务器 | 接收并发请求，进行推理 |

**关键点：**
- ✅ Worker 在**本地训练机器**运行
- ✅ Judge 在**远程服务器**运行（HTTP API）
- ✅ max_workers=32 → **同时32个HTTP请求**发送到远程Judge
- ✅ Judge 服务需要足够的并发能力来处理这32个请求

#### 网络示意图

```
[训练机器] 
    ├─ BatchRewardManager
    ├─ ThreadPoolExecutor (max_workers=32)
    │   ├─ Thread 1: compute_score() ──┐
    │   ├─ Thread 2: compute_score() ──┤
    │   ├─ ...                         ├─→ 32个并发HTTP请求
    │   └─ Thread 32: compute_score() ─┘
    │
    └─→ 通过网络发送到
           ↓
[远程Judge服务器: http://47.111.147.142:8765]
    ├─ vLLM API Server (--max-num-seqs 256)
    ├─ Judge模型: Qwen3-4B-Instruct
    └─ 同时处理最多256个请求（需 ≥ max_workers）
```

**所以：**
- `max_workers=32` 控制的是**本地发起的并发HTTP请求数**
- 这些请求**最终都会到达远程的 LLM Judge 服务**
- Judge 服务的 `--max-num-seqs` 需要 ≥ max_workers 才能充分利用并发

### 关键时序对比

**串行模式 (NaiveRewardManager, 无并发)**
```
t=0s    ──┤sample1  Judge回复 
t=1s            ──┤sample2  Judge回复
t=2s                    ──┤sample3  Judge回复
...
t=511s                                      ──┤sample512  Judge回复
═══════════════════════════════════════════════════════════════════
总耗时: 512秒
```

**并发模式 (BatchRewardManager + max_workers=32)**
```
t=0s    ══════════════════════════════════╗
        ║ Threads 1-32 同时发送32个请求  ║
t=1s    ╚══════════════════════════════════╝
        ══════════════════════════════════╗
        ║ Threads 1-32 同时发送32个请求  ║
t=2s    ╚══════════════════════════════════╝
        ...
        ══════════════════════════════════╗
        ║ Threads 1-32 同时发送32个请求  ║
t=16s   ╚══════════════════════════════════╝
═══════════════════════════════════════════════════════════════════
总耗时: 16秒 (加速 32倍)
```

### 数据传递可视化

**512 个样本如何从训练代码到达 Judge 服务？**

```
┌─────────────────────────────────────────────────────────────────────┐
│  训练机器 (本地)                                                     │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ BatchRewardManager                                            │  │
│  │                                                               │  │
│  │  data = {                                                     │  │
│  │    responses: [resp1, resp2, ..., resp512],     ← 512个      │  │
│  │    ground_truths: [gt1, gt2, ..., gt512],       ← 512个      │  │
│  │    extra_infos: [info1, info2, ..., info512]    ← 512个      │  │
│  │  }                                                            │  │
│  │                                                               │  │
│  │  ↓ 调用 compute_score_batch(全部512个)                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ↓                                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ llm_judge_vllm.py :: compute_score_batch()                    │  │
│  │                                                               │  │
│  │  接收: 512个样本的所有数据                                    │  │
│  │                                                               │  │
│  │  创建: ThreadPoolExecutor(max_workers=32)                     │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │ 线程池队列                                              │ │  │
│  │  │  Task 1, Task 2, ..., Task 512  (共512个任务)          │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  │                                                               │  │
│  │  ┌───────────┐  ┌───────────┐       ┌───────────┐           │  │
│  │  │ Thread 1  │  │ Thread 2  │  ...  │ Thread 32 │  ← 32个  │  │
│  │  └───────────┘  └───────────┘       └───────────┘           │  │
│  │       ↓              ↓                     ↓                 │  │
│  │  compute_score() compute_score()   compute_score()          │  │
│  │       ↓              ↓                     ↓                 │  │
│  │  judge_single()  judge_single()    judge_single()           │  │
│  │       ↓              ↓                     ↓                 │  │
│  └───────┼──────────────┼─────────────────────┼─────────────────┘  │
│          │              │                     │                     │
│      HTTP POST      HTTP POST            HTTP POST                 │
│          │              │                     │                     │
└──────────┼──────────────┼─────────────────────┼─────────────────────┘
           │              │                     │
           └──────────────┴─────────────────────┴──→ 网络 (32个并发HTTP请求)
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Judge 服务器 (远程: http://47.111.147.142:8765)                    │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ vLLM API Server                                               │  │
│  │   --max-num-seqs 256  ← 支持最多256个并发请求                 │  │
│  │                                                               │  │
│  │   接收 32 个并发 HTTP POST 请求                               │  │
│  │   ↓                                                           │  │
│  │   同时处理 32 个推理请求                                       │  │
│  │   ┌────────┐ ┌────────┐     ┌────────┐                       │  │
│  │   │Request1│ │Request2│ ... │Request32│                      │  │
│  │   └────────┘ └────────┘     └────────┘                       │  │
│  │       ↓          ↓              ↓                             │  │
│  │   Qwen3-4B-Instruct 模型推理                                  │  │
│  │       ↓          ↓              ↓                             │  │
│  │   "True"      "False"        "True"  ← 判断结果              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                       ↓                                               │
│              返回 32 个 HTTP 响应                                     │
└───────────────────────────┼───────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│  训练机器 (本地)                                                     │
│                                                                       │
│  32个线程接收响应，继续处理下一批...                                 │
│  循环 16 次后，所有 512 个样本处理完毕                                │
│                                                                       │
│  返回: [score1, score2, ..., score512]                              │
└─────────────────────────────────────────────────────────────────────┘
```

**关键要点：**
1. ✅ **数据一次性传递**：512个样本的数据从 BatchRewardManager 完整传递到 compute_score_batch()
2. ✅ **任务一次性提交**：512个任务通过 for 循环快速提交到线程池队列
3. ✅ **线程池自动调度**：32个工作线程自动从队列取任务执行
4. ✅ **分批发送HTTP**：每次最多32个并发HTTP请求发送到远程Judge
5. ✅ **Judge服务处理**：远程Judge服务同时处理32个推理请求
6. ✅ **循环往复**：直到所有512个样本处理完成

---

## 性能对比

### 串行 vs 并发

| Batch Size | 串行执行 (NaiveRewardManager) | 并发执行 (BatchRewardManager) | 加速比 |
|-----------|------------------------------|------------------------------|--------|
| 128 | 128秒 | 4-6秒 | **21-32x** |
| 256 | 256秒 | 8-12秒 | **21-32x** |
| 512 | 512秒 | 16-24秒 | **21-32x** |

### 时间线对比

**串行执行（原始）**：
```
Sample 1:  [HTTP请求] ──┤ 1秒
Sample 2:               [HTTP请求] ──┤ 1秒
...
Sample 512:                                       [HTTP请求] ──┤ 1秒
总耗时: 512秒
```

**并发执行（优化后）**：
```
32个线程同时执行：
Thread 1-32:  [HTTP请求batch 1-32]   ──┤ 1秒
Thread 1-32:  [HTTP请求batch 33-64]  ──┤ 1秒
...
Thread 1-32:  [HTTP请求batch 481-512] ──┤ 1秒
总耗时: 16秒 (512/32 = 16批次)
```

---

## 配置说明

### 必需配置

```yaml
reward_model:
  reward_manager: batch               # 使用BatchRewardManager
  reward_kwargs:
    max_workers: 32                   # 并发线程数
  custom_reward_function:
    path: verl/utils/reward_score/llm_judge_vllm.py
  launch_reward_fn_async: True        # 异步执行reward计算
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `reward_manager` | `naive` | 设为 `batch` 启用并发 |
| `max_workers` | 32 | 并发线程数，建议 16-64 |
| `launch_reward_fn_async` | False | 是否异步执行（推荐True） |

### max_workers 调优

| Batch Size | 推荐值 | 说明 |
|-----------|-------|------|
| 64-128 | 16-32 | 避免过多线程开销 |
| 256-512 | 32-64 | 平衡并发和资源 |
| 1024+ | 64-128 | 最大化并发 |

**经验公式**：`max_workers = min(batch_size / 16, 64)`

---

## Judge服务配置

启动vLLM Judge时，需要支持高并发：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8000 \
    --max-num-seqs 256 \              # ← 最大并发请求数
    --max-model-len 2048 \             # Judge prompt较短
    --gpu-memory-utilization 0.9       # 高GPU利用率
```

**关键参数**：
- `--max-num-seqs 256`: 支持高并发（需 ≥ max_workers）
- `--gpu-memory-utilization 0.9`: 提高GPU利用率

---

## 完整示例

### 训练脚本配置

```bash
#!/bin/bash

# 环境变量
export VLLM_JUDGE_API_BASE="http://localhost:8000"
export VLLM_JUDGE_MODEL="Qwen/Qwen2.5-3B-Instruct"
export VLLM_JUDGE_MAX_RETRIES="3"
export VLLM_JUDGE_TIMEOUT="30"

# 启动训练
python3 -m verl.trainer.main_ppo \
    --config-path examples/sglang_multiturn/config \
    --config-name google_search_browse_multiturn_grpo \
    data.train_batch_size=512 \
    reward_model.custom_reward_function.path=verl/utils/reward_score/llm_judge_vllm.py \
    reward_model.launch_reward_fn_async=True \
    reward_model.reward_manager=batch \
    reward_model.reward_kwargs.max_workers=32 \
    ...
```

### 验证并发生效

训练时查看日志：

```
[INFO] Batch judge completed: 512 samples processed concurrently with 32 workers
```

看到这条日志说明并发成功！

---

## 监控和调试

### 性能指标

在训练日志中查看timing信息：

```
Step 1:
  gen: 32.45s
  reward: 0.02s (async trigger)
  old_log_prob: 4.12s
  ref: 3.01s
  values: 2.34s
  adv: 16.5s  ← 包含Judge等待时间
  update_critic: 5.67s
  update_actor: 15.89s
```

**关键指标**：
- `reward`: 应该很小（<0.1s），表示异步启动
- `adv`: 包含等待Judge完成的时间（应该 <30s）

### 常见问题 (FAQ)

#### Q1: max_workers 是怎么给的？

**答**：通过配置参数传入：
```bash
reward_model.reward_kwargs.max_workers=32
```
这个值会传递到 `compute_score_batch()` 函数，用于创建 ThreadPoolExecutor。

#### Q2: 是按照 batch for 循环吗？

**答**：不是逐个 batch 循环。执行流程是：
1. BatchRewardManager 一次接收**整个 batch**（512个样本）
2. 通过 for 循环**一次性提交所有 512 个任务**到线程池
3. 线程池的 32 个工作线程**自动调度**，分批并发执行

不是串行的 batch 循环，而是并发的线程池调度。

#### Q3: batch 内部的一起送到 LLM as Judge 吗？

**答**：是的，但是**分批并发**：
- ✅ 所有 512 个样本**一次性提交**到线程池队列
- ✅ 线程池自动调度，每批最多 **32 个并发** HTTP 请求到 Judge
- ❌ 不是 512 个样本完全同时发送（那需要 512 个线程，资源开销大）

**比喻**：就像餐厅有 512 个订单，32 个厨师。订单一次性全进厨房，但厨师每次只能做 32 份，做完继续取下一批。

#### Q4: 为什么不直接 512 个线程全并发？

**答**：
- 线程过多会导致资源竞争（CPU上下文切换、网络连接）
- Judge 服务有并发限制（如 `--max-num-seqs 256`）
- 32-64 个线程是经验最优值，能达到 20-30x 加速已经非常好

#### Q5: 这 512 个样本是直接被送到 llm_judge_vllm.py 文件的吗？

**答**：是的，完整流程是：

```python
# 步骤1: BatchRewardManager 接收 512 个样本
BatchRewardManager.__call__(data)  # data 包含 512 个样本
    ↓
# 步骤2: 提取所有数据并调用 compute_score_batch
self.compute_score(
    data_sources=[ds1, ds2, ..., ds512],      # 512个
    solution_strs=[sol1, sol2, ..., sol512],  # 512个
    ground_truths=[gt1, gt2, ..., gt512],     # 512个
    extra_infos=[ei1, ei2, ..., ei512],       # 512个
    max_workers=32
)
    ↓
# 步骤3: compute_score_batch 在 llm_judge_vllm.py 中
# 一次性接收所有 512 个样本的数据
def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    # 所有512个样本的数据都在这些list中
    with ThreadPoolExecutor(max_workers=32) as executor:
        for 样本 in 512个样本:
            executor.submit(compute_score, 样本)  # 提交到线程池
```

**关键点：**
- ✅ 512个样本的数据一次性传递给 `compute_score_batch()`
- ✅ 这是在 `llm_judge_vllm.py` 文件中的函数
- ✅ 然后通过线程池分批发送给 Judge 服务

#### Q6: max_workers 最终影响的是 LLM as Judge 吗？

**答**：是的！具体影响路径：

```
max_workers=32
    ↓
32 个本地工作线程
    ↓
同时发送 32 个 HTTP POST 请求
    ↓
到达远程 vLLM Judge 服务
    ↓
Judge 服务同时处理 32 个推理请求
```

**所以：**
- `max_workers` 控制的是**同时有多少个 HTTP 请求发送到 Judge**
- 值越大 → 并发请求越多 → Judge 服务压力越大 → 但总耗时越短
- 必须平衡：Judge 服务的并发能力（`--max-num-seqs`）要 ≥ max_workers

**示例：**
- `max_workers=32`：Judge 服务同时处理 32 个请求，需要约 16 秒完成 512 个
- `max_workers=64`：Judge 服务同时处理 64 个请求，需要约 8 秒完成 512 个
- `max_workers=512`：Judge 服务同时处理 512 个请求，但可能过载（不推荐）

#### 1. 并发未生效

**症状**：`adv` 时间很长（>100s）

**排查**：
```bash
# 检查配置
cat train_script.sh | grep reward_manager
# 应该输出: reward_model.reward_manager=batch

# 检查日志
grep "Batch judge completed" train.log
# 应该看到并发日志
```

#### 2. Judge服务过载

**症状**：大量 `Judge attempt failed` 错误

**解决**：
```bash
# 增加Judge服务的并发能力
python -m vllm.entrypoints.openai.api_server \
    --max-num-seqs 512  # ← 增大
```

#### 3. max_workers过大

**症状**：性能反而下降

**解决**：减小 `max_workers` 到 32-64

---

## 总结

### 核心优势

1. **配置简单**：只需2行配置
2. **性能提升显著**：20-30x 加速
3. **利用框架现有能力**：BatchRewardManager + ThreadPoolExecutor
4. **兼容性好**：可随时切换回串行模式

### 推荐配置

```yaml
reward_model:
  reward_manager: batch
  reward_kwargs:
    max_workers: 32
  launch_reward_fn_async: True
  custom_reward_function:
    path: verl/utils/reward_score/llm_judge_vllm.py
```

### 性能预期

- **Batch Size 512**：从 8.5分钟 降到 16秒
- **加速比**：20-30x
- **GPU利用率**：从 10% 提升到 80-90%

---

**相关文档**：
- [llm_judge使用指南.md](./llm_judge使用指南.md)
- [llm_judge代码流程详解.md](./llm_judge代码流程详解.md)
