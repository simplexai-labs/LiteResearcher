# Benchmark Inference Guide

## 概述

本指南说明如何使用verl对多个benchmark进行推理测试（不进行训练）。

## 已处理的Benchmarks

所有benchmarks数据已处理为verl兼容格式，保存在 `/share/project/wanli/Search_Agent/verl/data/benchmarks_processed/`：

### 目录结构

```
data/benchmarks_processed/
├── all_benchmarks_test.parquet      # 合并的所有benchmarks (2847 samples)
├── individual/                       # 单个benchmark数据
│   ├── GAIA_test.parquet           # 103 samples
│   ├── GPQA_test.parquet           # 198 samples
│   ├── HLE_test.parquet            # 500 samples
│   ├── WebWalkerQA_test.parquet    # 680 samples
│   ├── Browsecomp_test.parquet     # 1266 samples
│   └── Xbench_test.parquet         # 100 samples
└── samples/                          # JSON样本文件（可视化）
    ├── all_benchmarks_first_5_samples.json
    ├── GAIA_first_5_samples.json
    ├── GPQA_first_5_samples.json
    ├── HLE_first_5_samples.json
    ├── WebWalkerQA_first_5_samples.json
    ├── Browsecomp_first_5_samples.json
    └── Xbench_first_5_samples.json
```

### Benchmark统计

1. **GAIA** - 103 samples (GAIA dev set - 需要深度研究的复杂问题)
2. **GPQA** - 198 samples (GPQA diamond - 研究生级别的科学问题)
3. **HLE** - 500 samples (HLE test set - 长文本理解)
4. **WebWalkerQA** - 680 samples (WebWalkerQA test - 网页导航问答)
5. **Browsecomp** - 1266 samples (Browsecomp - 浏览理解)
6. **Xbench** - 100 samples (Xbench QA pairs - 通用问答)

**总计**: 2847 samples

**数据分布** (已打乱顺序，保证均匀分布)

## 数据格式

所有benchmark数据已转换为与 `searchR1_processed_with_dual_tools` 完全一致的格式：

```python
{
    "data_source": "GAIA",  # 或其他benchmark名称
    "prompt": [...],  # 包含system和user消息
    "ability": "fact-reasoning",
    "agent_name": "tool_agent",
    "reward_model": {
        "ground_truth": {"target": np.array([...])},
        "style": "rule"
    },
    "extra_info": {
        "index": 0,
        "need_tools_kwargs": True,
        "question": "...",
        "split": "test",
        "tools_kwargs": {
            "browse": {"create_kwargs": {...}},
            "search": {"create_kwargs": {...}}
        }
    },
    "metadata": None
}
```

## 使用方法

### 1. 数据预处理（已完成）

数据已经预处理完成。如需重新处理：

```bash
conda activate /share/project/wanli/env/verl-v060
cd /share/project/wanli/Search_Agent/verl

# 处理所有benchmarks
python3 examples/data_preprocess/preprocess_all_benchmarks.py

# 只处理特定benchmarks
python3 examples/data_preprocess/preprocess_all_benchmarks.py \
    --benchmarks GAIA GPQA
```

### 2. 运行推理

#### 推理合并的所有benchmarks（推荐）

```bash
conda activate /share/project/wanli/env/verl-v060
cd /share/project/wanli/Search_Agent/verl

# 推理合并数据集 (2847 samples)
./examples/sglang_multiturn/search_browser/inference_benchmarks.sh all

# 或者直接运行（默认为 all）
./examples/sglang_multiturn/search_browser/inference_benchmarks.sh
```

这将使用 `all_benchmarks_test.parquet`，包含所有6个benchmark的数据（已打乱顺序）。

#### 推理特定benchmarks

```bash
# 只推理GAIA和GPQA
./examples/sglang_multiturn/search_browser/inference_benchmarks.sh "GAIA GPQA"

# 只推理单个benchmark
./examples/sglang_multiturn/search_browser/inference_benchmarks.sh "HLE"
```

单个benchmark的数据位于 `data/benchmarks_processed/individual/` 目录。

### 3. 查看结果

推理结果保存在 `validation_trajectory/` 目录：

```bash
# 查看最新的推理结果
ls -lt validation_trajectory/ | head -10

# 查看合并数据集的结果
ls -la validation_trajectory/benchmark_inference_all_benchmarks_*/

# 查看某个单独benchmark的结果
ls -la validation_trajectory/benchmark_inference_GAIA_inference_*/
```

每次推理会生成：
- `1.jsonl` - 推理轨迹（包含input, output, score, assistant_turns, user_turns, pred_ans等）
- 对应的log文件在 `logs/` 目录

### 3.5 查看样本JSON

在运行推理前，可以查看每个benchmark的前5个样本：

```bash
# 查看所有可用的样本JSON
ls -lh data/benchmarks_processed/samples/

# 查看GAIA的样本
cat data/benchmarks_processed/samples/GAIA_first_5_samples.json | python3 -m json.tool | less

# 查看合并数据集的样本
cat data/benchmarks_processed/samples/all_benchmarks_first_5_samples.json | python3 -m json.tool | less
```

这些JSON文件包含了完整的数据结构，便于理解数据格式和内容。

### 4. 可视化轨迹

使用visualization脚本分析推理结果：

```bash
# 可视化GAIA结果
python3 scripts/visualize_trajectory.py \
    validation_trajectory/benchmark_inference_GAIA_inference_*/1.jsonl \
    -n 50 \
    -v

# 打开HTML报告
xdg-open trajectory_visualization/*GAIA*report*.html
```

## 推理脚本配置

`inference_benchmarks.sh` 的关键配置：

```bash
# 关键参数
trainer.val_before_train=True      # 训练前执行验证
trainer.total_epochs=0              # 不进行训练，只验证

# 推理配置
data.val_batch_size=128             # 验证批次大小
actor_rollout_ref.rollout.n=1      # 每个prompt只生成1个响应

# 多轮对话配置
actor_rollout_ref.rollout.multi_turn.enable=true
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10
actor_rollout_ref.rollout.multi_turn.format=hermes

# 工具配置
actor_rollout_ref.rollout.multi_turn.tool_config_path="..."
```

## 工具配置

推理使用双工具（search + browse）配置：
- **配置文件**: `examples/sglang_multiturn/config/tool_config/google_search_browse_tool_config.yaml`
- **工具后端**:
  - Google Search: `http://47.111.147.142:8864`
  - Browse: `http://47.111.147.142:8084`

确保工具后端服务正在运行：

```bash
# 检查search服务
curl http://47.111.147.142:8864/health

# 检查browse服务
curl http://47.111.147.142:8084/health
```

## 评估指标

推理完成后，trajectory文件包含以下信息：

```json
{
    "input": "system\n...\nuser\n...",
    "output": "assistant\n<think>...</think><tool_call>...</tool_call>...",
    "gts": {"target": ["正确答案"]},
    "score": 1.0,
    "step": 1,
    "assistant_turns": 3,
    "user_turns": 3,
    "pred_ans": "模型预测的答案"
}
```

**评估指标**：
- `score`: 0/1，答案是否完全匹配（exact match）
- `assistant_turns`: 模型执行的轮数
- `user_turns`: 系统响应（包括工具调用）的轮数
- `pred_ans`: 从 `<answer>` 标签中提取的答案

## 故障排查

### 1. 工具调用失败

**症状**: Trajectory中没有工具调用，或工具返回错误

**解决**:
- 检查工具后端服务是否运行
- 检查网络连接到工具服务器
- 查看log中的错误信息

### 2. OOM错误

**症状**: CUDA out of memory

**解决**:
```bash
# 减少batch size
data.val_batch_size=64

# 减少GPU memory utilization
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
```

### 3. 推理太慢

**症状**: 处理时间过长

**解决**:
```bash
# 减少max_assistant_turns
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5

# 使用更多GPU进行tensor parallel
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

## 性能估算

基于Qwen2.5-3B-Instruct模型 + 8x GPU：

| Benchmark | Samples | 预计时间 | 输出目录大小 |
|-----------|---------|----------|-------------|
| GAIA | 103 | ~15分钟 | ~50MB |
| GPQA | 198 | ~30分钟 | ~100MB |
| HLE | 500 | ~1小时 | ~250MB |
| WebWalkerQA | 680 | ~1.5小时 | ~350MB |
| Browsecomp | 1266 | ~2.5小时 | ~650MB |
| Xbench | 100 | ~15分钟 | ~50MB |
| **Total** | **2847** | **~6小时** | **~1.5GB** |

*注: 实际时间取决于工具调用延迟、网络速度、GPU型号等因素*

## 后续分析

推理完成后，可以：

1. **计算总体准确率**:
```bash
python3 -c "
import json

trajectory_file = 'validation_trajectory/benchmark_inference_all_benchmarks_*/1.jsonl'
with open(trajectory_file) as f:
    scores = [json.loads(line)['score'] for line in f]
    accuracy = sum(scores) / len(scores) * 100
    print(f'Overall Accuracy: {accuracy:.2f}%')
    print(f'Correct: {sum(scores)}/{len(scores)}')
"
```

2. **按benchmark统计准确率**:
```bash
python3 -c "
import json
from collections import defaultdict

trajectory_file = 'validation_trajectory/benchmark_inference_all_benchmarks_*/1.jsonl'
stats = defaultdict(list)

with open(trajectory_file) as f:
    for line in f:
        data = json.loads(line)
        # Extract data_source from input (it's in the tools_kwargs)
        stats['all'].append(data['score'])

for bench, scores in stats.items():
    acc = sum(scores) / len(scores) * 100
    print(f'{bench}: {acc:.2f}% ({sum(scores)}/{len(scores)})')
"
```

3. **分析失败案例**: 使用visualization脚本查看score=0的样本

4. **统计工具使用情况**: 分析assistant_turns和user_turns分布

5. **对比不同benchmark**: 如果分别推理了单个benchmark，可以对比各benchmark的难度和模型表现

## 关于Reward计算

**重要**: Reward计算**不区分**data_source。

查看 `verl/utils/reward_score/search_r1_like_qa_em.py`，`compute_score()` 函数只使用：
- `solution_str`: 模型输出
- `ground_truth`: 标准答案

**不使用** `data_source` 字段。所有benchmark使用相同的评分标准：
- 使用exact match (EM)检查
- 归一化处理（去除冠词、标点、大小写）
- 答案必须完全匹配才得分

这意味着：
1. ✅ 可以安全地合并所有benchmarks一起推理
2. ✅ Reward计算对所有数据源一致
3. ✅ 可以直接比较不同benchmark的得分

## 相关文件

- **预处理脚本**: `examples/data_preprocess/preprocess_all_benchmarks.py`
- **合并脚本**: `examples/data_preprocess/merge_benchmarks.py`
- **推理脚本**: `examples/sglang_multiturn/search_browser/inference_benchmarks.sh`
- **测试脚本**: `examples/sglang_multiturn/search_browser/test_inference.sh`
- **可视化脚本**: `scripts/visualize_trajectory.py`
- **工具配置**: `examples/sglang_multiturn/config/tool_config/google_search_browse_tool_config.yaml`
- **Reward函数**: `verl/utils/reward_score/search_r1_like_qa_em.py`
- **数据目录**: `data/benchmarks_processed/`
- **结果目录**: `validation_trajectory/`

## 注意事项

1. **环境激活**: 所有命令都需要先激活conda环境
   ```bash
   conda activate /share/project/wanli/env/verl-v060
   ```

2. **工作目录**: 确保在verl项目根目录执行命令
   ```bash
   cd /share/project/wanli/Search_Agent/verl
   ```

3. **工具服务**: 推理前确保工具后端服务正在运行

4. **GPU资源**: 推理需要8个GPU，确保资源可用

5. **日志管理**: 推理会产生大量日志，定期清理 `logs/` 目录
