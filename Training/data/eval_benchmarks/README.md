# 评估基准数据集转换说明

本文件夹包含将 GAIA 和 Xbench 评估数据集转换为与训练数据 `stage2_rag_only.parquet` 相同格式的 parquet 文件。

## 文件列表

### 转换后的 Parquet 文件
- **GAIA_test.parquet**: GAIA 评估数据集 (103 条数据)
- **Xbench_test.parquet**: Xbench 评估数据集 (100 条数据)

### 可视化样本 JSON 文件
- **GAIA_test_samples.json**: GAIA 前 3 个样本的可视化
- **Xbench_test_samples.json**: Xbench 前 3 个样本的可视化

### 转换脚本
- **convert_benchmarks_to_parquet.py**: 数据转换脚本

## 数据格式说明

所有 parquet 文件包含以下字段（与 `stage2_rag_only.parquet` 完全一致）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | string | 问题文本 |
| `data_source` | string | 数据源标识 (GAIA / Xbench / rag_direct) |
| `prompt` | list | 包含 system 和 user 消息的对话列表 |
| `ability` | string | 能力类型，统一为 "search" |
| `reward_model` | dict | 奖励模型配置，包含 ground_truth 和 style |
| `extra_info` | dict | 额外信息，包含 index、split、original_id 等 |
| `metadata` | None | 元数据字段 |

## System Prompt

所有数据使用相同的 system prompt，提供：
- Deep AI Research Assistant 角色定义
- 搜索和浏览工具说明
- 输出格式要求（think/answer 或 think/tool_call）
- 工具调用规范

## 主要差异

与原始训练数据相比，评估数据的主要差异：

1. **data_source**: 
   - 训练数据: `rag_direct`, `rag_mqa` 等
   - 评估数据: `GAIA`, `Xbench`

2. **question**:
   - GAIA: 英文问题，需要深度研究
   - Xbench: 中文问题，涵盖多领域知识

3. **extra_info.split**:
   - 训练数据: `train`
   - 评估数据: `test`

## 使用方法

```python
import pandas as pd

# 读取评估数据
gaia_df = pd.read_parquet('GAIA_test.parquet')
xbench_df = pd.read_parquet('Xbench_test.parquet')

# 查看数据结构
print(gaia_df.columns)
print(gaia_df.head())
```

## 原始数据来源

- GAIA: `/share/project/wanli/Search_Agent/DeepResearch/inference/eval_data/benchmarks/GAIA.jsonl`
- Xbench: `/share/project/wanli/Search_Agent/DeepResearch/inference/eval_data/benchmarks/Xbench.jsonl`

## 训练数据可视化

训练数据的可视化样本已保存至：
- `/share/project/wanli/Search_Agent/verl/data/deepresearch_rl/stage2_final/stage2_rag_only_samples.json`

## 生成时间

2025-01-16
