# 数据处理指南

## 快速开始

### 1. 合并多个 parquet 文件

```python
from merge_rag_with_mask_url import merge_parquet_files

# 合并多个文件
merge_parquet_files(
    input_files=["path1.parquet", "path2.parquet", "path3.parquet"],
    output_file="output.parquet"
)
```

### 2. 随机采样并保存为 JSON

```python
from sample_parquet import sample_to_json

# 从 parquet 中随机采样 5 条保存为 JSON
sample_to_json(
    input_file="data.parquet",
    output_file="samples.json",
    sample_count=5
)
```

## 功能

1. **自动提取 data_source**：从文件名自动设置 `data_source`（如 `stage2_wiki.parquet` → `"stage2_wiki"`）
2. **正确处理 mask_url**：自动从 `extra_info['mask_url']` 构建完整的 `tools_kwargs` 结构
3. **支持多文件合并**：一次处理多个 parquet 文件

## 脚本说明

### merge_rag_with_mask_url.py

合并多个 parquet 文件并正确处理 mask_url。

**参数**：
- `input_files`: 输入文件路径列表，如 `["file1.parquet", "file2.parquet"]`
- `output_file`: 输出文件路径，如 `"merged.parquet"`
- `save_sample`: 是否保存 JSON 样本（默认: `True`）
- `sample_count`: 样本数量（默认: `10`）

### sample_parquet.py

从 parquet 文件中随机采样并保存为 JSON。

**参数**：
- `input_file`: 输入 parquet 文件路径
- `output_file`: 输出 JSON 文件路径
- `sample_count`: 采样数量（默认: `5`）
- `random_seed`: 随机种子（默认: `42`）

## 输入数据格式

最小格式（只有 `mask_url`）：

```json
{
  "extra_info": {
    "mask_url": "https://en.wikipedia.org/wiki/..."
  }
}
```

## 输出数据格式

自动构建 `tools_kwargs`：

```json
{
  "data_source": "stage2_wiki",  // 从文件名自动提取
  "extra_info": {
    "mask_url": "https://en.wikipedia.org/wiki/...",
    "need_tools_kwargs": true,
    "tools_kwargs": {
      "search": {
        "create_kwargs": {
          "url": "https://en.wikipedia.org/wiki/..."  // 从 mask_url 自动填充
        }
      },
      "browse": {
        "create_kwargs": {
          "url": "https://en.wikipedia.org/wiki/..."
        }
      }
    }
  }
}
```

## 训练时的 data_source

`ray_trainer.py` 会自动保存 `data_source` 到 rollout trajectory：
- 位置：`rollout_data_dir/{global_steps}.jsonl`
- 字段：每行 JSON 包含 `data_source` 字段

## 完整示例

```python
from merge_rag_with_mask_url import merge_parquet_files
from sample_parquet import sample_to_json

# 1. 合并多个数据文件
merge_parquet_files(
    input_files=[
        "stage2_wiki.parquet",
        "stage2_science.parquet"
    ],
    output_file="stage2_all.parquet"
)

# 2. 随机采样查看数据
sample_to_json(
    input_file="stage2_all.parquet",
    output_file="samples.json",
    sample_count=5
)
```
