# 验证轨迹可视化更新说明

## 📋 更新概览

本次更新为验证轨迹系统添加了 `data_source` 字段支持，并确保所有 key-value 都能在可视化中正常展示。

---

## 🔧 修改内容

### 1. **验证保存逻辑** (`verl/trainer/ppo/ray_trainer.py`)

**修改位置**: `_validate()` 方法（第 624-640 行）

**修改内容**:
```python
# 在保存验证轨迹前，将 data_source 添加到 reward_extra_infos_dict
data_sources = np.concatenate(data_source_lst, axis=0)
reward_extra_infos_dict["data_source"] = data_sources.tolist()

# 然后调用 _dump_generations 保存
self._dump_generations(
    inputs=sample_inputs,
    outputs=sample_outputs,
    gts=sample_gts,
    scores=sample_scores,
    reward_extra_infos_dict=reward_extra_infos_dict,  # 包含 data_source
    dump_path=val_data_dir,
)
```

**效果**: 现在生成的 jsonl 文件会包含 `data_source` 字段。

---

### 2. **可视化脚本** (`scripts/visualize_trajectory.py`)

所有修改已完成，支持显示所有字段！

---

## 📊 支持的所有字段

可视化脚本现在支持显示以下所有字段：

### 基础字段
- ✅ `step` - 训练步数
- ✅ `score` - 奖励分数
- ✅ `input` - 输入问题
- ✅ `output` - 模型输出（解析为多轮对话）
- ✅ `gts` - Ground Truth（标准答案）

### 新增字段
- ✅ `data_source` - **数据来源**（如 GAIA, GPQA, WebWalkerQA 等）
- ✅ `assistant_turns` - **助手轮次数**
- ✅ `user_turns` - **用户轮次数**
- ✅ `pred_ans` - **预测答案**
- ✅ `reward` - 奖励值
- ✅ 其他 `reward_extra_info` 中的字段

---

## 🚀 使用方法

### 1. 运行推理并保存验证轨迹

```bash
# 运行推理脚本（会自动保存验证轨迹）
bash examples/sglang_multiturn/search_browser/inference_from_checkpoint_real.sh
```

### 2. 验证 data_source 字段是否存在

```bash
python verify_data_source.py ./validation_trajectory/checkpoint_inference_*/331.jsonl
```

### 3. 可视化验证轨迹

```bash
python scripts/visualize_trajectory.py \
    ./validation_trajectory/checkpoint_inference_*/331.jsonl \
    -o ./visual_traj \
    -n 100
```

### 4. 查看 HTML 报告

```bash
xdg-open ./visual_traj/331_report_*.html  # Linux
open ./visual_traj/331_report_*.html      # macOS
```

---

## ✅ 验证通过

所有测试已通过！可视化脚本能正确显示所有 key-value 字段。
