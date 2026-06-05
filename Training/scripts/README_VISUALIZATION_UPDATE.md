# 可视化脚本更新 - 动态字段显示

## ✨ 更新亮点

可视化脚本已升级为**完全动态显示模式**，可以自动显示 JSON 中的所有字段，无需硬编码！

---

## 🔧 主要改进

### 1. **动态字段显示**

现在脚本会自动检测并显示 JSON 中的所有字段，包括：
- ✅ 预定义的常见字段（优先显示）
- ✅ 任何自定义字段（自动检测并显示）
- ✅ 向后兼容（缺少字段不会报错）

### 2. **字段显示优先级**

**高优先级字段**（显示在顶部）：
- `step` - 训练步数
- `score` - 奖励分数
- `data_source` - 数据来源
- `assistant_turns` - 助手轮次
- `user_turns` - 用户轮次
- `pred_ans` - 预测答案
- `reward` - 奖励值

**额外字段**（显示在 "Additional Fields" 部分）：
- 任何其他自定义字段都会自动显示

### 3. **特殊处理字段**

以下字段有特殊显示逻辑，不在基本信息中显示：
- `input` - 在问题区域显示
- `output` - 解析为多轮对话显示
- `gts` - 作为 Ground Truth 单独显示

---

## 📊 显示示例

### 示例 1: 包含所有新字段

**输入 JSON:**
```json
{
  "step": 100,
  "score": 1.0,
  "data_source": "GAIA",
  "assistant_turns": 1,
  "user_turns": 0,
  "pred_ans": "Paris",
  "reward": 1.0,
  "input": "...",
  "output": "...",
  "gts": "Paris"
}
```

**文本输出:**
```
================================================================================
📊 Entry #1
================================================================================
Step: 100
Score: 1.0
Data Source: GAIA
Assistant Turns: 1
User Turns: 0
Predicted Answer: Paris
Reward: 1.0

Ground Truth: "Paris"
Question: ...
```

**HTML 输出:**
```
Step: 100 | Score: 1.0 | Data Source: GAIA
Assistant Turns: 1 | User Turns: 0 | Predicted Answer: Paris | Reward: 1.0
```

---

### 示例 2: 只有基础字段（旧格式）

**输入 JSON:**
```json
{
  "step": 100,
  "score": 1.0,
  "input": "...",
  "output": "...",
  "gts": "Paris"
}
```

**文本输出:**
```
================================================================================
📊 Entry #1
================================================================================
Step: 100
Score: 1.0

Ground Truth: "Paris"
Question: ...
```

✅ **向后兼容** - 没有新字段也能正常显示！

---

### 示例 3: 包含自定义字段

**输入 JSON:**
```json
{
  "step": 100,
  "score": 1.0,
  "custom_metric": 0.99,
  "experiment_id": "exp_123",
  "input": "...",
  "output": "...",
  "gts": "4"
}
```

**文本输出:**
```
================================================================================
📊 Entry #1
================================================================================
Step: 100
Score: 1.0

📋 Additional Fields:
  - custom_metric: 0.99
  - experiment_id: exp_123

Ground Truth: "4"
Question: ...
```

**HTML 输出:**
```
Step: 100 | Score: 1.0
Custom Metric: 0.99 | Experiment Id: exp_123
```

✅ **自动检测自定义字段** - 无需修改代码！

---

## 🚀 使用方法

### 基本用法（没有变化）

```bash
# 可视化验证轨迹
python scripts/visualize_trajectory.py \
    ./validation_trajectory/checkpoint_*/331.jsonl \
    -o ./visual_traj \
    -n 100
```

### 查看输出

```bash
# 打开 HTML 报告
xdg-open ./visual_traj/331_report_*.html

# 或查看 JSON 结构化数据
cat ./visual_traj/331_structured_*.json | python -m json.tool | less
```

---

## 🎯 关键优势

### 1. **完全动态**
不需要为新字段修改代码，JSON 中有什么就显示什么！

### 2. **向后兼容**
旧格式数据（没有新字段）仍然可以正常可视化。

### 3. **自动检测**
任何自定义字段都会自动被检测并显示在 "Additional Fields" 部分。

### 4. **优先级显示**
常见字段按优先级显示在顶部，保持输出整洁有序。

### 5. **三种输出格式**
- **文本**: 适合快速浏览和调试
- **JSON**: 适合程序化处理和分析
- **HTML**: 适合详细查看和分享

---

## 📝 技术细节

### 字段处理逻辑

```python
# 1. 定义特殊字段（不在基本信息中显示）
special_fields = {'input', 'output', 'gts'}

# 2. 定义优先显示的基本字段
basic_fields = ['step', 'score', 'data_source', 'assistant_turns', 
                'user_turns', 'pred_ans', 'reward']

# 3. 显示优先字段（按顺序）
for field in basic_fields:
    if field in entry:
        display(field, entry[field])

# 4. 显示其他额外字段（自动检测）
extra_fields = entry.keys() - basic_fields - special_fields
for field in sorted(extra_fields):
    display(field, entry[field])
```

### JSON 结构化输出

所有字段都会原样保存到 JSON 文件中：

```python
# 动态添加所有字段
structured_entry = {
    'entry_id': i,
    'turns': turns,
    'statistics': {...}
}

# 添加所有原始字段（除了 output，已解析为 turns）
for key, value in entry.items():
    if key not in ['output']:
        structured_entry[key] = value
```

---

## ✅ 测试验证

### 测试 1: 旧格式数据

```bash
# 创建测试数据
echo '{"step": 100, "score": 1.0, "input": "...", "output": "...", "gts": "test"}' > test.jsonl

# 可视化
python scripts/visualize_trajectory.py test.jsonl -o ./output
```

**结果**: ✅ 正常显示，无错误

### 测试 2: 新格式数据（包含所有字段）

```bash
# 创建测试数据
echo '{"step": 100, "score": 1.0, "data_source": "GAIA", "assistant_turns": 1, ...}' > test.jsonl

# 可视化
python scripts/visualize_trajectory.py test.jsonl -o ./output
```

**结果**: ✅ 所有新字段正确显示

### 测试 3: 自定义字段

```bash
# 创建测试数据
echo '{"step": 100, "score": 1.0, "custom_field": "value", "another_metric": 0.99, ...}' > test.jsonl

# 可视化
python scripts/visualize_trajectory.py test.jsonl -o ./output
```

**结果**: ✅ 自定义字段在 "Additional Fields" 中显示

---

## 🔄 迁移指南

如果你之前使用旧版本的可视化脚本：

### 不需要任何修改！

新版本完全向后兼容，直接使用即可。

### 可选：添加新字段

如果想让验证轨迹包含更多信息，只需在保存时添加到 `reward_extra_infos_dict` 即可：

```python
# 在 ray_trainer.py 中
reward_extra_infos_dict["your_custom_field"] = your_values

# 可视化脚本会自动显示这个字段
```

---

## 📚 相关文件

- **可视化脚本**: `scripts/visualize_trajectory.py`（已更新）
- **验证脚本**: `verify_data_source.py`
- **验证保存**: `verl/trainer/ppo/ray_trainer.py`（已更新）

---

## 🎉 总结

✅ **完全动态** - 自动显示所有 JSON 字段  
✅ **向后兼容** - 旧数据正常工作  
✅ **零配置** - 无需修改代码  
✅ **三种格式** - 文本/JSON/HTML 全支持  

现在可以放心添加任何自定义字段，可视化脚本会自动处理！🚀
