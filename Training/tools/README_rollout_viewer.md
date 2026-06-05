# Rollout Viewer 使用说明

## 简介

`rollout_viewer.py` 是一个用于可视化 verl 训练过程中生成的 rollout trajectory JSONL 文件的工具。

## 使用方法

```bash
python tools/rollout_viewer.py <jsonl_path> [--port PORT]

# 示例
python tools/rollout_viewer.py validation_trajectory/xxx/0.jsonl --port 7788
```

## 多轮对话解析逻辑

### 输入格式

JSONL 中每条记录包含 `input` 和 `output` 字段，合并后形成完整对话。

Qwen chat template 生成的对话格式如下：

```
system
You are a helpful assistant...
user
Question: xxx
assistant
<think>思考内容</think>
<tool_call>{"name": "search", "arguments": {...}}</tool_call>user
<tool_response>{"result": "..."}</tool_response>
assistant
<think>继续思考</think>
<answer>最终答案</answer>
```

### 解析步骤

1. **预处理**：在 `</tool_call>user` 和 `</tool_response>assistant` 之间插入换行符
2. **分割 Role**：按行首的 `system\n`、`user\n`、`assistant\n` 分割
3. **识别 Tool**：如果 `user` 消息以 `<tool_response>` 开头，标记为 `tool` 角色
4. **解析标签**：提取 `<think>`、`<tool_call>`、`<tool_response>`、`<answer>` 等标签内容

### 示例

**原始输入：**
```
system
You are a helpful assistant.
user
Question: What is 2+2?
assistant
<think>Let me calculate.</think>
<answer>4</answer>
```

**解析结果：**
```
Message 1: role=system, content="You are a helpful assistant."
Message 2: role=user, content="Question: What is 2+2?"
Message 3: role=assistant, content="<think>Let me calculate.</think>\n<answer>4</answer>"
    - Block 1: type=think, content="Let me calculate."
    - Block 2: type=answer, content="4"
```

### 轮数统计

- **Total Turns**：所有消息数量
- **Assistant Turns**：`role=assistant` 的消息数
- **User/Tool Turns**：`role=user` 或 `role=tool` 的消息数

## Tool Call 解析逻辑

### 支持的标签格式

解析器使用正则表达式匹配以下**标准标签格式**：

| 标签类型 | 正则模式 | 示例 |
|---------|---------|------|
| think | `<think>(.*?)</think>` | `<think>让我思考...</think>` |
| tool_call | `<tool_call>(.*?)</tool_call>` | `<tool_call>{"name": "search", ...}</tool_call>` |
| answer | `<answer>(.*?)</answer>` | `<answer>最终答案</answer>` |
| tool_response | `<tool_response>(.*?)</tool_response>` | `<tool_response>{"result": ...}</tool_response>` |

### 容错机制

1. **缺少 `<think>` 开始标签**：Qwen 模型生成时可能只有 `</think>` 结束标签，解析器会将开头到 `</think>` 的内容识别为思考内容
   ```
   # 输入（缺少开始标签）
   Let me search for this...
   </think>
   <tool_call>...</tool_call>
   
   # 仍能正确解析为 think 块
   ```

2. **未匹配内容**：标签之间或标签外的内容会被标记为 `other` 类型，不会丢失
   ```
   <think>思考</think>
   一些额外说明...        <-- 被标记为 'other'
   <answer>答案</answer>
   ```

3. **嵌套处理**：使用非贪婪匹配 `(.*?)`，确保嵌套标签不会错误匹配

### 不支持的格式

以下情况**不会**被正确解析为工具调用：

```python
# ❌ 标签名拼写错误
<toolcall>...</toolcall>      # 应为 <tool_call>

# ❌ 使用其他标签格式
<function_call>...</function_call>
[TOOL_CALL]...[/TOOL_CALL]

# ❌ JSON 不在标签内
search({"query": "..."})

# ❌ 缺少结束标签
<tool_call>{"name": "search"}   # 没有 </tool_call>
```

### Tool Call 可视化示例

**标准格式输入：**
```
assistant
<think>我需要搜索相关信息</think>
<tool_call>{"name": "search", "arguments": {"query": "python教程"}}</tool_call>
```

**解析后显示：**
- 🧠 **Think 块**：浅蓝色背景，显示思考内容
- 🔧 **Tool Call 块**：浅紫色背景，JSON 格式化显示参数

**工具响应格式：**
```
user
<tool_response>{"result": "A Google search for 'python教程' found 10 results:\n\n## Web Results\n1. [Python 官方教程](https://docs.python.org)..."}</tool_response>
```

解析后识别为 `role=tool`，以绿色背景显示工具返回结果。

## Token 统计

如果 JSONL 文件包含 token 统计字段（`input_tokens`, `output_tokens`, `total_tokens`），可视化界面会显示：

- 📊 主页统计卡片：平均 Total/Output Tokens、最大 Total Tokens
- 📋 表格 Tokens 列：`Output/Total` 格式
- 📝 详情页 Tokens 行：Input/Output/Total 分别显示

**添加 Token 统计到现有文件：**
```bash
python tools/add_token_counts.py <jsonl_path> --inplace
```

## 功能特性

- 📊 统计准确率、平均分数、轮数分布、Token 统计
- 🔍 支持按正确/错误、Method、状态筛选
- 💬 可视化多轮对话，高亮显示 think/tool_call/answer 等块
- 🔗 自动将 URL 转为可点击链接
- 🎨 JSON 格式化显示工具调用参数

