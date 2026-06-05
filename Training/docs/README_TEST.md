# Dual Tools Rollout Test Script

## 概述

`test_dual_tools_rollout.sh` 是一个专门用于测试双工具（Search + Browse）Agent Loop rollout的脚本。

与完整训练脚本不同，这个脚本：
- ✅ **只进行一轮 rollout**，不执行训练
- ✅ **使用少量数据**（默认10条）进行快速测试
- ✅ **自动分析结果**，显示工具调用成功率
- ✅ **详细的错误检查**和日志分析
- ✅ **轻量配置**，占用更少GPU资源

## 使用方法

### 基本用法（使用默认参数）

```bash
cd /share/project/wanli/Search_Agent/verl
./examples/sglang_multiturn/search_browser/test_dual_tools_rollout.sh
```

### 指定参数

```bash
./examples/sglang_multiturn/search_browser/test_dual_tools_rollout.sh \
    <测试数据路径> \
    <模型路径> \
    <工具配置路径>
```

### 示例

```bash
# 示例1: 使用默认配置（推荐用于首次测试）
./examples/sglang_multiturn/search_browser/test_dual_tools_rollout.sh

# 示例2: 使用自定义测试数据
./examples/sglang_multiturn/search_browser/test_dual_tools_rollout.sh \
    ./data/my_test_data.json

# 示例3: 完全自定义
./examples/sglang_multiturn/search_browser/test_dual_tools_rollout.sh \
    ./data/searchR1_first_10_samples.json \
    /path/to/model \
    ./examples/sglang_multiturn/config/tool_config/google_search_browse_tool_config.yaml
```

## 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 测试数据 | `data/searchR1_first_10_samples.json` | 训练数据的前10条样本 |
| 模型路径 | `/share/project/wanli/model/Qwen2.5-3B-Instruct` | Qwen 3B模型 |
| 工具配置 | `config/tool_config/google_search_browse_tool_config.yaml` | 双工具配置 |
| Batch Size | 10 | 小批量用于快速测试 |
| GPU数量 | 1 | 单GPU测试 |
| Rollout次数 | 2 | 每个问题生成2条轨迹 |
| 最大turns | 5 | 最多5轮对话 |

## 输出说明

### 目录结构

```
./test_rollout_output/
└── test_dual_tools_rollout_YYYYMMDD_HHMMSS/
    ├── rollout/           # Rollout轨迹文件（.jsonl）
    └── validation/        # 验证数据（如果有）

./test_logs/
└── test_dual_tools_rollout_YYYYMMDD_HHMMSS.log  # 完整日志
```

### 测试报告

脚本会自动生成测试摘要：

```
============================================================================
Rollout Analysis
============================================================================
Total rollout files: 1

File: 1.jsonl
------------------------------------------------------------
Total entries:        20
Tool calls found:     15
  - search calls:     10
  - browse calls:     5
Successful executions: 12
Failed executions:     3

✅ Tool success rate: 80.0%
```

## 验证检查点

脚本会自动检查以下内容：

1. ✅ **文件存在性**
   - 测试数据文件
   - 模型目录
   - 工具配置文件

2. ✅ **工具配置**
   - 显示注册的工具名称
   - 验证工具类路径
   - 显示工具描述

3. ✅ **Rollout结果**
   - 工具调用次数
   - Search vs Browse 调用分布
   - 成功/失败统计
   - 成功率计算

4. ✅ **错误分析**
   - 总错误数
   - 工具执行错误
   - 常见错误类型

## 常见问题排查

### 1. 工具无法找到 (`Error when executing tool: 'xxx'`)

**检查**：
```bash
# 查看工具配置
cat examples/sglang_multiturn/config/tool_config/google_search_browse_tool_config.yaml

# 确认工具名称是否匹配训练数据
python3 << EOF
import yaml, json
with open('examples/sglang_multiturn/config/tool_config/google_search_browse_tool_config.yaml') as f:
    config = yaml.safe_load(f)
    print("Configured tools:", [t['tool_schema']['function']['name'] for t in config['tools']])

with open('data/searchR1_first_10_samples.json') as f:
    data = json.load(f)
    print("Expected tools in prompt:", ["search", "browse"])  # 从system prompt中提取
EOF
```

**解决方案**：
- 确保配置文件中的工具名称为 `search` 和 `browse`
- 不要使用 `google_search` 或其他名称

### 2. 工具服务连接失败

**检查服务是否运行**：
```bash
# 检查search服务
curl -X POST http://47.111.147.142:8864/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "search_type": "hybrid", "limit": 3}'

# 检查browse服务
curl -X POST http://47.111.147.142:8084/query \
  -H "Content-Type: application/json" \
  -d '{"link": "https://example.com", "what_to_find": "test"}'
```

### 3. 模型生成格式错误

**检查轨迹输出**：
```bash
# 查看第一个rollout文件的前几条
head -100 test_rollout_output/test_dual_tools_rollout_*/rollout/1.jsonl | python3 -m json.tool
```

**期望格式**：
```
<think>
推理过程...
</think>
<tool_call>
{"name": "search", "arguments": {"query": "搜索内容"}}
</tool_call>
```

**错误格式**：
```
<answer>
{"name": "search", ...}  # ❌ 错在使用了 <answer> 而不是 <tool_call>
</answer>
```

### 4. GPU内存不足

**调整参数**：
编辑脚本中的这些行：
```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.6  # 降低到 0.4 或 0.3
data.train_batch_size=10                               # 降低到 5 或 2
actor_rollout_ref.rollout.n=2                          # 降低到 1
```

## 与完整训练的区别

| 特性 | 测试脚本 | 训练脚本 |
|------|----------|----------|
| 目的 | 验证rollout功能 | 完整PPO训练 |
| 数据量 | 10条（可调） | 169,615条 |
| Epoch数 | 1 | 10-15 |
| Batch Size | 10 | 512 |
| GPU数量 | 1 | 8 |
| 运行时间 | 2-5分钟 | 数小时 |
| 保存模型 | 否 | 是 |
| 日志记录 | Console only | Console + WandB |

## 调试技巧

### 1. 查看详细日志
```bash
# 实时查看日志
tail -f test_logs/test_dual_tools_rollout_*.log

# 搜索特定错误
grep -i "error" test_logs/test_dual_tools_rollout_*.log | less
```

### 2. 分析单条轨迹
```bash
# 格式化查看第一条轨迹
head -1 test_rollout_output/test_dual_tools_rollout_*/rollout/1.jsonl | python3 -m json.tool

# 提取所有工具调用
grep -o '"name":[^,}]*' test_rollout_output/test_dual_tools_rollout_*/rollout/1.jsonl | sort | uniq -c
```

### 3. 验证工具执行
```bash
# 查看工具响应
python3 << 'EOF'
import json
with open('test_rollout_output/test_dual_tools_rollout_*/rollout/1.jsonl') as f:
    for line in f:
        data = json.loads(line)
        output = data['output']
        if '<tool_response>' in output:
            start = output.find('<tool_response>')
            end = output.find('</tool_response>') + len('</tool_response>')
            print(output[start:end])
            print('-' * 60)
EOF
```

## 快速验证清单

运行测试前，确认以下几点：

- [ ] 工具后端服务正在运行（search + browse）
- [ ] 配置文件中工具名称为 `search` 和 `browse`
- [ ] 测试数据文件存在且格式正确
- [ ] 有足够的GPU内存（至少12GB）
- [ ] 模型路径正确

运行测试后，检查：

- [ ] 是否有rollout文件生成
- [ ] 工具调用成功率 > 50%
- [ ] 没有大量 "Error when executing tool" 错误
- [ ] 生成的轨迹包含正确的 `<tool_call>` 格式

## 进一步调试

如果测试失败，按以下顺序排查：

1. **工具服务** → 测试后端API
2. **工具配置** → 检查名称和参数
3. **模型输出** → 检查格式是否正确
4. **代码逻辑** → 检查工具解析器

## 相关文件

- 训练脚本: `agentloop_search_browse.sh`
- 工具配置: `config/tool_config/google_search_browse_tool_config.yaml`
- 工具实现: `verl/tools/google_search_tool.py`, `verl/tools/browse_tool.py`
- 测试数据: `data/searchR1_first_10_samples.json`

## 联系与支持

如有问题，请检查：
1. 日志文件 (`test_logs/`)
2. Rollout输出 (`test_rollout_output/`)
3. 相关GitHub Issues
