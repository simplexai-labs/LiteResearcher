# tools_kwargs 完整调用链代码分析

## 数据流概览（AgentLoop 路径）

**注意**：当 `rollout.mode == "async"` 时，使用的是 AgentLoop 路径，而不是 SGLang Rollout 路径。

```
数据集 (parquet)
  ↓ extra_info['tools_kwargs']
RLDataset
  ↓ row_dict['tools_kwargs'] → DataProto.non_tensor_batch['tools_kwargs']
AgentLoopManager.generate_sequences()
  ↓ DataProto (包含 tools_kwargs)
AgentLoopWorker.generate_sequences()
  ↓ batch.non_tensor_batch['tools_kwargs'] → kwargs
_run_agent_loop()
  ↓ **kwargs (包含 tools_kwargs)
ToolAgentLoop.run()
  ↓ kwargs.get("tools_kwargs", {})
_call_tool()
  ↓ tools_kwargs['search']['create_kwargs']
GoogleSearchTool.create()
  ↓ create_kwargs['url']
self._instance_dict[instance_id]['masked_url']
  ↓
GoogleSearchTool.execute()
  ↓ self._instance_dict[instance_id]['masked_url']
_mask_url_in_search_result()
```

---

## 1. RLDataset 读取 tools_kwargs

**文件**: `verl/utils/dataset/rl_dataset.py`

```python
# 第 365 行：从 extra_info 中提取 tools_kwargs
tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})

# 第 367 行：检查是否需要 tools_kwargs
need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)

# 第 368-369 行：如果需要但没有，发出警告
if need_tools_kwargs and not tools_kwargs:
    logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])

# 第 371 行：将 tools_kwargs 添加到 row_dict
row_dict["tools_kwargs"] = tools_kwargs
```

**关键点**：
- 从 `extra_info['tools_kwargs']` 读取
- 如果 `need_tools_kwargs=True` 但 `tools_kwargs` 为空，会发出警告
- 最终 `row_dict['tools_kwargs']` 包含完整的嵌套结构

---

## 2. AgentLoopManager 和 AgentLoopWorker 传递 tools_kwargs

**文件**: `verl/experimental/agent_loop/agent_loop.py`

### 2.1 AgentLoopManager.generate_sequences()

```python
# 第 903 行：AgentLoopManager.generate_sequences() 接收 DataProto
def generate_sequences(self, prompts: DataProto) -> DataProto:
    # ...
    # 第 919 行：将 prompts 分块
    chunkes = prompts.chunk(len(self.agent_loop_workers))
    
    # 第 920-924 行：分发给各个 AgentLoopWorker
    outputs = ray.get(
        [
            worker.generate_sequences.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
        ]
    )
```

**关键点**：
- `prompts` 是 `DataProto`，包含 `non_tensor_batch['tools_kwargs']`
- 每个 chunk 都包含完整的 `tools_kwargs` 信息

### 2.2 AgentLoopWorker.generate_sequences()

```python
# 第 423 行：AgentLoopWorker.generate_sequences() 接收 batch
async def generate_sequences(self, batch: DataProto) -> DataProto:
    # ...
    # 第 515 行（或 528 行）：从 batch.non_tensor_batch 中提取所有字段作为 kwargs
    kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
    
    # 第 517 行（或 529 行）：传递给 _run_agent_loop()
    coro = self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)
```

**关键点**：
- `batch.non_tensor_batch['tools_kwargs']` 被提取到 `kwargs` 中
- `kwargs` 包含所有 non_tensor_batch 字段，包括 `tools_kwargs`

### 2.3 _run_agent_loop() 传递 kwargs

```python
# 第 542 行：_run_agent_loop() 接收 **kwargs
async def _run_agent_loop(
    self,
    sampling_params: dict[str, Any],
    trajectory: dict[str, Any],
    *,
    agent_name: str,
    **kwargs,  # 包含 tools_kwargs
) -> _InternalAgentLoopOutput:
    # ...
    # 第 569 行：将 kwargs 传递给 agent_loop.run()
    output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
```

**关键点**：
- `**kwargs` 包含从 `batch.non_tensor_batch` 提取的所有字段
- 包括 `tools_kwargs`、`raw_prompt`、`extra_info` 等

---

## 3. ToolAgentLoop 接收 tools_kwargs

**文件**: `verl/experimental/agent_loop/tool_agent_loop.py`

### 3.1 run() 方法接收 tools_kwargs

```python
# 第 118 行：run() 方法定义
async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
    # 第 119 行：从 kwargs 中提取 raw_prompt
    messages = list(kwargs["raw_prompt"])
    
    # 第 123 行：从 kwargs 中提取 tools_kwargs
    tools_kwargs = kwargs.get("tools_kwargs", {})
    
    # 第 141-149 行：创建 AgentData，包含 tools_kwargs
    agent_data = AgentData(
        messages=messages,
        image_data=image_data,
        metrics=metrics,
        request_id=request_id,
        tools_kwargs=tools_kwargs,  # 传递给 AgentData
        interaction=interaction,
        interaction_kwargs=interaction_kwargs,
    )
```

**关键点**：
- `kwargs` 来自 `_run_agent_loop()`，包含从 `batch.non_tensor_batch` 提取的所有字段
- `tools_kwargs` 直接从 `kwargs` 中获取

### 3.2 _call_tool() 方法使用 tools_kwargs

```python
# 第 520-522 行：_call_tool 方法定义
async def _call_tool(
    self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]
) -> tuple[ToolResponse, float, dict]:
    """Call tool and return tool response."""
    tool, instance_id = None, None
    tool_name = None
    
    try:
        # 第 528 行：获取工具名称（如 "search"）
        tool_name = tool_call.name
        
        # 第 529 行：解析工具参数
        tool_args = json.loads(tool_call.arguments)
        
        # 第 530 行：获取工具实例
        tool = self.tools[tool_name]
        
        # 第 531 行：从 tools_kwargs 中获取该工具的配置
        # 例如：tools_kwargs['search'] = {'create_kwargs': {'url': '...'}}
        kwargs = tools_kwargs.get(tool_name, {})
        
        # 第 533 行：调用 tool.create()，传入 create_kwargs
        # kwargs.get("create_kwargs", {}) 获取 {'url': '...'}
        create_result = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
        
        instance_id = create_result[0] if isinstance(create_result, tuple) else create_result
        
        # 第 536 行：执行工具
        tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
```

**关键点**：
- `tools_kwargs.get(tool_name, {})` 获取特定工具的配置（如 `tools_kwargs['search']`）
- `kwargs.get("create_kwargs", {})` 获取 `create_kwargs`（如 `{'url': '...'}`）
- 传递给 `tool.create(create_kwargs=...)`

---

## 4. Google Search Tool 读取 url

**文件**: `verl/tools/google_search_tool.py`

### 4.1 create() 方法读取 url

```python
# 第 324-336 行：create() 方法
async def create(self, instance_id: Optional[str] = None, create_kwargs: Optional[dict] = None, **kwargs) -> tuple[str, ToolResponse]:
    if instance_id is None:
        instance_id = str(uuid4())
    
    # 第 328 行：确保 create_kwargs 不为 None
    create_kwargs = create_kwargs or {}
    
    # 第 329 行：从 create_kwargs 中读取 'url' 键
    # 这就是 tools_kwargs['search']['create_kwargs']['url']
    masked_url = create_kwargs.get("url", "")
    
    # 第 331-335 行：将 masked_url 存储到实例字典中
    self._instance_dict[instance_id] = {
        "response": "",
        "reward": [],
        "masked_url": masked_url,  # 存储在这里，供后续使用
    }
    return instance_id, ToolResponse()
```

**关键点**：
- `create_kwargs.get("url", "")` 读取 `tools_kwargs['search']['create_kwargs']['url']`
- 将 `masked_url` 存储到 `self._instance_dict[instance_id]['masked_url']` 中

### 4.2 execute() 方法使用 masked_url

```python
# 第 355 行：execute() 方法定义
async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
    """Execute search tool."""
    # ...
    
    # 第 388 行：从实例字典中获取之前存储的 masked_url
    masked_url = self._instance_dict[instance_id].get("masked_url", "")
    
    # 第 409 行：调用去重方法，传入 masked_url
    results_by_query, total_results, unique_results = self._deduplicate_per_query(
        results_by_query, masked_url
    )
    
    # 第 433-434 行：如果 masked_url 存在，对结果进行 mask
    if masked_url:
        result_text = self._mask_url_in_search_result(result_text, masked_url)
```

### 4.3 _mask_url_in_search_result() 方法

```python
# 第 176 行：_mask_url_in_search_result() 方法定义
def _mask_url_in_search_result(self, search_result: str, masked_url: str) -> str:
    """从搜索结果中移除指定的 URL"""
    if not masked_url or not isinstance(search_result, str) or masked_url not in search_result:
        return search_result
    
    result = search_result
    # 第 186-189 行：循环移除所有出现的 masked_url
    while masked_url in result:
        # 找到 URL 的位置并移除
        url_pos = result.find(masked_url)
        # ... 移除逻辑 ...
    
    return result
```

**关键点**：
- `masked_url` 用于过滤搜索结果，移除包含该 URL 的条目
- 防止模型直接访问答案来源页面（避免作弊）

---

## 5. 完整调用链示例（AgentLoop 路径）

假设数据集中有：

```python
extra_info = {
    "tools_kwargs": {
        "search": {
            "create_kwargs": {
                "url": "https://en.wikipedia.org/wiki/Stanford%20on%20Teme"
            }
        },
        "browse": {
            "create_kwargs": {
                "url": "https://en.wikipedia.org/wiki/Stanford%20on%20Teme"
            }
        }
    }
}
```

### 调用流程（AgentLoop 路径）：

1. **RLDataset** (`verl/utils/dataset/rl_dataset.py:365`):
   ```python
   tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
   row_dict["tools_kwargs"] = tools_kwargs
   # tools_kwargs = {'search': {'create_kwargs': {'url': '...'}}, 'browse': {...}}
   # 最终存储在 DataProto.non_tensor_batch['tools_kwargs'] 中
   ```

2. **AgentLoopManager.generate_sequences()** (`verl/experimental/agent_loop/agent_loop.py:903`):
   ```python
   # 接收 DataProto，包含 non_tensor_batch['tools_kwargs']
   chunkes = prompts.chunk(len(self.agent_loop_workers))
   # 分发给各个 AgentLoopWorker
   ```

3. **AgentLoopWorker.generate_sequences()** (`verl/experimental/agent_loop/agent_loop.py:423`):
   ```python
   # 第 515 行：从 batch.non_tensor_batch 提取所有字段
   kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
   # kwargs 包含: {'tools_kwargs': {...}, 'raw_prompt': [...], 'extra_info': {...}, ...}
   ```

4. **AgentLoopWorker._run_agent_loop()** (`verl/experimental/agent_loop/agent_loop.py:542`):
   ```python
   # 第 569 行：将 kwargs 传递给 agent_loop.run()
   output = await agent_loop.run(sampling_params, **kwargs)
   ```

5. **ToolAgentLoop.run()** (`verl/experimental/agent_loop/tool_agent_loop.py:118`):
   ```python
   # 第 123 行：从 kwargs 中提取 tools_kwargs
   tools_kwargs = kwargs.get("tools_kwargs", {})
   # tools_kwargs = {'search': {'create_kwargs': {'url': '...'}}, 'browse': {...}}
   ```

6. **ToolAgentLoop._call_tool()** (`verl/experimental/agent_loop/tool_agent_loop.py:520`):
   ```python
   # 第 531 行：从 tools_kwargs 获取特定工具的配置
   kwargs = tools_kwargs.get("search", {})
   # kwargs = {'create_kwargs': {'url': '...'}}
   
   # 第 533 行：提取 create_kwargs 并传递给 tool.create()
   create_result = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
   # create_kwargs = {'url': 'https://en.wikipedia.org/wiki/Stanford%20on%20Teme'}
   ```

7. **GoogleSearchTool.create()** (`verl/tools/google_search_tool.py:324`):
   ```python
   # 第 329 行：从 create_kwargs 中读取 url
   masked_url = create_kwargs.get("url", "")
   # masked_url = "https://en.wikipedia.org/wiki/Stanford%20on%20Teme"
   
   # 第 334 行：存储到实例字典
   self._instance_dict[instance_id]["masked_url"] = masked_url
   ```

8. **GoogleSearchTool.execute()** (`verl/tools/google_search_tool.py:355`):
   ```python
   # 第 388 行：从实例字典获取 masked_url
   masked_url = self._instance_dict[instance_id].get("masked_url", "")
   # masked_url = "https://en.wikipedia.org/wiki/Stanford%20on%20Teme"
   
   # 第 433-434 行：使用 masked_url 过滤搜索结果
   if masked_url:
       result_text = self._mask_url_in_search_result(result_text, masked_url)
   ```

---

## 总结

### AgentLoop 路径（当 `rollout.mode == "async"` 时）

**关键路径**：
1. `extra_info['tools_kwargs']` → **RLDataset** 读取 (`rl_dataset.py:365`)
2. `DataProto.non_tensor_batch['tools_kwargs']` → **AgentLoopManager** 分发 (`agent_loop.py:903`)
3. `batch.non_tensor_batch['tools_kwargs']` → **AgentLoopWorker** 提取到 kwargs (`agent_loop.py:515`)
4. `kwargs['tools_kwargs']` → **ToolAgentLoop.run()** 接收 (`tool_agent_loop.py:123`)
5. `tools_kwargs['search']` → **ToolAgentLoop._call_tool()** 获取 (`tool_agent_loop.py:531`)
6. `tools_kwargs['search']['create_kwargs']` → 传递给 **tool.create()** (`tool_agent_loop.py:533`)
7. `create_kwargs['url']` → **GoogleSearchTool.create()** 读取并存储 (`google_search_tool.py:329`)
8. `self._instance_dict[instance_id]['masked_url']` → **GoogleSearchTool.execute()** 使用 (`google_search_tool.py:388, 433`)

### 入口点

**文件**: `verl/trainer/ppo/ray_trainer.py`

```python
# 第 969-975 行：当 rollout.mode == "async" 时，使用 AgentLoopManager
if self.config.actor_rollout_ref.rollout.mode == "async":
    from verl.experimental.agent_loop import AgentLoopManager
    
    self.async_rollout_mode = True
    self.async_rollout_manager = AgentLoopManager(
        config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
    )
```

**关键点**：
- 当 `rollout.mode == "async"` 时，使用 `AgentLoopManager` 而不是 `SGLangRollout`
- `AgentLoopManager` 通过 `AgentLoopWorker` 调用 `ToolAgentLoop`
- `tools_kwargs` 通过 `DataProto.non_tensor_batch` 传递

**必须包含的完整结构**：
```python
{
    "extra_info": {
        "tools_kwargs": {
            "search": {
                "create_kwargs": {
                    "url": "https://..."  # 必须
                }
            },
            "browse": {
                "create_kwargs": {
                    "url": "https://..."  # 必须
                }
            }
        }
    }
}
```
