# Google搜索工具统计日志功能

## 功能概述

为Google搜索工具添加了类似浏览器工具的实时统计日志功能，用于监控搜索请求的状态。

## 主要特性

1. **实时统计**：实时跟踪搜索请求的状态（排队、运行中、已完成）
2. **分层统计**：
   - **Total**: 总请求数统计（外层请求）
   - **API**: API调用统计（实际的搜索API调用）
3. **自动日志记录**：后台任务定期输出统计信息
4. **线程安全**：使用锁机制确保多线程环境下的统计准确性

## 统计指标

### Total（总请求）
- `running`: 正在处理的请求数
- `queued`: 排队等待处理的请求数
- `done`: 已完成的请求数

### API（API调用）
- `running`: 正在调用API的请求数
- `queued`: 等待调用API的请求数（已进入执行函数，但还未调用后端）
- `done`: API调用完成数

## 日志格式

```
2026-01-06 19:31:03 | INFO | REALTIME | Total[running=1, queued=0, done=0] | API[running=0, queued=0, done=1]
2026-01-06 19:31:06 | INFO | REALTIME | Total[running=2, queued=0, done=0] | API[running=0, queued=0, done=2]
```

## 配置说明

### 1. 工具配置文件

在 `tool_config.yaml` 中添加：

```yaml
tools:
  - class_name: verl.tools.google_search_tool.GoogleSearchTool
    config:
      search_service_url: http://47.111.147.142:8010/search
      num_workers: 300
      rate_limit: 300
      timeout: 120
      stats_log_dir: ./log_search  # 统计日志目录（可选，默认为 ./log_search）
```

### 2. 环境变量

可以通过环境变量设置日志输出间隔：

```bash
export SEARCH_STATS_LOG_INTERVAL=3  # 统计日志输出间隔（秒），默认为3秒
```

## 日志文件

统计日志会自动保存到指定目录，文件名格式：

```
<stats_log_dir>/<timestamp>_stats.log
```

例如：`./log_search/20260106_192105_stats.log`

## 工作流程

```
用户请求 → request_queue() → request_start()
         ↓
    处理请求参数
         ↓
    api_queue() → api_start() → 调用后端API → api_end()
         ↓
    request_end()
```

## 统计时机

1. **request_queue()**: 请求进入execute方法时
2. **request_start()**: 开始处理请求参数后
3. **api_queue()**: 准备调用后端API时
4. **api_start()**: 开始调用后端API时
5. **api_end()**: 后端API调用完成时
6. **request_end()**: 整个请求处理完成时（在finally块中确保执行）

## 对比浏览器工具

| 维度 | Google搜索工具 | 浏览器工具 |
|------|---------------|-----------|
| Total | 总请求数 | 总请求数 |
| 第二层 | API调用 | SQL查询 |
| 第三层 | - | 爬虫任务 |
| 第四层 | - | Summary生成 |

Google搜索工具相对简单，只有两层统计：
- 外层：整个搜索请求
- 内层：实际的API调用

## 测试

运行测试脚本验证功能：

```bash
cd /share/project/wanli/Search_Agent/verl
python test_search_stats.py
```

## 注意事项

1. **后台任务启动**：后台统计日志任务在工具初始化时启动，如果初始化时没有事件循环，会在首次调用时启动
2. **日志文件路径**：确保指定的日志目录有写入权限
3. **统计准确性**：使用了线程锁确保多线程环境下的统计准确性
4. **资源清理**：后台任务会在工具实例销毁时自动停止

## 实现文件

- `verl/tools/google_search_tool.py`: 主要实现文件
- `verl/tools/utils/google_search_utils.py`: 搜索工具函数
- `examples/sglang_multiturn/config/tool_config/google_search_browse_tool_config.yaml`: 配置文件

## 日志示例

```
2026-01-06 19:31:03,511 | INFO | 统计日志后台任务启动 | 间隔: 3s
2026-01-06 19:31:03,511 | INFO | REALTIME | Total[running=0, queued=0, done=0] | API[running=0, queued=0, done=0]
2026-01-06 19:31:06,511 | INFO | REALTIME | Total[running=2, queued=0, done=0] | API[running=0, queued=0, done=2]
2026-01-06 19:31:09,511 | INFO | REALTIME | Total[running=0, queued=0, done=2] | API[running=0, queued=0, done=2]
```

