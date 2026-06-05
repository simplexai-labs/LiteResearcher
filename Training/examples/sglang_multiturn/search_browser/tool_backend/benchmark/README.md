# Browse 服务压测工具

## 概述

本目录包含 Browse 服务的压测脚本，用于测试和分析服务的性能指标。

## 文件说明

| 文件 | 说明 |
|------|------|
| `browse_benchmark.py` | 基础压测脚本 |
| `browse_benchmark_detailed.py` | 详细压测脚本（带阶段时间估算） |
| `sample_queries.txt` | 样本查询文件 |

## 快速开始

### 1. 基础压测

```bash
# 使用默认查询
python browse_benchmark.py

# 指定查询
python browse_benchmark.py --queries "What is AI" "Python programming"

# 从文件读取查询
python browse_benchmark.py --query-file sample_queries.txt --concurrency 20

# 指定服务地址
python browse_benchmark.py \
    --search-url http://localhost:8017/search \
    --browse-url http://localhost:8010/query \
    --concurrency 30
```

### 2. 详细压测（推荐）

```bash
# 使用默认查询
python browse_benchmark_detailed.py

# 指定查询和并发数
python browse_benchmark_detailed.py \
    --queries "AI technology" "Machine learning" \
    --concurrency 50

# 从查询文件读取
python browse_benchmark_detailed.py \
    --query-file sample_queries.txt \
    --concurrency 30

# 直接指定 URL 列表
python browse_benchmark_detailed.py \
    --urls-file urls.txt \
    --goal "Find relevant information" \
    --concurrency 20

# 禁用服务监控
python browse_benchmark_detailed.py --no-monitor
```

## 参数说明

### browse_benchmark.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--queries` | - | 搜索查询列表 |
| `--query-file` | - | 查询文件路径 |
| `--search-url` | http://localhost:8017/search | Search 服务地址 |
| `--browse-url` | http://localhost:8010/query | Browse 服务地址 |
| `--concurrency` | 10 | 并发请求数 |
| `--search-limit` | 5 | 每个搜索返回的结果数 |

### browse_benchmark_detailed.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--queries` | - | 搜索查询列表 |
| `--query-file` | - | 查询文件路径 |
| `--urls-file` | - | URL 文件路径（直接指定 URL） |
| `--goal` | Find relevant information | URL 模式的查询目标 |
| `--search-url` | http://localhost:8017/search | Search 服务地址 |
| `--browse-url` | http://localhost:8010/query | Browse 服务地址 |
| `--stats-url` | http://localhost:8010/stats | Browse 统计接口 |
| `--concurrency` | 10 | 并发请求数 |
| `--search-limit` | 5 | 每个搜索返回的结果数 |
| `--no-monitor` | - | 禁用服务状态监控 |

## 输出文件

压测完成后会生成以下文件：

| 文件 | 说明 |
|------|------|
| `benchmark_YYYYMMDD_HHMMSS.log` | 详细日志 |
| `benchmark_YYYYMMDD_HHMMSS_result.json` | JSON 格式结果 |
| `benchmark_detailed_YYYYMMDD_HHMMSS.csv` | CSV 格式结果（详细版） |

## 统计指标

### 概览指标
- 总请求数、成功/失败数、成功率
- 缓存命中/未命中数、命中率

### 时间指标
- Search 阶段耗时
- Browse 阶段耗时
- 总耗时、QPS

### 响应时间统计
- 全部请求：avg, min, max, median, P95, P99
- 缓存命中（SQL + Summary）
- 缓存未命中（SQL + 爬虫 + Summary）

### 各阶段时间估算
- SQL 查询时间
- 爬虫抓取时间（仅缓存未命中）
- Summary LLM 时间

### 服务监控峰值
- 总并发峰值
- SQL/爬虫/Summary 各阶段并发峰值

## 示例输出

```
======================================================================
📊 Browse 压测报告
======================================================================

🔢 请求统计
   总请求数: 100
   成功: 95 | 失败: 5 | 成功率: 95.0%
   缓存命中: 30 | 未命中: 65 | 命中率: 31.6%

⏱️  耗时统计
   Search 阶段: 2.35s
   Browse 阶段: 45.67s
   总耗时: 48.02s
   QPS: 2.19

📈 响应时间（秒）
--------------------------------------------------
   全部请求:
      数量: 95
      平均: 8.45s | 中位: 6.23s
      最小: 0.35s | 最大: 45.67s
      P95: 25.34s | P99: 38.56s
   缓存命中 (SQL + Summary):
      数量: 30
      平均: 2.15s | 中位: 1.89s
   缓存未命中 (SQL + 爬虫 + Summary):
      数量: 65
      平均: 11.23s | 中位: 9.45s

🔧 各阶段时间估算（秒）
--------------------------------------------------
   SQL 查询:
      平均: 0.08s | 中位: 0.05s
   爬虫抓取:
      平均: 6.45s | 中位: 5.12s
   Summary LLM:
      平均: 3.38s | 中位: 2.89s

📡 服务监控峰值
--------------------------------------------------
   总并发峰值: 25
   SQL 并发峰值: 18
   爬虫并发峰值: 8
   Summary 并发峰值: 22

======================================================================
```

## 注意事项

1. 确保 Search 和 Browse 服务已启动
2. 大并发测试时注意服务器资源
3. 首次运行会有更多缓存未命中
4. 阶段时间为估算值，仅供参考




