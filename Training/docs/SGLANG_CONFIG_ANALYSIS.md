# SGLang Router 熔断错误配置分析

## 🔍 问题诊断

根据日志分析，出现 **11,285 次 "No workers (熔断器打开)" 错误**，这通常是由以下原因引起的：

### 1. 后端配置问题（最可能）

#### 当前配置分析

**Router 配置**：
- `ROUTER_MAX_CONCURRENT=8000` - Router 最大并发处理数
- `ROUTER_QUEUE_SIZE=10000` - Router 队列大小
- `ROUTER_QUEUE_TIMEOUT=600` - 队列超时时间

**Worker 配置**：
- `WORKER_MAX_RUNNING_REQUESTS=2048` - 每个 Worker 的最大并发请求数
- 8 个 Workers = **16,384 总容量**

**容量计算**：
```
Router 总容量 = 8000 (并发) + 10000 (队列) = 18,000
Worker 总容量 = 2048 × 8 = 16,384
实际可用 = min(18000, 16384) = 16,384
```

#### 潜在问题

1. **队列大小可能不足**：
   - 当请求速率 > 16,384 时，队列会满
   - 队列满后，新请求会被拒绝，触发熔断

2. **缺少请求超时配置**：
   - 如果 Worker 处理请求时间过长，可能导致请求堆积
   - 没有 `--request-timeout-secs` 参数来限制单个请求的最大时间

3. **Worker 健康检查问题**：
   - 如果某些 Worker 崩溃或过载，Router 可能无法及时检测
   - 导致 Router 继续向不健康的 Worker 发送请求

### 2. 前端配置问题

**browse_service.py 配置**：
- `SUMMARY_MAX_CONCURRENT=500` - 前端并发限制

**问题**：
- 如果前端有多个实例，总并发可能 > 500
- 或者前端没有正确限制并发，导致请求速率过高

## ✅ 优化方案

### 方案 1：优化 Router 配置（推荐）

**修改 `qwen3_4B_FP8_separate.sh`**：

```bash
# 增加队列大小
ROUTER_QUEUE_SIZE=15000  # 从 10000 增加到 15000

# 添加请求超时（如果 sglang_router 支持）
ROUTER_REQUEST_TIMEOUT=300  # 单个请求最大 5 分钟
```

**启动 Router 时添加参数**：
```bash
python -m sglang_router.launch_router \
    --host "$ROUTER_HOST" \
    --port "$ROUTER_PORT" \
    --worker-urls ${WORKER_URLS} \
    --policy cache_aware \
    --max-concurrent-requests ${ROUTER_MAX_CONCURRENT} \
    --queue-size ${ROUTER_QUEUE_SIZE} \
    --queue-timeout-secs ${ROUTER_QUEUE_TIMEOUT} \
    --request-timeout-secs ${ROUTER_REQUEST_TIMEOUT} \  # 新增
    --prometheus-port 9091
```

### 方案 2：优化 Worker 配置

**增加 Worker 并发容量**：
```bash
WORKER_MAX_RUNNING_REQUESTS=2500  # 从 2048 增加到 2500
```

**总容量变化**：
```
之前: 2048 × 8 = 16,384
之后: 2500 × 8 = 20,000
```

**注意**：需要确保 GPU 内存足够，否则可能导致 OOM。

### 方案 3：优化前端配置

**调整 browse_service.py**：
```bash
# 降低并发限制，避免超过后端容量
SUMMARY_MAX_CONCURRENT=400  # 从 500 降低到 400

# 或者根据实际容量计算
# 假设有 2 个 browse_service 实例
# 每个实例: 400 × 2 = 800 < 16384 ✓
```

### 方案 4：添加健康检查和自动恢复

**在启动脚本中添加 Worker 健康检查**：
```bash
# 在启动 Router 前，检查所有 Workers 是否健康
for port in 35000 35219 35438 35657 35876 36095 36314 36533; do
    if ! curl -s --max-time 5 "http://${ROUTER_HOST}:${port}/health" > /dev/null; then
        echo "⚠️  Worker ${port} 不健康，请检查"
    fi
done
```

## 🔧 诊断步骤

### 1. 运行诊断脚本

```bash
cd /share/project/wanli/Search_Agent/verl/examples/sglang_multiturn/search_browser/sgl_serve
chmod +x diagnose_sglang.sh
./diagnose_sglang.sh
```

### 2. 检查 Router 日志

```bash
# 查看 Router 日志，寻找熔断相关错误
tail -f log_sglang/*_separate.log | grep -i "circuit\|熔断\|no.*worker"
```

### 3. 检查 Worker 日志

```bash
# 检查每个 Worker 的日志
for i in {0..7}; do
    echo "=== Worker $i ==="
    tail -20 log_sglang/*_worker_${i}.log
done
```

### 4. 监控实时状态

```bash
# 使用 Prometheus 监控（如果启用）
curl http://172.24.132.205:9091/metrics | grep -i "circuit\|queue\|worker"
```

## 📊 配置建议总结

### 推荐配置（针对当前问题）

**Router**：
- `ROUTER_MAX_CONCURRENT=8000` ✓ (保持不变)
- `ROUTER_QUEUE_SIZE=15000` ⬆️ (从 10000 增加)
- `ROUTER_QUEUE_TIMEOUT=600` ✓ (保持不变)
- `ROUTER_REQUEST_TIMEOUT=300` ➕ (新增)

**Worker**：
- `WORKER_MAX_RUNNING_REQUESTS=2048` ✓ (保持不变，除非 GPU 内存充足)
- `WORKER_SCHEDULE_CONSERVATIVENESS=0.3` ✓ (保持不变)

**前端 (browse_service)**：
- `SUMMARY_MAX_CONCURRENT=400` ⬇️ (从 500 降低，更保守)

### 容量验证

```
优化后容量:
Router: 8000 + 15000 = 23,000
Worker: 2048 × 8 = 16,384
实际可用: min(23000, 16384) = 16,384

前端限制: 400 (单实例)
如果 2 个实例: 400 × 2 = 800 << 16,384 ✓
```

## 🚨 如果问题仍然存在

1. **检查 Worker 是否真的在处理请求**：
   ```bash
   # 查看 Worker 的 GPU 使用率
   nvidia-smi
   ```

2. **检查是否有 Worker 崩溃**：
   ```bash
   ps aux | grep sglang.launch_server
   ```

3. **检查网络连接**：
   ```bash
   # 测试 Router 到 Worker 的连接
   for port in 35000 35219 35438 35657 35876 36095 36314 36533; do
       curl -v http://172.24.132.205:${port}/health
   done
   ```

4. **查看详细错误日志**：
   - Router 日志：`log_sglang/*_separate.log`
   - Worker 日志：`log_sglang/*_worker_*.log`
   - Browse Service 日志：`log_browse/*.log`

## 📝 下一步

1. ✅ 运行诊断脚本确认问题
2. ✅ 优化 Router 配置（增加队列大小）
3. ✅ 检查 Worker 健康状态
4. ✅ 调整前端并发限制
5. ✅ 监控修复后的效果


