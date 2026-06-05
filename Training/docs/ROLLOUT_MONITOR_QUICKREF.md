# 🎯 Rollout监控快速参考

## 看到这样的输出？

```
[Step 1] Agent Rollout: 1%|  | 2/256 [00:39<1:18:00, 18.43s/sample]
```

### ❓ 疑问：为什么是256而不是2048？

**答案**：这是**正确的**！

```
原始batch: 256 prompts
    ↓
× GRPO n=8
    ↓
总样本: 2048
    ↓
÷ 8 workers
    ↓
每个worker: 256 samples  ← 你看到的数字！
```

## ✅ 正确理解

### 输出格式（更新后）

```
[Step 1] Worker 0/8 Rollout (Global: 2048 samples): 45%|████| 115/256
         ↑      ↑                    ↑                      ↑    ↑
      Worker ID  总worker数        全局样本数              进度  当前/本worker总数
```

### 关键点

| 项目 | 说明 |
|-----|------|
| **256** | 每个worker处理的样本数 |
| **2048** | 全局总样本数（256×8）|
| **8** | worker数量（并发执行）|
| **进度条** | 显示**单个worker**的进度 |
| **全局进度** | 在定期统计中显示（估算）|

## 📊 完整输出示例

### 1. 进度条（8个worker同时运行）

```bash
# Terminal 会显示多个进度条（每个worker一个）
[Step 1] Worker 0/8 Rollout (Global: 2048): 45%|████| 115/256 [02:15<02:47]
[Step 1] Worker 1/8 Rollout (Global: 2048): 42%|████| 108/256 [02:15<02:50]
[Step 1] Worker 2/8 Rollout (Global: 2048): 47%|████| 120/256 [02:15<02:44]
[Step 1] Worker 3/8 Rollout (Global: 2048): 44%|████| 113/256 [02:15<02:48]
[Step 1] Worker 4/8 Rollout (Global: 2048): 46%|████| 118/256 [02:15<02:45]
[Step 1] Worker 5/8 Rollout (Global: 2048): 43%|████| 110/256 [02:15<02:49]
[Step 1] Worker 6/8 Rollout (Global: 2048): 45%|████| 116/256 [02:15<02:46]
[Step 1] Worker 7/8 Rollout (Global: 2048): 48%|████| 123/256 [02:15<02:43]
```

### 2. 定期统计（每个worker，每5%）

```bash
[Step 1] Worker 0/8 Rollout Progress: 128/256 completed
  🌐 Global Progress: ~1024/2048 (estimated)
  ⏱️  Duration: avg=8.3s, p50=7.9s, p95=15.2s
  ✅ Success: 128, ❌ Failed: 0
```

### 3. 最终汇总（每个worker完成时）

```bash
================================================================================
[Step 1] Worker 0/8 - Rollout Complete Summary
================================================================================
🌐 Global Total:      2048 samples across 8 workers
📦 This Worker:       256 samples
✅ Completed:         256 (100.0%)
⏱️  Total Time:        38.2s (0.6min)
⚡ Throughput:        6.70 samples/s

🔄 Agent Statistics:
   Avg Turns/Sample:  3.2
   Avg Tools/Sample:  4.8

⏱️  Sample Duration:
   Min:  3.2s
   P50:  7.9s
   P95:  15.2s
   Max:  23.4s
================================================================================
```

## 🎯 如何看全局进度？

### 方法1：观察所有worker（近似）

```python
# 假设8个worker进度相似
Worker 0: 115/256 = 45%
Worker 1: 108/256 = 42%
...
平均: ~45%

全局进度 ≈ 45%
实际完成 ≈ 2048 × 45% = 920 samples
```

### 方法2：看定期统计中的估算

```
🌐 Global Progress: ~1024/2048 (estimated)
```

### 方法3：查看WandB指标（最准确）

训练结束后在WandB中看：
```
rollout/total_samples = 2048
rollout/completed = 2048
rollout/success_rate = 1.0
```

## 💡 常见疑问

### Q1: 为什么不直接显示2048？
**A**: 因为每个worker是独立的Ray Actor，无法实时获取其他worker的进度（需要跨进程通信，开销大）。

### Q2: 能否显示真实的全局进度？
**A**: 可以，但需要：
1. 建立跨worker通信机制（Ray shared state）
2. 定期同步进度（增加网络开销）
3. 可能影响性能

**当前方案**：每个worker显示自己的准确进度，全局进度作为估算参考。

### Q3: 哪个worker最慢？
**A**: 看最终汇总，对比各worker的"Total Time"和"Throughput"。

### Q4: 能否调整worker数量？
**A**: 可以，在配置中修改：
```yaml
actor_rollout_ref.rollout.agent.num_workers=16
```

但注意：
- worker数 ≤ CPU核心数（推荐）
- 太多worker会导致竞争，反而变慢

## 🔧 快速验证

### 检查配置

```bash
# 查看实际运行的worker数
grep "agent_loop_worker" logs/your_log.log | wc -l
```

### 计算验证

```python
原始batch = 256
rollout.n = 8
num_workers = 8

单worker样本数 = (256 × 8) / 8 = 256  ✓
全局样本数 = 256 × 8 = 2048  ✓
```

## 📚 详细文档

- **分布式说明**: `docs/ROLLOUT_MONITOR_DISTRIBUTED.md`
- **使用指南**: `docs/ROLLOUT_PROGRESS_MONITORING.md`
- **配置参考**: `docs/ROLLOUT_MONITOR_CONFIG.md`

## ✅ 总结

**你看到的是对的！**

- ✅ 256 = 每个worker的样本数
- ✅ 2048 = 全局样本数（256 × 8 workers）
- ✅ 进度条显示单worker进度（准确）
- ✅ 全局进度在统计中显示（估算）
- ✅ WandB记录真实全局指标

**现在理解了吗？** 🎉
