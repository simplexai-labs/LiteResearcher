┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. RolloutManager.generate_sequences (agent_loop.py:903)                        │
│    输入: prompts (2048 个样本)                                                   │
│    分发给 N 个 agent_loop_workers (假设 N=8)                                     │
│    每个 worker 处理 256 个样本                                                   │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. AgentLoopWorker.generate_sequences (agent_loop.py:444)                       │
│    每个 worker 并发处理 256 个样本                                               │
│    每个样本运行 ToolAgentLoop.run()                                             │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. ToolAgentLoop.run() (tool_agent_loop.py:118)                                 │
│    每个样本创建 metrics = {}                                                     │
│    在 _handle_generating_state 中:                                              │
│      if output.finish_reason == "length":                                       │
│          metrics["turn_truncated_count"] = 1 (或累加)                           │
│    返回 AgentLoopOutput(metrics=agent_data.metrics, ...)                        │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 4. _postprocess (agent_loop.py:722)                                             │
│    inputs: 256 个 _InternalAgentLoopOutput                                      │
│    metrics = [input.metrics.model_dump() for input in inputs]                   │
│    → metrics 是一个包含 256 个 dict 的 list                                      │
│    每个 dict 包含 {"generate_sequences": x, "tool_calls": y,                    │
│                   "turn_truncated_count": 0 或 n, ...}                          │
│    返回 DataProto(meta_info={"metrics": metrics, ...})                          │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 5. RolloutManager.generate_sequences (agent_loop.py:929)                        │
│    outputs: 8 个 worker 的返回结果                                               │
│    metrics = [out.meta_info.pop("metrics", []) for out in outputs]              │
│    → metrics 是 list[list[dict]]，形状 [8][256]                                 │
│                                                                                  │
│    调用 _performance_metrics(metrics, output)                                    │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 6. _performance_metrics (agent_loop.py:955)                                     │
│    # 展平所有样本的 metrics                                                      │
│    turn_truncated_counts = [metric.get("turn_truncated_count", 0)               │
│                             for chunk in metrics for metric in chunk]           │
│    → turn_truncated_counts 是 list，长度 2048                                   │
│                                                                                  │
│    total_samples = 2048                                                          │
│    samples_with_truncation = sum(1 for c in turn_truncated_counts if c > 0)     │
│    sample_ratio = samples_with_truncation / 2048                                 │
│                                                                                  │
│    timing["agent_loop/turn_truncated/sample_ratio"] = sample_ratio              │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 7. ray_trainer.py:1261                                                          │
│    timing_raw.update(gen_batch_output.meta_info["timing"])                      │
│    → timing_raw 包含 "agent_loop/turn_truncated/sample_ratio"                   │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 8. compute_timing_metrics (metric_utils.py:410)                                 │
│    return {f"timing_s/{name}": value for name, value in timing_raw.items(), ...}│
│    → {"timing_s/agent_loop/turn_truncated/sample_ratio": 0.15, ...}             │
└─────────────────────────────────────┬───────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 9. logger.log (ray_trainer.py:1497)                                             │
│    logger.log(data=metrics, step=self.global_steps)                             │
│    → Wandb 记录 "timing_s/agent_loop/turn_truncated/sample_ratio"               │
└─────────────────────────────────────────────────────────────────────────────────┘