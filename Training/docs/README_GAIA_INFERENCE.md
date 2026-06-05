# GAIA Test Inference with Real Search Tools

## Overview

This script performs **inference only** (no training) on the GAIA test set using real external search tools:
- **Google Search**: Serper API (http://47.111.147.142:8865/search)
- **Browse**: Crawl4AI + vLLM (http://47.111.147.142:8866/query)

## Prerequisites

### 1. Backend Services Running

Ensure both backend services are running:

```bash
# Check Search service (Serper API)
curl http://47.111.147.142:8865/health

# Check Browse service (Crawl4AI + vLLM)
curl http://47.111.147.142:8866/health
```

### 2. Data File Available

```bash
ls -lh /share/project/wanli/Search_Agent/verl/data/benchmarks_processed/individual/GAIA_test.parquet
```

### 3. Model Downloaded

Default model: `/share/project/wanli/model/Qwen2.5-3B-Instruct`

## Usage

### Basic Usage (Default Model)

```bash
cd /share/project/wanli/Search_Agent/verl
bash examples/sglang_multiturn/search_browser/inference_gaia_real_search.sh
```

### Custom Model

```bash
bash examples/sglang_multiturn/search_browser/inference_gaia_real_search.sh /path/to/your/model
```

### Additional Overrides

```bash
bash examples/sglang_multiturn/search_browser/inference_gaia_real_search.sh \
    actor_rollout_ref.rollout.temperature=0.7 \
    data.val_batch_size=32
```

## Key Configuration

### Inference Mode Settings

The script uses these key settings for inference-only mode:

```yaml
trainer.val_before_train=True   # Run validation before training
trainer.total_epochs=0          # No training epochs
actor_rollout_ref.rollout.n=1   # Single response per prompt (not 5 for training)
```

### Tool Concurrency (Conservative Settings)

```yaml
# Search Tool (Serper API)
num_workers: 30          # Backend supports 50 max concurrent
rate_limit: 30           # Conservative rate to avoid API limits
timeout: 30              # Serper is fast

# Browse Tool (Crawl4AI + vLLM)
num_workers: 5           # Backend supports 5 max concurrent
rate_limit: 5            # Match backend capacity
timeout: 180             # Crawl + vLLM needs time
```

### Multi-turn Agent Configuration

```yaml
max_assistant_turns: 10              # Max conversation turns
max_tool_response_length: 15000      # Tool response length limit
tool_response_truncate_side: left    # Truncate old content
max_parallel_calls: 1                # Tools per turn (per request)
```

## Output

### Directory Structure

```
validation_trajectory/gaia_inference_real_search_YYYYMMDD_HHMMSS/
├── data_0.parquet          # Validation results
├── metadata.json           # Run metadata
└── ...
```

### Log File

```
logs/gaia_inference_real_search_YYYYMMDD_HHMMSS.log
```

## Analysis

### Visualize Trajectories

```bash
python scripts/visualize_trajectory.py validation_trajectory/gaia_inference_*/
```

### Calculate Metrics

```bash
# If you have a scoring script
python verl/utils/reward_score/your_scoring_script.py \
    --data validation_trajectory/gaia_inference_*/data_0.parquet
```

## Troubleshooting

### Backend Service Issues

**Problem**: Connection refused or timeout errors

**Solution**:
```bash
# Check service status
curl http://47.111.147.142:8865/health  # Search
curl http://47.111.147.142:8866/health  # Browse

# Check service logs
ssh user@47.111.147.142
cd /path/to/service
tail -f logs/service.log
```

### Out of Memory (OOM)

**Problem**: GPU memory overflow

**Solution**: Reduce batch size
```bash
bash inference_gaia_real_search.sh \
    data.val_batch_size=32 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
```

### Slow Inference

**Problem**: Inference too slow

**Solution**: Adjust concurrency (if backends can handle it)
```bash
# Edit tool config: real_search.yaml
# Increase num_workers and rate_limit (within backend capacity)
```

### Tool Call Failures

**Problem**: "Browse tool failed" or "Search failed" errors

**Diagnosis**:
```bash
# Check backend stats
curl http://47.111.147.142:8865/stats  # Search stats
curl http://47.111.147.142:8866/stats  # Browse stats
```

**Solution**:
- If search fails: Check Serper API key and rate limits
- If browse fails: Check Crawl4AI service and vLLM availability

## Advanced Options

### Adjust Tool Behavior

Edit `examples/sglang_multiturn/config/tool_config/real_search.yaml`:

```yaml
tools:
  - class_name: verl.tools.google_search_tool.GoogleSearchTool
    config:
      num_workers: 30      # Increase if backend can handle
      rate_limit: 30       # Match num_workers
      timeout: 30          # Increase if slow network
```

### Change Model Generation Parameters

```bash
bash inference_gaia_real_search.sh \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.max_new_tokens=2048
```

### Disable WandB Logging

```bash
bash inference_gaia_real_search.sh \
    trainer.logger='["console"]'
```

## Expected Performance

### GAIA Test Set

- **Size**: ~300 questions
- **Time**: ~2-3 hours (depends on tool response time)
- **Success Rate**: Varies by model and tool quality

### Bottlenecks

- **Search**: Fast (~1-2s per call with Serper API)
- **Browse**: Slow (~30-60s per call with Crawl4AI + vLLM)
- **Overall**: Dominated by browse tool latency

## Comparison with Training Script

| Feature | Training Script | Inference Script |
|---------|----------------|------------------|
| `total_epochs` | 1 | 0 (no training) |
| `val_before_train` | False | True |
| `rollout.n` | 5 (GRPO sampling) | 1 (single response) |
| `train_batch_size` | 512 | N/A |
| `val_batch_size` | 256 | 64 (smaller) |
| Purpose | Train model | Evaluate model |

## Files

- **Script**: `inference_gaia_real_search.sh`
- **Config**: `examples/sglang_multiturn/config/google_search_browse_multiturn_grpo.yaml`
- **Tool Config**: `examples/sglang_multiturn/config/tool_config/real_search.yaml`
- **Data**: `/share/project/wanli/Search_Agent/verl/data/benchmarks_processed/individual/GAIA_test.parquet`
