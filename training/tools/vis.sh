#!/bin/bash
cd /share/project/wanli/Search_Agent/verl

# 使用方法: bash vis.sh <jsonl文件路径> [端口号]
# 示例: bash vis.sh /path/to/file.jsonl 7223

JSONL_FILE="${1:-/share/project/wanli/Search_Agent/verl/rollout_trajectory/qwen3_deepresearch_tis_rl_stage3_v4_resume_from_v2_ckpt_360/20260404_102459/70.jsonl}"
PORT="${2:-7824}"       

python /share/project/wanli/Search_Agent/verl/tools/rollout_viewer.py "$JSONL_FILE" --port "$PORT"
