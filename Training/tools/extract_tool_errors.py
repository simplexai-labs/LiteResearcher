#!/usr/bin/env python3
"""提取工具错误响应"""
import json
from pathlib import Path
import sys

error_file = Path("/share/project/wanli/Search_Agent/verl/rollout_trajectory/qwen3_deepresearch_tis_rl_stage2_onpolicy_bs128_all_rag-temp_1_length_48k/browse_error_analysis/trajectories_with_errors_incorrect.jsonl")

with open(error_file, 'r') as f, open('/tmp/tool_responses.txt', 'w') as out:
    line = f.readline()
    data = json.loads(line)
    traj = data['full_trajectory']
    output = traj['output']

    # 查找所有 </invoke> 并提取后面的响应
    idx = 0
    count = 0

    while True:
        invoke_end = output.find('</invoke>', idx)
        if invoke_end == -1 or count >= 5:
            break

        # 获取响应
        response = output[invoke_end + 9:invoke_end + 1000]

        out.write(f"\n{'='*80}\n")
        out.write(f"工具响应 #{count+1}:\n")
        out.write('='*80 + '\n')
        out.write(response + '\n\n')

        idx = invoke_end + 9
        count += 1

print("结果已保存到 /tmp/tool_responses.txt", file=sys.stderr)
