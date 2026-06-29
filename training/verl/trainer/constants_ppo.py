# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
        # SwanLab 配置（必须传递到 Ray worker）
        "SWANLAB_API_KEY": os.environ.get("SWANLAB_API_KEY", ""),
        "SWANLAB_MODE": os.environ.get("SWANLAB_MODE", "cloud"),
        "SWANLAB_LOG_DIR": os.environ.get("SWANLAB_LOG_DIR", "swanlog"),
        # LLM Judge 配置（必须传递到 Ray worker）
        "LLM_JUDGE_API_BASE": os.environ.get("LLM_JUDGE_API_BASE", ""),
        "LLM_JUDGE_MODEL": os.environ.get("LLM_JUDGE_MODEL", ""),
        "LLM_JUDGE_MAX_RETRIES": os.environ.get("LLM_JUDGE_MAX_RETRIES", "3"),
        "LLM_JUDGE_TIMEOUT": os.environ.get("LLM_JUDGE_TIMEOUT", "60"),
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    # 这些环境变量必须传递给 Ray worker，即使主进程中已设置
    must_pass_keys = {
        "SWANLAB_API_KEY", "SWANLAB_MODE", "SWANLAB_LOG_DIR",
        "LLM_JUDGE_API_BASE", "LLM_JUDGE_MODEL", "LLM_JUDGE_MAX_RETRIES", "LLM_JUDGE_TIMEOUT"
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None and key not in must_pass_keys:
            runtime_env["env_vars"].pop(key, None)
    return runtime_env
