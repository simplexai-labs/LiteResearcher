#!/bin/bash
# =====================================================================
#  LiteResearcher  Stage-2 RL Training  (on-policy, 48K context)
# ---------------------------------------------------------------------
#  Two-node 16xH20.  Continues training from the Stage-1 RL checkpoint
#  on a broader mix dataset (RAG synthesized + Biology + Chemistry +
#  Math + Wiki) at sequence length 48K, GRPO + TIS, ZERO2.
#
#  Prerequisites (start BEFORE launching this script):
#    1. RAG Search service                            (local_rag_diskann_server.py)
#    2. Browse service                                (browser_service.py, BROWSE_PORT=8084)
#    3. LLM Judge endpoint                            (any OpenAI-compatible chat completions)
#    4. Ray head + worker nodes set up                (see ray_setup_head.sh)
#
#  Override any of these env vars to customize:
#    PROJECT_DIR   CONDA_ENV   TRAIN_DATA   VAL_DATA   MODEL_PATH
#    CHECKPOINT_PATH   ENV_FILE   RESUME_MODE
# =====================================================================

# ---- Project root & conda env (override-able) ----
PROJECT_DIR="${PROJECT_DIR:-/share/project/wanli/Search_Agent/verl}"
CONDA_ENV="${CONDA_ENV:-/share/project/wanli/env/verl-v060}"

cd "$PROJECT_DIR" || { echo "PROJECT_DIR not found: $PROJECT_DIR"; exit 1; }
# shellcheck disable=SC1090
[ -d "$CONDA_ENV" ] && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"

set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}" && echo "CUDA_HOME=$CUDA_HOME"

# ⚠️ Use absolute paths for multinode (avoid Ray temp dir clobber)
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# ==================== Training Data ====================
# Stage-2 RL mix dataset (multi-hop + single-hop RAG + Bio/Chem/Math + Wiki)
#   Available on 🤗:
#     hf download simplex-ai-inc/LiteResearcher-Data --repo-type dataset \
#                 --local-dir ./literesearcher_data
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/data/deepresearch_rl/stage2/stage2_all_0210.parquet}"
# No separate validation set is released — see LiteResearcher-Data README.
# Default to train so verl's required data.val_files is non-empty.
VAL_DATA="${VAL_DATA:-$TRAIN_DATA}"

# Dual tools config
TOOL_CONFIG="$CONFIG_PATH/tool_config/google_search_browse_tool_config.yaml"

# ==================== Resume Configuration ====================
# Set RESUME_MODE="resume_path" + CHECKPOINT_PATH to resume from an earlier run
RESUME_MODE="${RESUME_MODE:-resume_path}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_DIR/checkpoints/qwen3_deepresearch_tis_rl/stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k/global_step_190}"
# ==================== Rollout Worker Configuration ====================
# Agent Loop Workers - 并发处理rollout样本
# 推荐值: 32 (适合196核心机器，充分利用CPU处理工具调用)
# 说明: 更多worker可提升工具调用并发度，形成LLM/CPU流水线
# 减少并发数，降低内存压力
NUM_WORKERS=8  # 或者 4

# ==================== TIS Configuration ====================
# Rollout Importance Sampling (TIS) - corrects rollout/training mismatch
# TIS corrects distribution mismatch between SGLang (BF16) and FSDP (FP32)
rollout_is_threshold=2.0              # Upper threshold for IS weights
rollout_is=true                       # Apply weights to loss (false = metrics only)
rollout_is_threshold_lower=null       # Lower threshold (null = auto 1/upper = 0.5)
rollout_is_level=token                # Aggregation level: token/sequence/geometric
rollout_is_mode=truncate              # Bounding mode: truncate (TIS) / mask (MIS)
rollout_is_veto_threshold=1e-4        # Per-token veto threshold (reject if ratio < this)

# ==================== Experiment Configuration ====================
# 在这里定义变量，供Python和tee命令共同使用
PROJECT_NAME="qwen3_deepresearch_tis_rl"
EXPERIMENT_NAME="stage2_onpolicy_new_ckpt220_bs128_all_rag-temp_1_length_48k"

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROLLOUT_DIR="$PROJECT_DIR/rollout_trajectory/${PROJECT_NAME}_${EXPERIMENT_NAME}/${TIMESTAMP}"
VALIDATION_DIR="$PROJECT_DIR/validation_trajectory/${PROJECT_NAME}_${EXPERIMENT_NAME}/${TIMESTAMP}"


# ==================== Ray & NCCL Configuration ====================
# 增加 Ray 内存阈值，避免过早触发进程终止
export RAY_memory_usage_threshold=0.95
export RAY_memory_monitor_refresh_ms=5000

# 增加各种超时设置
export RAY_gcs_server_request_timeout_seconds=1200
export RAY_health_check_timeout_ms=900000    # 从120秒增加到600秒(10分钟)，避免token等待时触发keepalive超时
export RAY_health_check_period_ms=60000      # 保持60秒检查一次

# NCCL 超时设置
export NCCL_TIMEOUT=1800

# export SWANLAB_API_KEY="<your-swanlab-key>"   # or set in env / .env
export SWANLAB_MODE="cloud"
export SWANLAB_LOG_DIR="swanlog"  # 默认日志目录


# export WANDB_API_KEY="<your-wandb-key>"       # or set in env / .env
export WANDB_DIR="$PROJECT_DIR/wandb"
export WANDB_CACHE_DIR="$PROJECT_DIR/wandb/.cache"

# ============ LLM Judge config (loaded from .env in tool_backend/) ============
# Set explicit env vars OR populate tool_backend/.env (see .env.example)
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/examples/sglang_multiturn/search_browser/tool_backend/.env}"
if [ -f "$ENV_FILE" ]; then
    echo "Loading LLM Judge config from $ENV_FILE"
    export LLM_JUDGE_API_BASE=$(grep -E "^LLM_JUDGE_API_BASE=" "$ENV_FILE" | cut -d'=' -f2-)
    export LLM_JUDGE_MODEL=$(grep -E "^LLM_JUDGE_MODEL=" "$ENV_FILE" | cut -d'=' -f2-)
    export LLM_JUDGE_MAX_RETRIES=$(grep -E "^LLM_JUDGE_MAX_RETRIES=" "$ENV_FILE" | cut -d'=' -f2-)
    export LLM_JUDGE_TIMEOUT=$(grep -E "^LLM_JUDGE_TIMEOUT=" "$ENV_FILE" | cut -d'=' -f2-)
    # Fall back to SUMMARY_* if no dedicated Judge config
    if [ -z "$LLM_JUDGE_API_BASE" ]; then
        export LLM_JUDGE_API_BASE=$(grep -E "^SUMMARY_API_BASE=" "$ENV_FILE" | cut -d'=' -f2-)
    fi
fi
echo "LLM_JUDGE_API_BASE=$LLM_JUDGE_API_BASE"
echo "LLM_JUDGE_MODEL=$LLM_JUDGE_MODEL"

# 创建目录 (packing 版本使用独立的 logs_packing 目录)
mkdir -p "$ROLLOUT_DIR"
mkdir -p "$VALIDATION_DIR"
mkdir -p "./logs_packing_new"
mkdir -p "$WANDB_DIR"
mkdir -p "$WANDB_CACHE_DIR"

# 归档旧的 log 文件 (packing 版本)
if [ -d "./logs_packing" ]; then
    # 查找所有 .log 文件（排除 backup 目录）
    OLD_LOGS=$(find ./logs_packing -maxdepth 1 -name "*.log" -type f)
    if [ -n "$OLD_LOGS" ]; then
        BACKUP_DIR="./logs_packing_new/backup/${TIMESTAMP}"
        mkdir -p "$BACKUP_DIR"
        mv ./logs_packing_new/*.log "$BACKUP_DIR/" 2>/dev/null || true
        echo "📦 已归档旧 log 到: $BACKUP_DIR"
    fi
fi


python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='google_search_browse_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=49152 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.shuffle=True \
    +data.seed=42 \
    actor_rollout_ref.model.path="${MODEL_PATH:-/share/project/wanli/RL_ckpt/onpolicy_bs128_local_rag_only_token-mean-seq-mean-temp_0.7_length_32k_nokl/global_step_220}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=115376 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio_high=0.4 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    actor_rollout_ref.actor.fsdp_config.use_zero2=True \
    actor_rollout_ref.rollout.max_model_len=51152 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=701344 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    +actor_rollout_ref.rollout.multi_turn.max_turn_length=4096 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.terminate_on_answer=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=60 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=600000 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_WORKERS} \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=701344 \
    actor_rollout_ref.ref.fsdp_config.use_zero2=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_is=${rollout_is} \
    algorithm.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_is_threshold_lower=${rollout_is_threshold_lower} \
    algorithm.rollout_is_level=${rollout_is_level} \
    algorithm.rollout_is_mode=${rollout_is_mode} \
    algorithm.rollout_is_veto_threshold=${rollout_is_veto_threshold} \
    reward_model.enable=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=verl/utils/reward_score/llm_judge_async.py \
    custom_reward_function.name=compute_score_batch \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=10 \
    trainer.test_freq=120 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    trainer.default_local_dir="$PROJECT_DIR/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    trainer.rollout_data_dir="$ROLLOUT_DIR" \
    trainer.validation_data_dir="$VALIDATION_DIR" \
    trainer.total_epochs=100 \
    trainer.resume_mode=$RESUME_MODE \
    trainer.resume_from_path="$CHECKPOINT_PATH" \
    $@ 2>&1 | tee "./logs_packing/${PROJECT_NAME}_${EXPERIMENT_NAME}_${TIMESTAMP}.log"
