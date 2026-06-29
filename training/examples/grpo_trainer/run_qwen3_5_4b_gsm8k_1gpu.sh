#!/bin/bash
# Minimal: 1 GPU, very small, only verify model loads + 1 step works
set -x
cd /share/project/wanli/Search_Agent/verl
source /root/anaconda3/etc/profile.d/conda.sh
conda activate /share/project/wanli/env/verl-cp312-qwen35

export HYDRA_FULL_ERROR=1
ulimit -n 65535

PROJECT_DIR="/share/project/wanli/Search_Agent/verl"
MODEL_PATH=/share/project/kunluo/Projects/ScienceAgent/GeneralSearchAgent/SearchAgent-SLM/train/trained_results/browsecomp_qwen35_4b_megatron_0417_recipe/qwen3.5_4b_0417_recipe_2node_v0/v8-20260508-151127/checkpoint-105
TRAIN_DATA="$PROJECT_DIR/data/gsm8k/train.parquet"
VAL_DATA="$PROJECT_DIR/data/gsm8k/test.parquet"
PROJECT_NAME="qwen3_5_smoke"
EXPERIMENT_NAME="gsm8k_1gpu_$(date +%Y%m%d_%H%M%S)"

mkdir -p logs_qwen3_5_smoke

# CUDA / Ray safety
# CUDA_VISIBLE_DEVICES inherited from caller (allow GPU selection)
export RAY_DEDUP_LOGS=0
export RAY_BACKEND_LOG_LEVEL=info
export NCCL_TIMEOUT=1800
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export WANDB_MODE=disabled
# sglang 0.5.9 cuDNN compatibility check: our cudnn 9.10 has a known torch 2.9.1 Conv3d bug.
# We don't use Conv3d (text-only RL training), so skip the check.
export SGLANG_DISABLE_CUDNN_CHECK=1

# CUDA 12.8 toolchain (matches torch cu128)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_model_len=512 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.default_local_dir="$PROJECT_DIR/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME" \
    "$@" 2>&1 | tee "logs_qwen3_5_smoke/${EXPERIMENT_NAME}.log"
