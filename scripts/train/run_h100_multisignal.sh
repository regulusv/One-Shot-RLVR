#!/bin/bash
# =============================================================================
# H100 Multi-Signal RLVR Training Script (Optimized)
# =============================================================================
# Hardware: Single NVIDIA H100 (80GB VRAM)
# Model: Qwen2.5-Math-1.5B (Full Parameter Fine-Tuning)
# Reward: Multi-Signal (Outcome + Format + Reflection)
# 
# Usage:
#   Full training (2000 epochs):  ./run_h100_multisignal.sh
#   Quick test (10 epochs):       ./run_h100_multisignal.sh --quick
#   Custom epochs:                EPOCHS=500 ./run_h100_multisignal.sh
# =============================================================================
set -x

# ─────────────────────────────────────────────────────────────────────────────
# Parse Arguments
# ─────────────────────────────────────────────────────────────────────────────
QUICK_MODE=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK_MODE=true ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# H100-Optimized Environment Variables
# ─────────────────────────────────────────────────────────────────────────────
# Use Flash Attention backend for vLLM (faster than XFORMERS on H100)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Memory optimization for large batch training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optimize CUDA connections for single GPU
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Enable TF32 for faster matmul on Hopper (H100)
export NVIDIA_TF32_OVERRIDE=1

# Multi-Signal Reward Weights: R = α·R_verify + β·R_format + γ·R_reflect
export REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0,0.5,0.5}"

# WandB Configuration (set via environment, not hardcoded)
export WANDB_PROJECT="${WANDB_PROJECT:-verl_one_shot_rlvr}"
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WANDB_API_KEY not set. Logging to console only."
    LOGGER_CONFIG="trainer.logger=['console']"
else
    LOGGER_CONFIG="trainer.logger=['console','wandb']"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Training Configuration (configurable via environment)
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
mkdir -p "$CHECKPOINTS_DIR"

# Mode-dependent settings
if [ "$QUICK_MODE" = true ]; then
    echo "🚀 Quick Mode: Running 10 epochs for verification"
    EPOCHS=10
    SAVE_FREQ=5
    TEST_FREQ=5
    GROUP_SIZE=8
    MAX_RESPONSE_LEN=2048
    EXPERIMENT_SUFFIX="quick"
else
    EPOCHS="${EPOCHS:-2000}"
    SAVE_FREQ="${SAVE_FREQ:-100}"
    TEST_FREQ="${TEST_FREQ:-50}"
    GROUP_SIZE="${GROUP_SIZE:-32}"
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-3072}"
    EXPERIMENT_SUFFIX="fullft"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Launch Training
# ─────────────────────────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/train/one_shot_rlvr/pi1_r128.parquet \
    data.val_files=data/test/math500.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=500 \
    data.max_prompt_length=1024 \
    data.max_response_length="$MAX_RESPONSE_LEN" \
    reward_model.reward_manager='multi_signal' \
    actor_rollout_ref.model.path='Qwen/Qwen2.5-Math-1.5B' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    +actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.88 \
    actor_rollout_ref.rollout.n="$GROUP_SIZE" \
    +actor_rollout_ref.rollout.n_val=1 \
    +actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    +actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    +actor_rollout_ref.model.use_liger=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    "$LOGGER_CONFIG" \
    trainer.project_name='verl_one_shot_rlvr' \
    trainer.experiment_name="Qwen2.5-Math-1.5B-pi1-multisignal-h100-${EXPERIMENT_SUFFIX}" \
    trainer.checkpoints_dir="$CHECKPOINTS_DIR" \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs="$EPOCHS" 2>&1 | tee "verl_h100_multisignal_${EXPERIMENT_SUFFIX}.log"
