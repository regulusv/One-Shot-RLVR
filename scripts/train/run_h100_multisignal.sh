#!/bin/bash
# =============================================================================
# H100 Multi-Signal RLVR Training Script (Optimized)
# =============================================================================
# Hardware: Single NVIDIA H100 (80GB VRAM)
# Model: Qwen2.5-Math-1.5B (Full Parameter Fine-Tuning)
# Reward: Multi-Signal (Outcome + Format + Reflection)
# 
# Usage:
#   Full training (default 220 steps):  ./run_h100_multisignal.sh
#   Quick test:                         ./run_h100_multisignal.sh --quick
#   Custom epochs:                      EPOCHS=220 ./run_h100_multisignal.sh
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

# Unique run naming
RUN_SUFFIX=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_BASE="Qwen2.5-Math-1.5B-pi1-multisignal-h100"

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

# WandB Configuration
export WANDB_PROJECT="${WANDB_PROJECT:-verl_one_shot_rlvr}"

# Load WandB key from secrets file if not set
if [ -z "$WANDB_API_KEY" ] && [ -f "secrets/wandb_key" ]; then
    export WANDB_API_KEY=$(cat secrets/wandb_key | tr -d '\n')
    echo "✅ Loaded WANDB_API_KEY from secrets/wandb_key"
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WANDB_API_KEY not set. Logging to console only."
    LOGGER_CONFIG="trainer.logger=['console']"
else
    echo "✅ WandB enabled. Project: $WANDB_PROJECT"
    LOGGER_CONFIG="trainer.logger=['console','wandb']"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Training Configuration (configurable via environment)
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
mkdir -p "$CHECKPOINTS_DIR"

# Mode-dependent settings (extreme acceleration; full run tuned for H100 throughput)
if [ "$QUICK_MODE" = true ]; then
    echo "🚀 Quick Mode: Running 3 ultra-fast epochs for verification"
    EPOCHS=3
    SAVE_FREQ=-1          # skip ckpt for quick run
    TEST_FREQ=-1          # skip val for quick run
    GROUP_SIZE=2          # smallest rollout group for speed
    MAX_RESPONSE_LEN=512
    EXPERIMENT_SUFFIX="quick"
else
    # High-throughput full run on H100 (stable)
    EPOCHS="${EPOCHS:-220}"            # hard stop target
    SAVE_FREQ="${SAVE_FREQ:-50}"
    TEST_FREQ="${TEST_FREQ:-50}"
    GROUP_SIZE="${GROUP_SIZE:-4}"      # stable rollout
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1536}"
    EXPERIMENT_SUFFIX="fullft"
fi

EXPERIMENT_NAME="${EXPERIMENT_BASE}-${EXPERIMENT_SUFFIX}-${RUN_SUFFIX}"
LOG_FILE="verl_h100_multisignal_${EXPERIMENT_SUFFIX}_${RUN_SUFFIX}.log"

# ─────────────────────────────────────────────────────────────────────────────
# Launch Training
# ─────────────────────────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/train/one_shot_rlvr/pi1_r128.parquet \
    data.val_files=data/test/math500.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=500 \
    data.max_prompt_length=1024 \
    data.max_response_length="$MAX_RESPONSE_LEN" \
    reward_model.reward_manager='multi_signal' \
    actor_rollout_ref.model.path='Qwen/Qwen2.5-Math-1.5B' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    ++actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    ++actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n="$GROUP_SIZE" \
    ++actor_rollout_ref.rollout.n_val=1 \
    ++actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    ++actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    ++actor_rollout_ref.model.use_liger=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    "$LOGGER_CONFIG" \
    trainer.project_name='verl_one_shot_rlvr' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.checkpoints_dir="$CHECKPOINTS_DIR" \
    ++trainer.val_before_train=False \
    ++trainer.resume_mode=off \
    ++trainer.resume_from_path=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs="$EPOCHS" \
    ++trainer.total_training_steps="$EPOCHS" 2>&1 | tee "$LOG_FILE"
