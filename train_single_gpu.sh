#!/bin/bash
# Single GPU Training Script for One-Shot RLVR
# Based on original training_1.5b_dsr_sub.sh, adapted for single GPU (T4) and small model
# Reproduces GRPO training method

set -x

# Set environment variables
export VLLM_ATTENTION_BACKEND=XFORMERS
export TRANSFORMERS_NO_FLASH_ATTENTION_2=1

# Data file paths (relative to project root)
TRAIN_FILE="data/train/one_shot_rlvr/dsr_sub.parquet"
VAL_FILE="data/test/math500.parquet"

# Single GPU PPO/GRPO training
python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files="$TRAIN_FILE" \
 data.val_files="$VAL_FILE" \
 data.train_batch_size=8 \
 data.val_batch_size=32 \
 data.max_prompt_length=1024 \
 data.max_response_length=2048 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path='Qwen/Qwen2.5-Math-1.5B' \
 +actor_rollout_ref.model.override_config.attn_implementation=eager \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=False \
 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000 \
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
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.rollout.n=4 \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console'] \
 trainer.project_name='verl_few_shot' \
 trainer.experiment_name='Qwen2.5-Math-1.5B-dsr_sub-single-gpu' \
 trainer.checkpoints_dir=$HOME/ckpts/ \
 +trainer.val_before_train=True \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=20 \
 trainer.test_freq=20 \
 trainer.default_hdfs_dir=null \
 trainer.total_epochs=2000 2>&1 | tee verl_demo.log

