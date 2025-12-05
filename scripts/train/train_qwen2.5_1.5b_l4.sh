#!/bin/bash
# =============================================================================
# L4 Multi-Signal RLVR Training Script
# =============================================================================
# Hardware: NVIDIA L4 (24GB VRAM) - Memory constrained
# Model: Qwen2.5-Math-1.5B with LoRA (memory optimization)
# =============================================================================
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export REWARD_WEIGHTS="${REWARD_WEIGHTS:-1.0,0.5,0.5}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl_few_shot}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/train/one_shot_rlvr/pi1_r128_multisignal.parquet \
    data.val_files=data/test/math500.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=100 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    reward_model.reward_manager='multi_signal' \
    actor_rollout_ref.model.path='Qwen/Qwen2.5-Math-1.5B' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.n=2 \
    +actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_few_shot' \
    trainer.experiment_name='Qwen2.5-Math-1.5B-pi1_r128-multisignal' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=500 \
    +actor_rollout_ref.model.lora_rank=64 \
    +actor_rollout_ref.model.lora_alpha=32 \
    +actor_rollout_ref.model.target_modules='all-linear' \
    +actor_rollout_ref.model.use_liger=False

