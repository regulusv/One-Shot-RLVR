## Compute Environment Setup

### Lambda Labs H100 Instance

We use a Lambda Labs H100 PCIe instance for training. The instance is configured as follows:

#### Instance Specifications
- **Type**: `gpu_1x_h100_pcie` (Lambda Labs)
- **IP Address**: `209.20.157.194`
- **Region**: `us-west-3`
- **GPU**: NVIDIA H100 PCIe (81,559 MiB / ~80GB VRAM)
- **GPU Driver**: 570.148.08
- **OS**: Ubuntu 22.04 (Linux 6.8.0-60-generic)
- **CPU**: x86_64
- **System Memory**: 221 GiB total (205 GiB available)
- **Disk Space**: 993 GB total (361 GB available, 64% used)

#### Software Environment
- **Python**: 3.10.12 (`/usr/bin/python3`)
- **PyTorch**: 2.9.0+cu128
- **CUDA**: 12.8 (available and working)
- **vLLM**: 0.12.0
- **Flash Attention**: Installed but may require rebuild (symbol compatibility issue with PyTorch 2.9.0)

#### Connecting to the Instance

**SSH Connection:**
```bash
# SSH key location
SSH_KEY="/Users/gilbert/lambda-ssh-1.pem"
LAMBDA_IP="209.20.157.194"
LAMBDA_USER="ubuntu"

# Connect to instance
ssh -i "$SSH_KEY" "$LAMBDA_USER@$LAMBDA_IP"

# Test connection (from local machine)
ssh -i /Users/gilbert/lambda-ssh-1.pem -o ConnectTimeout=10 \
    ubuntu@209.20.157.194 "echo 'Connection successful'"
```

**Set SSH key permissions:**
```bash
chmod 600 /Users/gilbert/lambda-ssh-1.pem
```

#### Verifying the Environment

Once connected, verify the compute environment:

```bash
# Check GPU status
nvidia-smi

# Check system info
uname -a
df -h    # disk space
free -h  # memory usage

# Verify Python and CUDA
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"
```

#### Syncing Code to Lambda Instance

From your local machine, sync the project to the Lambda instance:

```bash
# From project root directory
rsync -avz -e "ssh -i /Users/gilbert/lambda-ssh-1.pem" \
    --exclude '.git' \
    --exclude 'checkpoints' \
    --exclude 'outputs' \
    --exclude 'wandb' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    /Users/gilbert/Documents/One-Shot-RLVR/ \
    ubuntu@209.20.157.194:~/One-Shot-RLVR/
```

#### Running Training on Lambda Instance

After syncing code and verifying the environment:

```bash
# SSH into instance
ssh -i /Users/gilbert/lambda-ssh-1.pem ubuntu@209.20.157.194

# On the instance
cd ~/One-Shot-RLVR

# Run verification script (optional)
./scripts/train/verify_h100_multisignal.sh

# Run training (with nohup to persist after disconnect)
nohup ./scripts/train/run_h100_multisignal.sh > training_h100.log 2>&1 &

# Monitor training
tail -f training_h100.log

# Check GPU usage
watch -n 1 nvidia-smi
```

#### Environment Optimizations for H100

The training script (`run_h100_multisignal.sh`) applies the following H100-specific optimizations:

- **VLLM_ATTENTION_BACKEND=FLASH_ATTN**: Uses Flash Attention backend for vLLM (faster than XFORMERS on H100)
- **NVIDIA_TF32_OVERRIDE=1**: Enables TF32 for faster matmul on Hopper architecture
- **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True**: Memory optimization for large batch training
- **CUDA_DEVICE_MAX_CONNECTIONS=1**: Optimizes CUDA connections for single GPU
- **Chunked Prefill**: Enabled in vLLM for efficient memory usage
- **Liger Kernel**: Enabled for fused operations in FSDP

#### Current Environment Status (as of last check)

- ✅ GPU: NVIDIA H100 PCIe detected and working
- ✅ CUDA: Available and functional (CUDA 12.8)
- ✅ PyTorch: 2.9.0+cu128 installed
- ✅ vLLM: 0.12.0 installed
- ⚠️ Flash Attention: Installed but may need rebuild for PyTorch 2.9.0 compatibility
- ✅ Project directory: Present at `~/One-Shot-RLVR`
- ✅ Sufficient disk space: 361 GB available
- ✅ Sufficient memory: 205 GiB available

---

## Run Summary (H100 Multi-Signal RLVR)

- Latest best checkpoint: `~/One-Shot-RLVR/checkpoints_backup/global_step_220`
- Run name: `Qwen2.5-Math-1.5B-pi1-multisignal-h100-fullft-20251205_220716`
- Validation (math500) peaked at ~0.604 at step 220.
- Reward weights: α=1.0, β=0.5, γ=0.5 (verify / format / reflect).
- Training steps: 220 (hard stop equals `total_training_steps`).
- Hardware: 1× NVIDIA H100 80GB, BF16, FlashAttention.

## Training Configuration (stable high-throughput)
- Batch: `data.train_batch_size=64`, `data.val_batch_size=500`
- Max lengths: prompt 1024, response 1536
- PPO: `ppo_max_token_len_per_gpu=20000`, `ppo_mini_batch_size=64`
- Rollout (vLLM): group `n=4`, `max_num_batched_tokens=16384`, `gpu_memory_utilization=0.65`, temperature 0.6, chunked prefill enabled
- FSDP: no offload, gradient checkpointing on, liger enabled
- Optim: lr=1e-6, kl_loss_coef=0.001, low_var_kl
- Logging: WandB + console
- Saving/val: save_freq=50, test_freq=50, val_before_train=False
- Resume: forced off (`resume_mode=off`, `resume_from_path=null`, guard in `ray_trainer.py`)

## Why These Hyperparameters (selection rationale)
- Steps 220: matches target wall-clock on single H100 (few hours) while giving clear reward lift; hard stop via `total_training_steps=EPOCHS` to prevent overruns.
- Batch 64: best throughput without triggering OOM under response 1536 and group 4.
- Response length 1536: 2048/3072 caused KV init OOM at higher rollout group; 1536 kept quality with stable memory.
- Rollout group n=4: n≥8 led to OOM on vLLM KV cache; n=4 balances speed and memory.
- `max_num_batched_tokens=16384` and `gpu_memory_utilization=0.65`: reduced from earlier aggressive (32768 / 0.88) after OOM during vLLM startup.
- LR 1e-6 + KL 0.001 (low_var_kl): conservative to avoid divergence on small dataset and keep reward stable.
- Temperature 0.6: slightly diversified samples while keeping format adherence for the multi-signal reward.
- FSDP no offload + gradient checkpointing + liger: best latency on H100; offload hurt speed without saving enough memory.
- Save/test every 50 steps: reduces I/O overhead yet provides periodic eval.
- Quick mode (3 steps, short length/group) kept for smoke tests only; not for quality.

## What We Changed (key fixes)
1) Added guard in `ray_trainer.py` to skip checkpoint loading when `resume_mode` is off/disable.
2) Script now generates unique run name per launch (`RUN_SUFFIX`) to avoid WandB name collisions.
3) Stabilized memory to avoid OOM: reduced rollout group/kv size/util and response length.
4) Hard stop via `total_training_steps=EPOCHS` to prevent overshooting.
5) Backups preserved: `checkpoints_backup/global_step_200`, `.../global_step_220`, `.../global_step_250`.

## Naive Reward Baseline (in progress)
- Run: `Qwen2.5-Math-1.5B-pi1-naive-baseline-20251205`
- Config: same stable params, but `reward_model.reward_manager=naive`, 180 steps target.
- Status (last check): running around step ~70, GPU ~33% util.

## How to Train (current stable script)
- Script: `scripts/train/run_h100_multisignal.sh`
- Defaults (full run): 220 steps, batch 64, response 1536, group 4, kv_tokens 16384, gpu_mem 0.65.
- Quick mode: 3 ultra-fast epochs, response 512, group 2 (for sanity only).
- Usage:
  ```bash
  # full run
  ./scripts/train/run_h100_multisignal.sh
  
  # quick sanity
  ./scripts/train/run_h100_multisignal.sh --quick
  ```

## Notes / Next Steps
- Evaluate `global_step_220` after training load frees GPU.
- Compare multi-signal vs naive baseline when the naive run finishes (math500).
- Optional ablations: tweak reward weights (e.g., β/γ) with short 150–200 step runs.


