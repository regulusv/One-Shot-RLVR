# One-Shot-RLVR: Multi-Signal Reinforcement Learning

This repository extends the original One-Shot-RLVR codebase with support for single-GPU training and multi-signal reward functions for mathematical reasoning.

## Overview

**Goal**: Train `Qwen2.5-Math-1.5B` (or smaller models) using **GRPO** (Group Relative Policy Optimization) with a **multi-signal reward function** on a single GPU.

**Key Features**:
- Multi-signal reward combining verification, format, and reflection signals
- Single GPU training support (tested on NVIDIA L4 24GB)
- Optimized configurations for different validation/training scenarios
- GCP cloud training infrastructure

## Quick Start

### Training on GCP

1. **SSH to GCP instance**:
   ```bash
   gcloud compute ssh instance-20251202-055916 \
       --project=one-shot-rlvr-cs229 \
       --zone=northamerica-northeast1-b
   ```

2. **Start training**:
   ```bash
   # Ultra-quick validation (2-3 minutes)
   bash ~/One-Shot-RLVR/scripts/remote_start_training.sh ultra-quick
   
   # Quick validation (22 minutes)
   bash ~/One-Shot-RLVR/scripts/remote_start_training.sh quick
   
   # Full training (17 hours)
   bash ~/One-Shot-RLVR/scripts/remote_start_training.sh full
   ```

See `scripts/README.md` for detailed usage.

## Key Modifications

### 1. Multi-Signal Reward Function
- **File**: `verl/utils/reward_score/multi_signal_math.py`
- **Purpose**: Composite reward combining:
  - **Verification** ($r_{verify}$): Correctness of mathematical answer
  - **Format** ($r_{format}$): Presence of required XML tags
  - **Reflection** ($r_{reflect}$): Quality of self-reflection
- **Formula**: $r = \alpha \cdot r_{verify} + \beta \cdot r_{format} + \gamma \cdot r_{reflect}$
- **Integration**: Modified `verl/trainer/main_ppo.py` to support `multi_signal` reward manager

### 2. Single GPU Training Support
- **Model**: Qwen2.5-0.5B (494M parameters)
- **Script**: `examples/grpo_trainer/run_qwen0.5b_l4_vllm.sh`
- **Configuration**: Optimized for L4 GPU (24GB) with vLLM rollout
- **Memory**: ~16GB usage with full precision (bfloat16)
- **Why 0.5B?**: 1.5B model requires ~22GB, exceeding L4 capacity

### 3. Training Modes

| Mode | Steps | Time | Dataset | Purpose |
|------|-------|------|---------|---------|
| **ultra-quick** | 1 | 2-3 min | 512 | Workflow validation |
| **quick** | ~15 | 22 min | 7,424 | Effect validation |
| **dry-run** | 58 | 70 min | 7,424 | Full validation |
| **full** | 870 | 17h | 7,424 | Final training |

### 4. GCP Infrastructure
- **Scripts**: `scripts/remote_start_training.sh`, `scripts/create_quick_dataset.sh`
- **Setup**: `scripts/setup_gpu.sh`, `scripts/setup_l4_gpu.sh`
- **Monitoring**: `scripts/monitor_training.sh`

## Technical Notes

### vLLM Compatibility
- **vLLM does NOT support 4-bit quantization** - requires full precision (fp16/bf16)
- Attempts to use 4-bit + vLLM result in weight synchronization failures
- **Solution**: Use full precision with smaller models (0.5B) or more GPUs

### Memory Constraints
- **L4 GPU (24GB)**: Can train 0.5B models, not 1.5B
- **1.5B model**: Requires 2× L4 or 1× A100 (40GB+)
- **0.5B model**: Fits comfortably (~16GB usage)

### Training Configuration
- **Rollout**: vLLM (faster than HF rollout)
- **Precision**: bfloat16 (full precision)
- **Batch size**: 128 (normal), 512 (quick validation)
- **Steps per epoch**: 58 (with batch_size=128)

## Project Structure

```
One-Shot-RLVR/
├── verl/
│   ├── utils/
│   │   └── reward_score/
│   │       └── multi_signal_math.py    # Multi-signal reward
│   ├── trainer/
│   │   └── main_ppo.py                 # Modified for multi_signal
│   └── workers/
│       └── fsdp_workers.py              # 4-bit + LoRA support
├── examples/
│   └── grpo_trainer/
│       ├── run_qwen0.5b_l4_vllm.sh      # Main training script
│       ├── run_qwen0.5b_l4_vllm_quick.sh # Quick validation
│       └── run_qwen0.5b_l4_vllm_ultra_quick.sh # Ultra-quick
├── scripts/
│   ├── remote_start_training.sh         # Unified startup script
│   ├── create_quick_dataset.sh         # Create small dataset
│   └── monitor_training.sh             # Monitor progress
└── docs/
    └── TRAINING_CONCEPTS.md             # Training concepts explained
```

## Documentation

- **Training Scripts**: `scripts/README.md`
- **Training Concepts**: `docs/TRAINING_CONCEPTS.md`
- **GCP Setup**: `GCP_SETUP_GUIDE.md`
- **Project Instructions**: `PROJECT_INSTRUCTIONS.md`

## Environment Variables

- `REWARD_WEIGHTS`: Comma-separated weights for multi-signal reward (default: "1.0,0.5,0.5")
- `WANDB_API_KEY`: Weights & Biases API key for logging
- `VLLM_ATTENTION_BACKEND`: Set to `FLASH_ATTN` for L4 GPUs

## Monitoring

### Wandb
- **URL**: https://wandb.ai
- **Project**: `verl_grpo_example_gsm8k`

### Logs
- Ultra-quick: `~/ultra_quick_log.txt`
- Other modes: `~/qwen0.5b_log.txt`

## Known Issues & Solutions

### Issue: OOM with 1.5B model
**Solution**: Use 0.5B model or upgrade to larger GPU

### Issue: vLLM + 4-bit incompatibility
**Solution**: Use full precision (bfloat16) with vLLM

### Issue: Slow training
**Solution**: Use `quick` or `ultra-quick` modes for validation

## License

See `LICENSE` file for details.
