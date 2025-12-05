# Training Scripts

## 🎯 H100 Training (Recommended)

### Main Script: `train/run_h100_multisignal.sh`

Fully optimized for NVIDIA H100 (80GB) with Full Parameter Fine-Tuning.

```bash
# Run verification first
./scripts/train/verify_h100_multisignal.sh

# Quick test (10 epochs, ~15 min)
./scripts/train/run_h100_multisignal.sh --quick

# Full training (2000 epochs, ~8-10 hours)
./scripts/train/run_h100_multisignal.sh

# Custom configuration via environment
EPOCHS=500 GROUP_SIZE=16 ./scripts/train/run_h100_multisignal.sh
```

### H100 Optimizations Applied

| Optimization | Setting | Benefit |
|--------------|---------|---------|
| Flash Attention | `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | 2-3x faster attention |
| Liger Kernel | `use_liger=True` | Fused operations |
| Chunked Prefill | `enable_chunked_prefill=True` | Better memory utilization |
| GPU Memory | `gpu_memory_utilization=0.88` | Maximum throughput |
| Batch Tokens | `max_num_batched_tokens=32768` | Large batch efficiency |
| No Offloading | `param_offload=False` | No CPU-GPU transfers |

### Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `EPOCHS` | 2000 | Total training epochs |
| `GROUP_SIZE` | 32 | GRPO group size (rollout.n) |
| `SAVE_FREQ` | 100 | Checkpoint save frequency |
| `TEST_FREQ` | 50 | Validation frequency |
| `REWARD_WEIGHTS` | "1.0,0.5,0.5" | α, β, γ for multi-signal |
| `WANDB_API_KEY` | (required) | WandB API key |

---

## 📱 L4 Training (Memory Constrained)

### `train/train_qwen2.5_1.5b_l4.sh`

For NVIDIA L4 (24GB) with LoRA for memory efficiency.

```bash
./scripts/train/train_qwen2.5_1.5b_l4.sh
```

---

## 📁 Script Summary

| Script | Hardware | Method | Purpose |
|--------|----------|--------|---------|
| `run_h100_multisignal.sh` | H100 80GB | Full FT | **Main training** |
| `verify_h100_multisignal.sh` | Any | - | Setup verification |
| `train_qwen2.5_1.5b_l4.sh` | L4 24GB | LoRA | Memory-constrained |
| `training_1.5b_pi1_r128.sh` | 8x GPU | Full FT | Reference (multi-GPU) |
| `training_1.5b_dsr_sub.sh` | 8x GPU | Full FT | DSR subset training |

---

## 🔧 Other Utility Scripts

### `monitor_training.sh`
```bash
./scripts/monitor_training.sh 60  # Refresh every 60s
```

### `create_quick_dataset.sh` / `create_medium_dataset.sh`
```bash
./scripts/create_quick_dataset.sh 512
./scripts/create_medium_dataset.sh 3712 train_medium
```

---

## 📊 Expected Performance (H100)

| Metric | Quick Mode | Full Training |
|--------|------------|---------------|
| Epochs | 10 | 2000 |
| Time | ~15 min | ~8-10 hours |
| Throughput | ~3 epochs/min | ~3 epochs/min |
| Peak VRAM | ~65 GB | ~70 GB |

---

## 🔑 Required Environment Variables

```bash
export WANDB_API_KEY="your_wandb_key"
export WANDB_PROJECT="verl_one_shot_rlvr"  # optional
```
