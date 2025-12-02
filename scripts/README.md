# Training Scripts

## Unified Training Script

### `scripts/remote_start_training.sh`

**Execute in GCP Console SSH terminal** (recommended)

```bash
# Ultra-quick validation (2-3 minutes, 1 step, 512 samples)
bash ~/One-Shot-RLVR/scripts/remote_start_training.sh ultra-quick

# Quick validation (22 minutes, ~15 steps)
bash ~/One-Shot-RLVR/scripts/remote_start_training.sh quick

# Normal validation (70 minutes, 58 steps)
bash ~/One-Shot-RLVR/scripts/remote_start_training.sh dry-run

# Fast training (3-4 hours, 145 steps, 50% data)
bash ~/One-Shot-RLVR/scripts/remote_start_training.sh fast

# Ultra-fast training (1-2 hours, 48 steps, 20% data)
bash ~/One-Shot-RLVR/scripts/remote_start_training.sh ultra-fast

# Full training (17 hours, 870 steps)
bash ~/One-Shot-RLVR/scripts/remote_start_training.sh full

# Custom experiment name
bash ~/One-Shot-RLVR/scripts/remote_start_training.sh fast my_experiment_v1
```

### Mode Comparison

| Mode | Steps | Time | Dataset | Purpose |
|------|-------|------|---------|---------|
| **ultra-quick** | 1 | 2-3 min | 512 | Ultra-fast workflow validation |
| **quick** | ~15 | 22 min | 7,424 | Quick effect validation |
| **dry-run** | 58 | 70 min | 7,424 | Full validation |
| **fast** | 145 | 3-4h | 3,712 (50%) | Fast training with good balance |
| **ultra-fast** | 48 | 1-2h | 1,536 (20%) | Fastest training, minimal data |
| **full** | 870 | 17h | 7,424 | Complete training with full dataset |

## Other Scripts

### `scripts/create_quick_dataset.sh`
Create small dataset for ultra-quick validation (auto-called in ultra-quick mode)

```bash
bash scripts/create_quick_dataset.sh 512
```

### `scripts/create_medium_dataset.sh`
Create medium-sized dataset for fast training (auto-called in fast/ultra-fast modes)

```bash
# Create 50% dataset (3712 samples) for fast mode
bash scripts/create_medium_dataset.sh 3712 train_medium

# Create 20% dataset (1536 samples) for ultra-fast mode
bash scripts/create_medium_dataset.sh 1536 train_ultra_fast
```

### `scripts/monitor_training.sh`
Monitor training progress (run locally)

```bash
bash scripts/monitor_training.sh 60  # Refresh every 60 seconds
```

## Configuration

- **Model**: Qwen2.5-0.5B
- **Rollout**: vLLM
- **Precision**: bfloat16 (full precision)
- **GPU**: 1 x NVIDIA L4 (24GB)
- **Logger**: Console + Wandb

## Monitoring

### View Logs
```bash
# Ultra-quick
tail -f ~/ultra_quick_log.txt

# Fast modes
tail -f ~/fast_log.txt          # fast mode
tail -f ~/ultra_fast_log.txt    # ultra-fast mode

# Other modes
tail -f ~/qwen0.5b_log.txt      # quick, dry-run, full
```

### Check Progress
```bash
grep -E "step:[0-9]+" ~/qwen0.5b_log.txt | tail -5
```

### Wandb
Visit: https://wandb.ai  
Project: `verl_grpo_example_gsm8k`
