#!/bin/bash
# Create a small dataset for quick validation
# Usage: bash scripts/create_quick_dataset.sh [num_samples]

NUM_SAMPLES="${1:-512}"  # Default: 512 samples (1 step with batch_size=512)

echo "=== Creating Quick Validation Dataset ==="
echo "Samples: $NUM_SAMPLES"
echo ""

python3 << PYEOF
import pandas as pd
import os

# Paths
train_file = os.path.expanduser('~/data/gsm8k/train.parquet')
quick_train_file = os.path.expanduser('~/data/gsm8k/train_quick.parquet')

val_file = os.path.expanduser('~/data/gsm8k/test.parquet')
quick_val_file = os.path.expanduser('~/data/gsm8k/test_quick.parquet')

# Read full dataset
print(f"Reading {train_file}...")
df_train = pd.read_parquet(train_file)
print(f"Original train size: {len(df_train)}")

# Take first N samples
df_train_quick = df_train.head($NUM_SAMPLES)
print(f"Quick train size: {len(df_train_quick)}")

# Save quick dataset
print(f"Saving to {quick_train_file}...")
df_train_quick.to_parquet(quick_train_file, index=False)
print("✅ Quick train dataset created")

# Also create quick validation set (smaller)
if os.path.exists(val_file):
    print(f"\nReading {val_file}...")
    df_val = pd.read_parquet(val_file)
    print(f"Original val size: {len(df_val)}")
    
    # Take first 128 samples for validation
    df_val_quick = df_val.head(128)
    print(f"Quick val size: {len(df_val_quick)}")
    
    print(f"Saving to {quick_val_file}...")
    df_val_quick.to_parquet(quick_val_file, index=False)
    print("✅ Quick val dataset created")

print(f"\n📊 Quick dataset ready!")
print(f"   Train: {len(df_train_quick)} samples")
if os.path.exists(val_file):
    print(f"   Val: {len(df_val_quick)} samples")
PYEOF

echo ""
echo "✅ Quick dataset created!"
echo "   Use: data.train_files=\$HOME/data/gsm8k/train_quick.parquet"
echo "   Use: data.val_files=\$HOME/data/gsm8k/test_quick.parquet"

