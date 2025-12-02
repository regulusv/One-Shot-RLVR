#!/bin/bash
# Create medium-sized dataset for faster training
# Usage: bash scripts/create_medium_dataset.sh [num_samples] [output_name]
# Examples:
#   bash scripts/create_medium_dataset.sh 3712 train_medium  # 50% data
#   bash scripts/create_medium_dataset.sh 2227 train_medium  # 30% data

NUM_SAMPLES="${1:-3712}"  # Default: 3712 samples (50% of 7424)
OUTPUT_NAME="${2:-train_medium}"

echo "=== Creating Medium Training Dataset ==="
echo "Samples: $NUM_SAMPLES"
echo "Output: ${OUTPUT_NAME}.parquet"
echo ""

python3 << PYEOF
import pandas as pd
import os
import numpy as np

# Paths
train_file = os.path.expanduser('~/data/gsm8k/train.parquet')
output_train_file = os.path.expanduser(f'~/data/gsm8k/${OUTPUT_NAME}.parquet')

val_file = os.path.expanduser('~/data/gsm8k/test.parquet')
output_val_file = os.path.expanduser(f'~/data/gsm8k/test_${OUTPUT_NAME}.parquet')

# Read full dataset
print(f"Reading {train_file}...")
df_train = pd.read_parquet(train_file)
print(f"Original train size: {len(df_train)}")

# Randomly sample N samples (for better diversity)
if $NUM_SAMPLES < len(df_train):
    df_train_medium = df_train.sample(n=$NUM_SAMPLES, random_state=42).reset_index(drop=True)
else:
    df_train_medium = df_train.copy()
    
print(f"Medium train size: {len(df_train_medium)}")

# Save medium dataset
print(f"Saving to {output_train_file}...")
df_train_medium.to_parquet(output_train_file, index=False)
print("✅ Medium train dataset created")

# Also create medium validation set (proportional)
if os.path.exists(val_file):
    print(f"\nReading {val_file}...")
    df_val = pd.read_parquet(val_file)
    print(f"Original val size: {len(df_val)}")
    
    # Take proportional amount (or max 512)
    val_samples = min(512, int(len(df_val) * ($NUM_SAMPLES / len(df_train))))
    df_val_medium = df_val.sample(n=val_samples, random_state=42).reset_index(drop=True)
    print(f"Medium val size: {len(df_val_medium)}")
    
    print(f"Saving to {output_val_file}...")
    df_val_medium.to_parquet(output_val_file, index=False)
    print("✅ Medium val dataset created")

print(f"\n📊 Medium dataset ready!")
print(f"   Train: {len(df_train_medium)} samples → {output_train_file}")
if os.path.exists(val_file):
    print(f"   Val: {len(df_val_medium)} samples → {output_val_file}")
PYEOF

echo ""
echo "✅ Medium dataset created!"
echo "   Use: data.train_files=\$HOME/data/gsm8k/${OUTPUT_NAME}.parquet"
echo "   Use: data.val_files=\$HOME/data/gsm8k/test_${OUTPUT_NAME}.parquet"

