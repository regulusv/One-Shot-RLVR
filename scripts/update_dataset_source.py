import pandas as pd
import os

# Define paths
input_path = 'data/train/one_shot_rlvr/pi1_r128.parquet'
output_path = 'data/train/one_shot_rlvr/pi1_r128_multisignal.parquet'

# Check if input exists
if not os.path.exists(input_path):
    print(f"Error: {input_path} not found.")
    exit(1)

# Load data
df = pd.read_parquet(input_path)
print(f"Loaded {len(df)} rows from {input_path}")
print(f"Original data_source: {df['data_source'].unique()}")

# Update data_source
df['data_source'] = 'one_shot_rlvr'
print(f"Updated data_source to: {df['data_source'].unique()}")

# Save to new file
df.to_parquet(output_path)
print(f"Saved modified dataset to {output_path}")

