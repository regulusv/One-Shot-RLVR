#!/bin/bash
set -e
cd ~/One-Shot-RLVR

echo "📦 Installing dependencies..."
# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install main package
pip install -e .
# Install dependencies from requirements_gcp.txt
pip install -r requirements_gcp.txt

echo "🧪 Running Unit Tests..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 tests/test_multi_signal_reward.py

echo "🔍 Checking Environment (4-bit)..."
python3 scripts/check_env_4bit.py

