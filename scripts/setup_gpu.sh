#!/bin/bash
set -e

echo "🔍 Checking GPU status..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA Driver already installed!"
    nvidia-smi
    exit 0
fi

echo "⚠️  NVIDIA Driver not found. Attempting installation..."

# Update and install build dependencies
echo "📦 Updating package lists..."
sudo apt-get update
sudo apt-get install -y build-essential linux-headers-$(uname -r) software-properties-common

# Install drivers (using standard repository)
echo "📦 Installing NVIDIA Drivers (535)..."
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

echo "✅ Installation complete."
echo "🔄 You typically need to REBOOT for drivers to load."
echo "   Run: sudo reboot"

