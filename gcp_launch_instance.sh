#!/bin/bash
# GCP Launch Script - One-Shot RLVR Training Instance

set -e

# Add gcloud to PATH if not already there
export PATH=/Users/gil/google-cloud-sdk/bin:$PATH
export CLOUDSDK_PYTHON=/opt/homebrew/bin/python3.11

# Configuration variables
PROJECT_ID="one-shot-rlvr-cs229"
INSTANCE_NAME="cs229v1-20251201-231519"
ZONE="us-east1-d"

# GPU Configuration - Adjust based on your needs
# GPU_TYPE options:
#   - "T4"     : NVIDIA T4 (16GB) - Easiest quota to obtain, suitable for testing and small-scale training
#   - "L4"     : NVIDIA L4 (24GB) - Quota relatively easy to obtain, better performance
#   - "A100"   : NVIDIA A100 (40GB) - Requires quota approval, best performance
# GPU_COUNT: 1, 2, 4, or 8 (T4/L4 typically use 1-4, A100 can use 1-8)

GPU_TYPE="T4"    # Change to "T4", "L4", or "A100"
GPU_COUNT=1      # Change to 1, 2, 4, or 8

# Set GPU accelerator type and machine type based on GPU_TYPE
case $GPU_TYPE in
    "T4")
        ACCELERATOR_TYPE="nvidia-tesla-t4"
        # T4 uses n1 machine types
        case $GPU_COUNT in
            1) MACHINE_TYPE="n1-standard-4" ;;   # 4 vCPU, 15GB RAM
            2) MACHINE_TYPE="n1-standard-8" ;;   # 8 vCPU, 30GB RAM
            4) MACHINE_TYPE="n1-standard-16" ;;  # 16 vCPU, 60GB RAM
            8) MACHINE_TYPE="n1-standard-32" ;;   # 32 vCPU, 120GB RAM
            *) echo "❌ Error: T4 GPU_COUNT must be 1, 2, 4, or 8"; exit 1 ;;
        esac
        ;;
    "L4")
        ACCELERATOR_TYPE="nvidia-l4"
        # L4 uses n1 machine types
        case $GPU_COUNT in
            1) MACHINE_TYPE="n1-standard-4" ;;
            2) MACHINE_TYPE="n1-standard-8" ;;
            4) MACHINE_TYPE="n1-standard-16" ;;
            *) echo "❌ Error: L4 GPU_COUNT must be 1, 2, or 4"; exit 1 ;;
        esac
        ;;
    "A100")
        ACCELERATOR_TYPE="nvidia-tesla-a100"
        # A100 uses a2 machine types
        case $GPU_COUNT in
            1) MACHINE_TYPE="a2-highgpu-1g" ;;
            2) MACHINE_TYPE="a2-highgpu-2g" ;;
            4) MACHINE_TYPE="a2-highgpu-4g" ;;
            8) MACHINE_TYPE="a2-highgpu-8g" ;;
            *) echo "❌ Error: A100 GPU_COUNT must be 1, 2, 4, or 8"; exit 1 ;;
        esac
        ;;
    *)
        echo "❌ Error: GPU_TYPE must be 'T4', 'L4', or 'A100'"
        exit 1
        ;;
esac

IMAGE_FAMILY="common-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="500GB"
BOOT_DISK_TYPE="pd-balanced"

# Optional: Use Preemptible instances to save costs (70% cheaper)
USE_PREEMPTIBLE=true  # Change to false to use standard instances

echo "🚀 Creating GCP instance..."
echo "Project: $PROJECT_ID"
echo "Instance name: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Machine type: $MACHINE_TYPE"
echo "GPU: $GPU_COUNT x $GPU_TYPE"

# Build create command
CREATE_CMD="gcloud compute instances create $INSTANCE_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --accelerator=type=$ACCELERATOR_TYPE,count=$GPU_COUNT \
  --maintenance-policy=TERMINATE \
  --image-family=$IMAGE_FAMILY \
  --image-project=$IMAGE_PROJECT \
  --boot-disk-size=$BOOT_DISK_SIZE \
  --boot-disk-type=$BOOT_DISK_TYPE \
  --boot-disk-device-name=$INSTANCE_NAME \
  --metadata=install-nvidia-driver=True \
  --scopes=https://www.googleapis.com/auth/cloud-platform"

# If using Preemptible
if [ "$USE_PREEMPTIBLE" = true ]; then
  CREATE_CMD="$CREATE_CMD --preemptible"
  echo "💰 Using Preemptible instance (about 70% cheaper)"
fi

# Execute creation
eval $CREATE_CMD

echo ""
echo "✅ Instance created successfully!"
echo ""
echo "📝 Connection method:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "Or view in GCP Console:"
echo "https://console.cloud.google.com/compute/instances?project=$PROJECT_ID"
