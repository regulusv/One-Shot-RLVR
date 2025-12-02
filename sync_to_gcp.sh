#!/bin/bash
# Sync local code to GCP instance

# Add gcloud to PATH if not already there (based on local environment)
export PATH=/Users/gil/google-cloud-sdk/bin:$PATH

# Configuration - L4 GPU Instance
INSTANCE_NAME="instance-20251202-055916"
ZONE="northamerica-northeast1-b"
PROJECT_ID="one-shot-rlvr-cs229"

# Remote directory
REMOTE_DIR="~/One-Shot-RLVR"

echo "🔄 Syncing code to $INSTANCE_NAME ($ZONE)..."

# Ensure remote directory exists
gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE --command="mkdir -p $REMOTE_DIR"

# Sync files using gcloud compute scp with recursion
# We explicitly list directories to avoid syncing heavy/unnecessary folders
gcloud compute scp --project=$PROJECT_ID --zone=$ZONE --recurse \
    verl \
    examples \
    data \
    tests \
    scripts \
    requirements.txt \
    setup.py \
    pyproject.toml \
    README.md \
    $INSTANCE_NAME:$REMOTE_DIR

echo "✅ Sync complete!"
echo "To connect: gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE"
