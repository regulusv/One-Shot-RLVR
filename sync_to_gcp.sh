#!/bin/bash
# Sync local code to GCP instance

# Add gcloud to PATH if not already there (based on local environment)
export PATH=/Users/gil/google-cloud-sdk/bin:$PATH

# Configuration
INSTANCE_NAME="cs229v1-20251201-231519"
ZONE="us-east1-d"
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
    requirements_gcp.txt \
    setup.py \
    PROJECT_INSTRUCTIONS.md \
    $INSTANCE_NAME:$REMOTE_DIR

echo "✅ Sync complete!"
echo "To connect: gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE"
