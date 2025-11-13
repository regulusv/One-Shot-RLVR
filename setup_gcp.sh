#!/bin/bash
# GCP Setup Script - Automated setup for GCP configuration

set -e

# Add gcloud to PATH
export PATH=/Users/gil/google-cloud-sdk/bin:$PATH
export CLOUDSDK_PYTHON=/opt/homebrew/bin/python3.11

PROJECT_ID="one-shot-rlvr-cs229"

echo "🔧 Setting up GCP configuration..."
echo ""

# Check if already authenticated
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "✅ Already authenticated"
    gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1
else
    echo "📝 Please authenticate with Google Cloud..."
    echo "This will open a browser window for you to sign in."
    echo ""
    gcloud auth login --no-launch-browser
    echo ""
    echo "✅ Authentication complete!"
fi

echo ""
echo "🔧 Setting project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

echo ""
echo "🔧 Setting default compute zone..."
gcloud config set compute/zone us-central1-a

echo ""
echo "🔧 Checking billing account..."
BILLING_ACCOUNT=$(gcloud billing projects describe $PROJECT_ID --format="value(billingAccountName)" 2>/dev/null || echo "")

if [ -z "$BILLING_ACCOUNT" ] || [ "$BILLING_ACCOUNT" = "" ]; then
    echo "❌ ERROR: Billing account not found!"
    echo ""
    echo "⚠️  You must set up billing before enabling APIs."
    echo ""
    echo "Please:"
    echo "  1. Visit: https://console.cloud.google.com/billing"
    echo "  2. Create a billing account (if needed)"
    echo "  3. Link project '$PROJECT_ID' to the billing account"
    echo "     Or visit: https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
    echo ""
    echo "After setting up billing, run this script again."
    exit 1
else
    echo "✅ Billing account found: $BILLING_ACCOUNT"
fi

echo ""
echo "🔧 Enabling Compute Engine API..."
if gcloud services enable compute.googleapis.com 2>&1 | grep -q "FAILED_PRECONDITION"; then
    echo "❌ Failed to enable API. Please check billing account setup."
    exit 1
fi

echo ""
echo "✅ GCP setup complete!"
echo ""
echo "Current configuration:"
gcloud config list
echo ""
echo "⚠️  Important: You still need to:"
echo "   1. Set up billing account (if not already done)"
echo "   2. Request GPU quota (A100 GPUs require quota approval)"
echo "      Visit: https://console.cloud.google.com/iam-admin/quotas"
echo "      Search for 'NVIDIA A100' and request at least 8 GPUs"
echo ""

