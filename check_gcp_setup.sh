#!/bin/bash
# GCP Setup Verification Script
# This script checks if all prerequisites are met before running gcp_launch_instance.sh

set -e

# Add gcloud to PATH if not already there
export PATH=/Users/gil/google-cloud-sdk/bin:$PATH
export CLOUDSDK_PYTHON=/opt/homebrew/bin/python3.11

echo "🔍 Checking GCP setup prerequisites..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if gcloud is installed
echo -n "1. Checking gcloud CLI installation... "
if command -v gcloud &> /dev/null; then
    echo -e "${GREEN}✓ Installed${NC}"
    gcloud --version | head -n 1
else
    echo -e "${RED}✗ Not installed${NC}"
    echo "   Please install gcloud: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
echo ""

# Check if user is authenticated
echo -n "2. Checking authentication... "
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${GREEN}✓ Authenticated${NC}"
    echo "   Account: $(gcloud auth list --filter=status:ACTIVE --format='value(account)')"
else
    echo -e "${RED}✗ Not authenticated${NC}"
    echo "   Please run: gcloud auth login"
    exit 1
fi
echo ""

# Check if project is set
echo -n "3. Checking project configuration... "
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
EXPECTED_PROJECT="one-shot-rlvr-cs229"
if [ -n "$PROJECT_ID" ] && [ "$PROJECT_ID" != "(unset)" ]; then
    if [ "$PROJECT_ID" = "$EXPECTED_PROJECT" ]; then
        echo -e "${GREEN}✓ Project set correctly${NC}"
        echo "   Project ID: $PROJECT_ID"
    else
        echo -e "${YELLOW}⚠ Project set to: $PROJECT_ID${NC}"
        echo "   Expected: $EXPECTED_PROJECT"
        echo "   Run: gcloud config set project $EXPECTED_PROJECT"
    fi
else
    echo -e "${RED}✗ No project set${NC}"
    echo "   Please run: gcloud config set project $EXPECTED_PROJECT"
    exit 1
fi
echo ""

# Check if Compute Engine API is enabled
echo -n "4. Checking Compute Engine API... "
if gcloud services list --enabled --filter="name:compute.googleapis.com" --format="value(name)" | grep -q compute; then
    echo -e "${GREEN}✓ Enabled${NC}"
else
    echo -e "${YELLOW}⚠ Not enabled${NC}"
    echo "   Enabling Compute Engine API..."
    gcloud services enable compute.googleapis.com
    echo -e "${GREEN}✓ Enabled${NC}"
fi
echo ""

# Check billing account
echo -n "5. Checking billing account... "
BILLING_ACCOUNT=$(gcloud billing projects describe $PROJECT_ID --format="value(billingAccountName)" 2>/dev/null)
if [ -n "$BILLING_ACCOUNT" ] && [ "$BILLING_ACCOUNT" != "" ]; then
    echo -e "${GREEN}✓ Billing enabled${NC}"
    echo "   Billing account: $BILLING_ACCOUNT"
else
    echo -e "${RED}✗ No billing account${NC}"
    echo "   Please set up billing: https://console.cloud.google.com/billing"
    echo "   This is required even for free tier usage"
fi
echo ""

# Check GPU quota (informational)
echo "6. Checking GPU quota (informational)..."
echo "   Note: A100 GPU quota may need to be requested separately"
QUOTA_INFO=$(gcloud compute project-info describe --project=$PROJECT_ID --format="yaml(quotas)" 2>/dev/null | grep -i "nvidia.*a100" || echo "")
if [ -n "$QUOTA_INFO" ]; then
    echo -e "   ${GREEN}GPU quota information found${NC}"
    echo "   $QUOTA_INFO"
else
    echo -e "   ${YELLOW}⚠ Could not find A100 GPU quota info${NC}"
    echo "   You may need to request GPU quota: https://console.cloud.google.com/iam-admin/quotas"
    echo "   Search for 'NVIDIA A100' and request at least 8 GPUs"
fi
echo ""

# Check if script is configured
echo -n "7. Checking gcp_launch_instance.sh configuration... "
if [ -f "gcp_launch_instance.sh" ]; then
    if grep -q "PROJECT_ID=\"$EXPECTED_PROJECT\"" gcp_launch_instance.sh; then
        echo -e "${GREEN}✓ Script configured correctly${NC}"
        echo "   Project ID: $EXPECTED_PROJECT"
    elif grep -q 'PROJECT_ID="your-project-id"' gcp_launch_instance.sh; then
        echo -e "${YELLOW}⚠ Not configured${NC}"
        echo "   Please edit gcp_launch_instance.sh and set PROJECT_ID to: $EXPECTED_PROJECT"
    else
        echo -e "${YELLOW}⚠ Script found but project ID may be different${NC}"
    fi
else
    echo -e "${RED}✗ Script not found${NC}"
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Setup Summary:"
echo ""
echo "✓ Basic setup looks good!"
echo ""
echo "⚠️  Important reminders:"
echo "   1. Make sure GPU quota is approved (may take 1-2 days)"
echo "   2. Update PROJECT_ID in gcp_launch_instance.sh"
echo "   3. Be aware of costs: A100 instances are expensive!"
echo "   4. Consider using Preemptible instances to save ~70%"
echo ""
echo "📚 For detailed setup instructions, see: GCP_SETUP_GUIDE.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


