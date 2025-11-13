# GCP Setup Guide

Before using the `gcp_launch_instance.sh` script, you need to complete the following preparations.

## 📋 Prerequisites Checklist

### 1. Create Google Cloud Platform (GCP) Account

1. **Register GCP Account**
   - Visit https://cloud.google.com/
   - Click "Get started for free" or "Try free"
   - Sign in with your Google account
   - Enter credit card information (new users get $300 free credit, valid for 12 months)

2. **Create Project**
   - Visit https://console.cloud.google.com/
   - Click the project selector at the top, select "New Project"
   - Enter project name (e.g., "one-shot-rlvr")
   - Record the project ID (format like `my-project-123456`)
   - Click "Create"
   - Project: one-shot-rlvr-cs229

### 2. Install Google Cloud SDK (gcloud CLI)

**macOS Installation:**

```bash
# Method 1: Using Homebrew (recommended)
brew install --cask google-cloud-sdk

# Method 2: Manual installation
# Download installer: https://cloud.google.com/sdk/docs/install
```

**Verify Installation:**
```bash
gcloud --version
```

### 3. Initialize gcloud and Authenticate

```bash
# Initialize gcloud
gcloud init

# This will guide you through:
# 1. Sign in to Google account
# 2. Select project
# 3. Set default region (recommended: us-central1-a)
```

**Or manually set up authentication:**
```bash
# Sign in
gcloud auth login

# Set default project
gcloud config set project YOUR_PROJECT_ID

# Set default zone
gcloud config set compute/zone us-central1-a
```

### 4. Enable Required APIs

The script requires the following APIs to be enabled:

```bash
# Enable Compute Engine API
gcloud services enable compute.googleapis.com

# Verify API is enabled
gcloud services list --enabled
```

### 5. Request GPU Quota (Important!)

⚠️ **A100 GPUs require special quota, default accounts don't have it!**

```bash
# Check current quota
gcloud compute project-info describe --project=YOUR_PROJECT_ID

# Request quota increase (must be done via Web Console)
# 1. Visit: https://console.cloud.google.com/iam-admin/quotas
# 2. Search for "NVIDIA A100" or "GPU"
# 3. Select the corresponding quota (e.g., "NVIDIA A100 GPUs")
# 4. Click "EDIT QUOTAS"
# 5. Request at least 8 A100 GPUs (script needs 8x A100)
# 6. Fill in request reason (e.g., "For deep learning training")
# 7. Submit request (usually takes 1-2 business days for approval)
```

**Notes:**
- New accounts typically don't have GPU quota
- Need to request quota increase from Google
- Request may require usage reason and business information
- For testing, you can start by requesting 1-2 GPU quota

### 6. Set Up Billing Account

Ensure the project is linked to a billing account:
- Visit https://console.cloud.google.com/billing
- Ensure the project is linked to a valid billing account
- Check that the billing account status is normal

### 7. Configure Script

✅ **Completed** - Project ID `one-shot-rlvr-cs229` has been configured in `gcp_launch_instance.sh`

### 8. Verify Prerequisites

Run the following commands to verify:

```bash
# Check authentication status
gcloud auth list

# Check current project
gcloud config get-value project

# Check if API is enabled
gcloud services list --enabled | grep compute

# Check quota (need to wait for quota approval)
gcloud compute project-info describe --project=YOUR_PROJECT_ID | grep -i quota
```

## 🚀 Run Script

After completing the above steps, run:

```bash
chmod +x gcp_launch_instance.sh
./gcp_launch_instance.sh
```

## ⚠️ Important Notes

1. **Cost Warning**
   - A100 GPU instances are very expensive (~$10-15/hour)
   - 8x A100 instances may cost $80-120 per hour
   - Using Preemptible instances can save about 70% cost
   - Remember to delete instances promptly after use!

2. **Quota Limitations**
   - New accounts don't have GPU quota by default
   - Must request quota to create GPU instances
   - Quota requests may take 1-2 business days

3. **Region Limitations**
   - Not all regions have A100 GPUs
   - Common available regions: us-central1, us-east1, europe-west4
   - Script defaults to us-central1-a

4. **Preemptible Instances**
   - Cheaper but may be terminated at any time
   - Not suitable for long training tasks
   - Suitable for testing and short-term tasks

## 🔧 Troubleshooting

**Issue: Insufficient Quota**
```bash
# Check quota
gcloud compute project-info describe --project=YOUR_PROJECT_ID
# Need to request quota increase (see step 5)
```

**Issue: Authentication Failed**
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

**Issue: API Not Enabled**
```bash
# Enable Compute Engine API
gcloud services enable compute.googleapis.com
```

**Issue: GPU Not Found**
```bash
# Check if region has GPUs
gcloud compute accelerator-types list --filter="zone:us-central1-a"
```

## 📚 Reference Resources

- GCP Official Documentation: https://cloud.google.com/docs
- gcloud CLI Documentation: https://cloud.google.com/sdk/docs
- GPU Quota Request: https://console.cloud.google.com/iam-admin/quotas
- Pricing Calculator: https://cloud.google.com/products/calculator
