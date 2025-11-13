# GCP Quick Start

## ✅ Completed

- ✅ Google Cloud SDK installed (`/Users/gil/google-cloud-sdk`)
- ✅ Project ID configured: `one-shot-rlvr-cs229`
- ✅ Scripts ready

## 🚀 Next 3 Steps

### 1️⃣ Authentication and Configuration

```bash
./setup_gcp.sh
```

This will open a browser for you to sign in and automatically configure the project.

⚠️ **Note**: If you see "Billing account not found" error, you need to complete step 2 first.

### 2️⃣ Set Up Billing Account ⚠️ **Must Complete First!**

**Must be completed before enabling APIs!** Otherwise you'll get an error.

1. Visit: https://console.cloud.google.com/billing
2. If you don't have a billing account:
   - Click "Create Billing Account"
   - Fill in information (new users get $300 free credit, valid for 12 months)
3. Link project:
   - On the billing page, click "Link Project"
   - Select project `one-shot-rlvr-cs229`
   - Or visit: https://console.cloud.google.com/billing/linkedaccount?project=one-shot-rlvr-cs229

After completion, run `./setup_gcp.sh` again to enable APIs.

### 3️⃣ Request GPU Quota ⚠️

**Required step!** Cannot create instances without GPU quota.

**First decide how many GPUs you need:**
- Script default: **1 GPU** (suitable for testing and small-scale training)
- For more, edit `gcp_launch_instance.sh`, change `GPU_COUNT=1` to 2, 4, or 8

**Cost Reference (Preemptible):**
- 1 GPU: ~$3-4/hour
- 2 GPU: ~$6-8/hour  
- 4 GPU: ~$12-16/hour
- 8 GPU: ~$24-36/hour

**Request Steps:**
1. Visit: https://console.cloud.google.com/iam-admin/quotas?project=one-shot-rlvr-cs229
2. Search: `NVIDIA A100`
3. Request the number of GPUs you need (recommended to start with 1-2)
4. Wait for approval (1-2 business days)

## ✅ Verify Setup

```bash
./check_gcp_setup.sh
```

## 🚀 Launch Instance

```bash
./gcp_launch_instance.sh
```

## 💰 Cost Reminder

**Default Configuration (1 GPU, Preemptible)**: ~$3-4/hour

**Other Configurations (Preemptible)**:
- 2 GPU: ~$6-8/hour
- 4 GPU: ~$12-16/hour
- 8 GPU: ~$24-36/hour

**Standard Instances (Non-Preemptible)**: About 3x the price

**Remember to delete when done!**

```bash
# Delete instance
gcloud compute instances delete one-shot-rlvr-train --zone=us-central1-a
```

## 📝 Common Commands

```bash
gcloud config list                    # View configuration
gcloud auth list                      # View authenticated accounts
gcloud compute ssh one-shot-rlvr-train --zone=us-central1-a  # SSH connection
```

---

📚 Detailed Guide: [GCP_SETUP_GUIDE.md](GCP_SETUP_GUIDE.md)
