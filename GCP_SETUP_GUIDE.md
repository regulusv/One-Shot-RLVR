# GCP Setup & Usage Guide

This guide covers the complete process of setting up Google Cloud Platform (GCP) for the One-Shot RLVR project, from account creation to launching training instances.

## ✅ Current Status & Prerequisites
**The following have been pre-configured in this repository:**
- Project ID: `one-shot-rlvr-cs229`
- Scripts: `setup_gcp.sh`, `gcp_launch_instance.sh`, `check_gcp_setup.sh`

## 🚀 Quick Start Steps

### 1. Account & Billing (Required First)
Before running any scripts, you must have a GCP account with billing enabled.
1.  **GCP Console**: Visit [Google Cloud Console](https://console.cloud.google.com/).
2.  **Billing**: Go to [Billing](https://console.cloud.google.com/billing) and create/link a billing account (new users get $300 credit).
    *   **Link Project**: Ensure project `one-shot-rlvr-cs229` is linked to your billing account.
    *   Direct link: [Link Billing for one-shot-rlvr-cs229](https://console.cloud.google.com/billing/linkedaccount?project=one-shot-rlvr-cs229)

### 2. Install Google Cloud SDK
If you haven't installed the `gcloud` CLI:
*   **macOS**: `brew install --cask google-cloud-sdk`
*   **Manual**: [Install Guide](https://cloud.google.com/sdk/docs/install)
*   Verify: `gcloud --version`

### 3. Authentication & Auto-Configuration
Run the setup script to authenticate and enable required APIs:

```bash
./setup_gcp.sh
```

This will:
*   Open a browser to sign in.
*   Configure the project ID.
*   Enable Compute Engine APIs.

### 4. Request GPU Quota (CRITICAL)
**You cannot create instances without a GPU quota.**
1.  **Check Quota**: [IAM & Admin > Quotas](https://console.cloud.google.com/iam-admin/quotas?project=one-shot-rlvr-cs229)
2.  **Search**: `NVIDIA A100` (or the GPU type you intend to use).
3.  **Request Increase**:
    *   Region: `us-central1` (or your preferred region).
    *   Quantity: At least **1** (Script defaults to 1, max 8 for full training).
    *   Approval time: Usually 1-2 business days.

## 🛠️ Launching Instances

### Verify Setup
Before launching, run the check script to ensure environment is ready:
```bash
./check_gcp_setup.sh
```

### Launch Training Instance
To create and configure the VM:
```bash
./gcp_launch_instance.sh
```
*   **Default**: 1 GPU, Preemptible instance (cheaper, short-lived).
*   **Modify**: Edit `GPU_COUNT` in the script to change (e.g., 2, 4, 8).

### Manage Instance
*   **SSH Connection**:
    ```bash
    gcloud compute ssh one-shot-rlvr-train --zone=us-central1-a
    ```
*   **Delete Instance** (IMPORTANT: Do this when done to avoid charges):
    ```bash
    gcloud compute instances delete one-shot-rlvr-train --zone=us-central1-a
    ```

## 💰 Costs & Instance Info

| Configuration (Preemptible) | Est. Cost | Notes |
|-----------------------------|-----------|-------|
| **1 GPU** (Default) | ~$3-4/hr | Good for testing/debugging |
| **8 GPUs** | ~$24-36/hr | Full scale training |
| **Standard Instance** | ~3x Cost | Use if preemptible keeps killing job |

*   **Preemptible**: Instances can be terminated by Google at any time (usually after 24h max). Great for cost savings.
*   **Region**: Script defaults to `us-central1-a`. Change in `gcp_launch_instance.sh` if needed.

## 🔧 Troubleshooting

*   **"Billing account not found"**: Complete Step 1 (Link Billing).
*   **"Quota exceeded"**: Complete Step 4 (Request Quota) or reduce GPU count.
*   **"Zone resource not available"**: Try a different zone in `gcp_launch_instance.sh` (e.g., `us-central1-b`, `us-central1-c`).

## 📚 Common Commands
```bash
gcloud config list                    # View config
gcloud auth list                      # Check active account
gcloud services list --enabled        # Check enabled APIs
gcloud compute accelerator-types list # Check available GPUs in zone
```
