# Lambda Labs Setup Guide

## Instance Information
- **Type**: gpu_1x_h100_pcie
- **IP**: 209.20.157.194
- **Region**: us-west-3
- **SSH Key**: lambda-ssh-1

## 1. Connect to Instance
Use the following command to SSH into your instance (assuming your key is at `~/.ssh/lambda-ssh-1`):
```bash
ssh -i ~/.ssh/lambda-ssh-1 ubuntu@209.20.157.194
```

## 2. Environment Setup (On the Instance)
Lambda Labs instances usually come with pre-installed drivers and generic tools, but you'll need to set up the project environment.

```bash
# Clone the repository (or rsync from your local machine)
git clone https://github.com/your-username/One-Shot-RLVR.git
cd One-Shot-RLVR

# Install dependencies (using the pre-installed python environment or create a new one)
# Lambda often has a good default environment, but a venv is safer
sudo apt-get update && sudo apt-get install -y python3-venv

python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements_train.txt
pip install flash-attn --no-build-isolation
```

## 3. Data Transfer
You need to upload your data and scripts. From your LOCAL machine:

```bash
# Upload the project code and data
rsync -avz -e "ssh -i ~/.ssh/lambda-ssh-1" \
    --exclude '.git' \
    --exclude 'checkpoints' \
    --exclude 'outputs' \
    --exclude 'wandb' \
    /Users/gilbert/Documents/One-Shot-RLVR/ \
    ubuntu@209.20.157.194:~/One-Shot-RLVR/
```

## 4. Run Training
On the remote instance:

```bash
cd ~/One-Shot-RLVR
source venv/bin/activate

# Make script executable
chmod +x scripts/train/run_lambda_h100.sh

# Run with nohup to keep running after disconnect
nohup ./scripts/train/run_lambda_h100.sh > training_h100.log 2>&1 &

# Monitor
tail -f training_h100.log
```

## 5. Monitoring
- **WandB**: Check your project `verl_few_shot` on WandB.
- **GPU Usage**: Watch `nvidia-smi` on the instance.

