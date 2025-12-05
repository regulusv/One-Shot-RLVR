# Role: Senior AI Engineer & RL Researcher
# Project: Multi-Signal One-Shot RLVR (Qwen2.5-Math-1.5B)
# Hardware: Single NVIDIA H100 GPU (80GB VRAM)

## 🎯 Objective
We are reproducing and improving the "One-Shot RLVR" paper using **Full Parameter Fine-Tuning** on a high-end H100 GPU.
Our goal is to implement a **Multi-Signal Reward** system (Outcome + Format + Reflection) and train `Qwen2.5-Math-1.5B` on the single `pi1` example.

## 🛠️ Hardware Context (UNCONSTRAINED)
* **GPU:** NVIDIA H100 (80GB VRAM).
* **Compute Capability:** Hopper Arch (FP8/BF16 optimized).
* **Strategy:** Maximize throughput and performance. **No LoRA. No Quantization.**

## 📝 Task List

### Task 1: Verify & Implement Multi-Signal Reward
**Context:** Check `verl/utils/reward_score/math.py`.
**Action:**
1.  Implement `MultiSignalMathReward` class in `verl/utils/reward_score/multi_signal_math.py`.
2.  **Logic Requirement:**
    * $R = \alpha \cdot R_{verify} + \beta \cdot R_{format} + \gamma \cdot R_{reflect}$
    * **$R_{verify}$:** Use `latex2sympy` logic (Outcome).
    * **$R_{format}$:** Valid XML tags (`<think>`, `\boxed{}`, `<reflection>`).
    * **$R_{reflect}$:**
        * Reward (+1) if `verify=True` AND reflection says "correct".
        * Reward (+1) if `verify=False` AND reflection says "wrong".
        * Else 0.

### Task 2: Configure Model Loading (Full Precision)
**Action:** Modify `verl/utils/model.py` (or `verl/workers/actor/megatron_actor.py` if needed).
* **Precision:** Force `torch_dtype=torch.bfloat16`.
* **No Quantization:** Ensure `load_in_4bit` is **False**.
* **Optimization:** Enable `attn_implementation="flash_attention_2"`.

### Task 3: Create H100 Training Script (Full Power)
**Action:** Create `scripts/train/run_h100_multisignal.sh`.
**Reference:** Base on `scripts/train/training_1.5b_pi1_r128.sh`.

**Key Configuration Changes (For H100):**
1.  **Compute:**
    * `trainer.n_gpus_per_node=1`
    * `actor_rollout_ref.rollout.n=32` (Set Group Size to **32**). *Reason: H100 has huge memory; larger group size reduces GRPO variance significantly.*
    * `actor_rollout_ref.model.enable_gradient_checkpointing=True` (Keep enabled to allow massive rollouts).
2.  **Training Strategy (Full FT):**
    * **Disable LoRA:** Remove any PEFT/LoRA flags. We will fine-tune all 1.5B parameters.
    * `actor_rollout_ref.actor.fsdp_config.param_offload=False` (Keep on GPU).
    * `actor_rollout_ref.actor.fsdp_config.optimizer_offload=False` (Keep on GPU).
3.  **Steps/Epochs:**
    * `trainer.total_epochs=2000`. *Reason: H100 is fast enough to run the full paper schedule.*
4.  **Batch Size:**
    * `data.train_batch_size=128`.
    * `actor_rollout_ref.actor.ppo_mini_batch_size=64` (Try larger mini-batch).
5.  **Data:** `data/train/one_shot_rlvr/pi1_r128.parquet`.
6.  **Reward:** Use the new Multi-Signal reward manager.

### Task 4: Prompt Engineering
**Action:** Check `verl/utils/dataset/rl_dataset.py`.
* Ensure the system prompt enforces the `<think> ... \boxed{} ... <reflection>` format.

## 🚀 Execution Order
1.  Code the Reward Function (Task 1).
2.  Create the `run_h100_multisignal.sh` script (Task 3).
3.  Double-check Model Loading (Task 2) ensures BF16 and no bitsandbytes.