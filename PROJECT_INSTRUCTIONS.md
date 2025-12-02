
# Project: Multi-Signal One-Shot RLVR (Qwen2.5-Math-1.5B)
# Hardware: Single NVIDIA L4 GPU (24GB VRAM, supports BF16)

## 🎯 Objective
We are modifying the `One-Shot-RLVR` repository to implement a **Multi-Signal Reward** system (Outcome + Format + Reflection).
The goal is to reproduce "One-Shot RLVR" on a single example (`pi1`) but with enhanced reward signals.

## 🛠️ Hardware Context (CRITICAL)
* **GPU:** NVIDIA L4 (24GB VRAM).
* **Precision:** Use `bfloat16` (BF16) strictly. Do NOT use `float16`.
* **Quantization:** Try to run **without quantization** first (BF16 Full) or with LoRA. If OOM, fallback to 4-bit (NF4).

## 📝 Task List

### Task 1: Verify & Implement Multi-Signal Reward
**Context:** Basic math reward logic already exists in `verl/utils/reward_score/math.py` or `verl/utils/reward_score/prime_math/`.
**Action:**
1.  **Analyze** the existing reward files first. Do not blindly create a new file if we can subclass or extend `math.py`.
2.  **Implement** the `MultiSignalMathReward` class (either in a new file `multi_signal_math.py` or extending the existing one).
3.  **Logic Requirement:**
    * Compute: $R = \alpha \cdot R_{verify} + \beta \cdot R_{format} + \gamma \cdot R_{reflect}$
    * **$R_{verify}$:** Reuse existing `math_score` / `latex2sympy` logic.
    * **$R_{format}$:** Check for tags: `<think>`, `\boxed{}`, `<reflection>`.
    * **$R_{reflect}$:** Parse content inside `<reflection>...</reflection>`.
        * Reward (+1) if `verify=True` AND reflection says "correct".
        * Reward (+1) if `verify=False` AND reflection says "wrong".
        * Else 0.

### Task 2: Configure Model Loading for L4 (BF16)
**Action:** Modify `verl/utils/model.py` (or relevant actor class).
* **Force BF16:** Ensure `torch_dtype=torch.bfloat16`.
* **Attn Implementation:** Ensure `attn_implementation="flash_attention_2"` is enabled (L4 supports it).

### Task 3: Create Training Script (500 Steps)
**Action:** Create `scripts/train/run_l4_multisignal.sh`.
**Reference:** Base strictly on `scripts/train/training_1.5b_pi1_r128.sh` (Paper implementation).

**Key Parameter Changes:**
1.  **Steps/Epochs:** Set `trainer.total_epochs=500`. (Since dataset size = batch size = 128, 1 epoch = 1 step. We want **500 steps** for a quick 1-hour experiment).
2.  **Compute:**
    * `trainer.n_gpus_per_node=1`
    * `actor_rollout_ref.rollout.n=8` (Group Size = 8). (L4 24GB can handle larger groups than T4).
    * `actor_rollout_ref.model.enable_gradient_checkpointing=True`.
3.  **Data:** Use `data/train/one_shot_rlvr/pi1_r128.parquet`.
4.  **Reward:** Configure to use the new Multi-Signal reward.
5.  **LoRA:** Enable LoRA to save VRAM for larger rollouts.

### Task 4: Prompt Engineering
**Action:** Check `verl/utils/dataset/rl_dataset.py` or the script config.
* Update the system prompt to explicitly ask the model to output `<reflection>` tags after the answer.

## 🚀 Execution Order
1.  Check existing reward code -> Implement Task 1.
2.  Create the run script (Task 3) ensuring `total_epochs=500`.
3.  Modify model loader (Task 2).