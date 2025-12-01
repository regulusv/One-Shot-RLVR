# Project Instructions: Multi-Signal RL for One-Shot LLM (T4 GPU)

## 1. Project Context & Objective
We are modifying the `One-Shot-RLVR` repository to implement a **Multi-Signal Reinforcement Learning** approach for mathematical reasoning.
**Goal:** Train `Qwen2.5-Math-1.5B` on a **single training example** using **GRPO** (Group Relative Policy Optimization) on a single **NVIDIA T4 GPU**.

**Key Constraints:**
* **Hardware:** Single T4 GPU (16GB VRAM).
* **Quantization:** MUST use **4-bit quantization (NF4)** + **LoRA** to fit in memory.
* **Algorithm:** GRPO (as suggested by reviewers to utilize group-based normalization).

---

## 2. Implementation Tasks for Cursor/AI

### Task 1: Implement Multi-Signal Reward Function
**Objective:** Create a composite reward function $r = \alpha \cdot r_{verify} + \beta \cdot r_{format} + \gamma \cdot r_{reflect}$.

* **Action:** Create a new file `verl/utils/reward_score/multi_signal_math.py`.
* **Reference:** Use `verl/utils/reward_score/math.py` as a template.
* **Requirements:**
    1.  **$r_{verify}$ (Outcome):** Use the existing logic (latex2sympy or string match) to verify the content inside `\boxed{}`.
    2.  **$r_{format}$ (Format):** Return 1.0 if the output contains valid XML tags: `<think>`, `\boxed{}`, and `<reflection>`. Else 0.0.
    3.  **$r_{reflect}$ (Reflection - NEW):**
        * Parse the text inside `<reflection>...</reflection>`.
        * **Logic:**
            * If **Correct Answer** AND Reflection contains "correct/confident" $\rightarrow$ +1.0.
            * If **Incorrect Answer** AND Reflection contains "wrong/mistake" $\rightarrow$ +1.0.
            * Otherwise $\rightarrow$ 0.0.
    4.  **Weighted Sum:** The `compute_score` function should return the weighted sum. Allow weights ($\alpha, \beta, \gamma$) to be passed or hardcoded for now (e.g., 1.0, 0.5, 0.5).

### Task 2: Enable 4-bit Quantization (Crucial for T4)
**Objective:** Modify the model loading logic to support `bitsandbytes` NF4 quantization.

* **Target File:** Likely `verl/utils/model.py` or `verl/workers/actor/megatron_actor.py` (depending on the launcher). Since we are likely using the HuggingFace single-GPU setup for T4, check `verl/utils/model.py`.
* **Action:**
    * Import `BitsAndBytesConfig` from `transformers`.
    * When initializing `AutoModelForCausalLM.from_pretrained`:
        * Add `quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)`.
        * Ensure `device_map="auto"` is set appropriately for the single GPU.
    * **Note:** Ensure LoRA (PEFT) is enabled in the config YAML, as full fine-tuning 1.5B on T4 is impossible.

### Task 3: Configure GRPO Training Script
**Objective:** Create a launch script based on the reviewer's suggestion to use GRPO with $n > 1$ sampling.

* **Reference:** `examples/grpo_trainer/run_qwen2-7b.sh`.
* **Action:** Create a new script `examples/grpo_trainer/run_qwen1.5b_t4_multisignal.sh`.
* **Modifications:**
    * **Model:** Change to `Qwen/Qwen2.5-Math-1.5B`.
    * **Rollout Config:**
        * Set `trainer.n_rollout` (Group Size) to **4** or **8** (Higher is better for GRPO variance reduction, but limited by T4 memory. Start with 4).
    * **Reward Function:** Point to the new `multi_signal_math` function created in Task 1.
    * **Hyperparameters:**
        * `learning_rate`: `1e-6` (Conservative for One-Shot).
        * `micro_batch_size`: `1` (To save VRAM).
        * `gradient_accumulation`: Increase if needed.
    * **LoRA:** Ensure the script flags enable LoRA (`use_lora=True`).

### Task 4: Prompt Engineering for Reflection
**Objective:** Ensure the model knows it needs to generate a reflection.

* **Target File:** `verl/utils/dataset/rl_dataset.py` or the data preprocessing script `data/data_selection.py`.
* **Action:** Modify the prompt template appended to the single training example.
* **New System/User Prompt:**
    > "Solve the math problem. Show your reasoning in <think> tags. Output the final answer in \boxed{}. Finally, analyze your own solution in <reflection> tags, stating whether you believe it is 'correct' or 'wrong'."

---

## 3. Verification Plan
After implementing the changes, run the following verification:

1.  **Environment Check:** Run a dummy python script to load `Qwen2.5-Math-1.5B` with 4-bit quantization and print memory usage (should be < 5GB VRAM).
2.  **Reward Check:** Write a unit test for `multi_signal_math.py` with mock outputs (correct+confident, wrong+confident, etc.) to verify scoring logic.
3.  **Training Dry-Run:** Run the new shell script for 10 steps. Check `wandb` or logs to ensure:
    * `r_verify`, `r_format`, `r_reflect` are logged.
    * No OOM (Out Of Memory) errors.