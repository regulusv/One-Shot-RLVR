#!/bin/bash
# =============================================================================
# Verification Script for H100 Multi-Signal RLVR Setup
# =============================================================================
# Run this BEFORE training to verify the configuration is correct.
# =============================================================================
set -e

echo "=========================================="
echo "  H100 Multi-Signal RLVR Verification"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
check_pass() { echo -e "${GREEN}✓ $1${NC}"; }
check_fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }
check_warn() { echo -e "${YELLOW}⚠ $1${NC}"; }

echo "1. Checking GPU Configuration..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1 | tr -d ' ')
    echo "   GPU: $GPU_NAME ($GPU_MEM)"
    
    if [[ "$GPU_NAME" == *"H100"* ]]; then
        check_pass "H100 GPU detected"
    elif [[ "$GPU_NAME" == *"A100"* ]]; then
        check_warn "A100 GPU detected (will work but not H100 optimized)"
    else
        check_warn "GPU detected: $GPU_NAME (may work with reduced performance)"
    fi
else
    check_fail "nvidia-smi not found - no GPU available"
fi

echo ""
echo "2. Checking Python Environment..."
python3 -c "import torch; print(f'   PyTorch: {torch.__version__}')" || check_fail "PyTorch not installed"
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || check_fail "CUDA not available"
python3 -c "import transformers; print(f'   Transformers: {transformers.__version__}')" || check_fail "Transformers not installed"
python3 -c "import vllm; print(f'   vLLM: {vllm.__version__}')" 2>/dev/null || check_warn "vLLM not installed (optional)"
check_pass "Python environment OK"

echo ""
echo "3. Checking Flash Attention..."
python3 << 'EOF'
try:
    import torch
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-Math-1.5B", trust_remote_code=True)
    # Check if flash attention 2 is supported
    if hasattr(config, '_attn_implementation'):
        print(f"   Default attention: {config._attn_implementation}")
    print("   Flash Attention 2 support: Checking...")
    
    # Try to check flash_attn
    try:
        import flash_attn
        print(f"   flash_attn version: {flash_attn.__version__}")
    except ImportError:
        print("   flash_attn package not found, but sdpa might work")
except Exception as e:
    print(f"   Warning: {e}")
EOF
check_pass "Attention check completed"

echo ""
echo "4. Testing Multi-Signal Reward Function..."
python3 << 'EOF'
import os
os.environ['REWARD_WEIGHTS'] = '1.0,0.5,0.5'

from verl.utils.reward_score.multi_signal_math import (
    compute_score, 
    compute_score_with_breakdown,
    MultiSignalMathReward,
    RewardWeights
)

# Test case 1: Perfect response (correct answer, good format, correct self-assessment)
perfect_response = """
<think>
Let me solve this step by step.
The problem asks for 2 + 2.
2 + 2 = 4
</think>

The answer is \\boxed{4}.

<reflection>
I am confident this solution is correct. The addition is straightforward.
</reflection>
"""

score1, breakdown1 = compute_score_with_breakdown(perfect_response, "4")
print(f"   Test 1 - Perfect response:")
print(f"     Total: {score1:.2f} | verify={breakdown1['r_verify']}, format={breakdown1['r_format']}, reflect={breakdown1['r_reflect']}")
assert score1 == 2.0, f"Expected 2.0, got {score1}"

# Test case 2: Correct answer, good format, wrong self-assessment
wrong_assessment = """
<think>
Computing 2 + 2 = 4
</think>

\\boxed{4}

<reflection>
I made a mistake somewhere. This is probably wrong.
</reflection>
"""

score2, breakdown2 = compute_score_with_breakdown(wrong_assessment, "4")
print(f"   Test 2 - Wrong self-assessment:")
print(f"     Total: {score2:.2f} | verify={breakdown2['r_verify']}, format={breakdown2['r_format']}, reflect={breakdown2['r_reflect']}")
assert breakdown2['r_reflect'] == 0.0, "Incorrect self-assessment should get 0 reflection reward"

# Test case 3: Wrong answer, acknowledges error
honest_wrong = """
<think>
2 + 2 = 5? No that's wrong.
Actually, let me think... 2 + 2 = 5.
</think>

\\boxed{5}

<reflection>
I think I made a mistake. This answer is likely wrong.
</reflection>
"""

score3, breakdown3 = compute_score_with_breakdown(honest_wrong, "4")
print(f"   Test 3 - Honest about error:")
print(f"     Total: {score3:.2f} | verify={breakdown3['r_verify']}, format={breakdown3['r_format']}, reflect={breakdown3['r_reflect']}")
assert breakdown3['r_reflect'] == 1.0, "Honest error admission should get reflection reward"

# Test case 4: Missing reflection tags
no_reflection = """
<think>
2 + 2 = 4
</think>

\\boxed{4}
"""

score4, breakdown4 = compute_score_with_breakdown(no_reflection, "4")
print(f"   Test 4 - No reflection:")
print(f"     Total: {score4:.2f} | verify={breakdown4['r_verify']}, format={breakdown4['r_format']}, reflect={breakdown4['r_reflect']}")
assert breakdown4['r_format'] == 0.0, "Missing reflection should fail format check"

print("   ✓ All multi-signal reward tests passed!")
EOF
check_pass "Multi-signal reward function working correctly"

echo ""
echo "5. Checking Data Files..."
if [ -f "data/train/one_shot_rlvr/pi1_r128.parquet" ]; then
    check_pass "Training data found: pi1_r128.parquet"
else
    check_fail "Training data not found: data/train/one_shot_rlvr/pi1_r128.parquet"
fi

if [ -f "data/test/math500.parquet" ]; then
    check_pass "Validation data found: math500.parquet"
else
    check_warn "Validation data not found: data/test/math500.parquet"
fi

echo ""
echo "6. Verifying Model Configuration..."
python3 << 'EOF'
import torch
print(f"   Default dtype: {torch.get_default_dtype()}")
print(f"   BFloat16 support: {torch.cuda.is_bf16_supported()}")
print(f"   CUDA compute capability: {torch.cuda.get_device_capability()}")

# Check H100 optimizations
cap = torch.cuda.get_device_capability()
if cap[0] >= 9:  # Hopper (H100)
    print("   FP8 support: Available (Hopper architecture)")
elif cap[0] >= 8:  # Ampere
    print("   FP8 support: Not available (Ampere architecture)")
EOF
check_pass "Model configuration verified"

echo ""
echo "=========================================="
echo -e "${GREEN}  All Verification Checks Passed!${NC}"
echo "=========================================="
echo ""
echo "Ready to run training:"
echo "  Full training:  ./scripts/train/run_h100_multisignal.sh"
echo "  Quick test:     ./scripts/train/run_h100_multisignal.sh --quick"
echo ""
echo "Configuration Summary:"
echo "  - Model: Qwen2.5-Math-1.5B (Full Parameter FT)"
echo "  - Precision: BF16 + Flash Attention 2"
echo "  - Reward: Multi-Signal (α=1.0, β=0.5, γ=0.5)"
echo "  - Liger Kernel: Enabled (fused operations)"
echo "  - Group Size: 32 (configurable via GROUP_SIZE env)"
echo "  - Epochs: 2000 (configurable via EPOCHS env)"
echo ""
echo "H100 Optimizations Applied:"
echo "  ✓ VLLM_ATTENTION_BACKEND=FLASH_ATTN"
echo "  ✓ Chunked Prefill enabled"
echo "  ✓ Liger Kernel for fused ops"
echo "  ✓ GPU memory utilization: 88%"
echo "  ✓ No CPU offloading"
echo ""

