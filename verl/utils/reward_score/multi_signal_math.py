import re
import os
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed, is_equiv

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """
    Computes the multi-signal reward score.
    
    Weights:
    Controlled by REWARD_WEIGHTS env var (comma-separated).
    Default:
    alpha (verify) = 1.0
    beta (format) = 0.5
    gamma (reflect) = 0.5
    """
    
    # Weights
    alpha = 1.0
    beta = 0.5
    gamma = 0.5
    
    reward_weights = os.getenv('REWARD_WEIGHTS')
    if reward_weights:
        try:
            weights = [float(w) for w in reward_weights.split(',')]
            if len(weights) == 3:
                alpha, beta, gamma = weights
            else:
                print(f"Warning: REWARD_WEIGHTS must contain 3 values, got {len(weights)}. Using defaults.")
        except ValueError:
            print(f"Warning: Could not parse REWARD_WEIGHTS '{reward_weights}'. Using defaults.")
    
    # 1. r_verify (Outcome)
    r_verify = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                r_verify = 1.0
    except Exception as e:
        print(f"Error in r_verify: {e}")
        
    # 2. r_format (Format)
    # Return 1.0 if the output contains valid XML tags: <think>, \boxed{}, and <reflection>. Else 0.0.
    r_format = 0.0
    if "<think>" in solution_str and "\\boxed{" in solution_str and "<reflection>" in solution_str and "</reflection>" in solution_str:
         r_format = 1.0
         
    # 3. r_reflect (Reflection)
    r_reflect = 0.0
    reflection_match = re.search(r'<reflection>(.*?)</reflection>', solution_str, re.DOTALL)
    
    if reflection_match:
        reflection_text = reflection_match.group(1).lower()
        
        # Logic:
        # If Correct Answer AND Reflection contains "correct/confident" -> +1.0
        # If Incorrect Answer AND Reflection contains "wrong/mistake" -> +1.0
        # Otherwise -> 0.0
        
        is_correct = (r_verify == 1.0)
        
        if is_correct and any(keyword in reflection_text for keyword in ["correct", "confident"]):
            r_reflect = 1.0
        elif not is_correct and any(keyword in reflection_text for keyword in ["wrong", "mistake"]):
            r_reflect = 1.0
            
    # Weighted Sum
    total_score = (alpha * r_verify) + (beta * r_format) + (gamma * r_reflect)
    
    return total_score

