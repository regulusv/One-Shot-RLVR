# Copyright 2024 One-Shot-RLVR Project
# Licensed under the Apache License, Version 2.0

"""
Multi-Signal Math Reward Function for One-Shot RLVR

Implements the reward function:
    R = α·R_verify + β·R_format + γ·R_reflect

Where:
    - R_verify: Outcome reward (answer correctness via latex2sympy)
    - R_format: Format reward (valid XML tags: <think>, \boxed{}, <reflection>)
    - R_reflect: Reflection reward (metacognition consistency)
"""

import re
import os
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed, is_equiv


@dataclass
class RewardWeights:
    """Configuration for multi-signal reward weights."""
    alpha: float = 1.0   # Outcome (verification) weight
    beta: float = 0.5    # Format compliance weight
    gamma: float = 0.5   # Reflection consistency weight
    
    @classmethod
    def from_env(cls) -> 'RewardWeights':
        """Load weights from REWARD_WEIGHTS environment variable."""
        weights = cls()
        reward_weights_str = os.getenv('REWARD_WEIGHTS')
        if reward_weights_str:
            try:
                parts = [float(w.strip()) for w in reward_weights_str.split(',')]
                if len(parts) == 3:
                    weights.alpha, weights.beta, weights.gamma = parts
                else:
                    print(f"Warning: REWARD_WEIGHTS expects 3 values, got {len(parts)}. Using defaults.")
            except ValueError as e:
                print(f"Warning: Could not parse REWARD_WEIGHTS '{reward_weights_str}': {e}. Using defaults.")
        return weights


class MultiSignalMathReward:
    """
    Multi-Signal Reward Calculator for Math RLVR.
    
    Combines three signals:
    1. Verification (outcome): Is the final answer correct?
    2. Format compliance: Does the output follow the required structure?
    3. Reflection consistency: Does self-assessment align with actual correctness?
    """
    
    # Keywords indicating correct self-assessment
    CORRECT_KEYWORDS = frozenset(["correct", "confident", "right", "accurate", "verified"])
    # Keywords indicating incorrect self-assessment
    INCORRECT_KEYWORDS = frozenset(["wrong", "mistake", "error", "incorrect", "failed", "uncertain"])
    
    def __init__(self, weights: Optional[RewardWeights] = None):
        """
        Initialize the reward calculator.
        
        Args:
            weights: RewardWeights configuration. If None, loads from environment.
        """
        self.weights = weights or RewardWeights.from_env()
    
    def compute_verification_reward(self, solution_str: str, ground_truth: str) -> float:
        """
        Compute outcome/verification reward using latex2sympy equivalence.
        
        Args:
            solution_str: Model's generated solution
            ground_truth: Expected answer
            
        Returns:
            1.0 if answer is correct, 0.0 otherwise
        """
        try:
            boxed_string = last_boxed_only_string(solution_str)
            if boxed_string is not None:
                answer = remove_boxed(boxed_string)
                if is_equiv(answer, ground_truth):
                    return 1.0
        except Exception as e:
            # Log but don't crash - return 0 reward for malformed answers
            print(f"[MultiSignalReward] Verification error: {e}")
        return 0.0
    
    def compute_format_reward(self, solution_str: str) -> float:
        """
        Compute format compliance reward.
        
        Checks for:
        - <think> tag (reasoning start)
        - \\boxed{} (final answer)
        - <reflection>...</reflection> tags (self-assessment)
        
        Args:
            solution_str: Model's generated solution
            
        Returns:
            1.0 if all format requirements met, 0.0 otherwise
        """
        has_think = "<think>" in solution_str
        has_boxed = "\\boxed{" in solution_str or "\\boxed " in solution_str
        has_reflection_open = "<reflection>" in solution_str
        has_reflection_close = "</reflection>" in solution_str
        
        if has_think and has_boxed and has_reflection_open and has_reflection_close:
            return 1.0
        return 0.0
    
    def compute_reflection_reward(self, solution_str: str, is_correct: bool) -> float:
        """
        Compute reflection consistency reward.
        
        Rewards:
        - +1.0 if correct answer AND reflection indicates confidence/correctness
        - +1.0 if incorrect answer AND reflection indicates error/uncertainty
        - 0.0 otherwise (inconsistent metacognition)
        
        Args:
            solution_str: Model's generated solution
            is_correct: Whether the answer was actually correct
            
        Returns:
            Reflection reward score
        """
        # Extract reflection content
        reflection_match = re.search(
            r'<reflection>(.*?)</reflection>', 
            solution_str, 
            re.DOTALL | re.IGNORECASE
        )
        
        if not reflection_match:
            return 0.0
        
        reflection_text = reflection_match.group(1).lower()
        
        # Check for self-assessment indicators
        indicates_correct = any(kw in reflection_text for kw in self.CORRECT_KEYWORDS)
        indicates_incorrect = any(kw in reflection_text for kw in self.INCORRECT_KEYWORDS)
        
        # Reward consistent metacognition
        if is_correct and indicates_correct and not indicates_incorrect:
            return 1.0
        elif not is_correct and indicates_incorrect and not indicates_correct:
            return 1.0
        
        return 0.0
    
    def compute_total_reward(
        self, 
        solution_str: str, 
        ground_truth: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the total multi-signal reward.
        
        Args:
            solution_str: Model's generated solution
            ground_truth: Expected answer
            
        Returns:
            Tuple of (total_reward, component_rewards_dict)
        """
        # Compute individual rewards
        r_verify = self.compute_verification_reward(solution_str, ground_truth)
        r_format = self.compute_format_reward(solution_str)
        r_reflect = self.compute_reflection_reward(solution_str, is_correct=(r_verify == 1.0))
        
        # Weighted sum
        total = (
            self.weights.alpha * r_verify +
            self.weights.beta * r_format +
            self.weights.gamma * r_reflect
        )
        
        components = {
            'r_verify': r_verify,
            'r_format': r_format,
            'r_reflect': r_reflect,
            'alpha': self.weights.alpha,
            'beta': self.weights.beta,
            'gamma': self.weights.gamma,
        }
        
        return total, components


# Module-level instance for convenience
_default_reward_calculator: Optional[MultiSignalMathReward] = None


def get_reward_calculator() -> MultiSignalMathReward:
    """Get or create the default reward calculator."""
    global _default_reward_calculator
    if _default_reward_calculator is None:
        _default_reward_calculator = MultiSignalMathReward()
    return _default_reward_calculator


def compute_score(
    data_source: Any, 
    solution_str: str, 
    ground_truth: str, 
    extra_info: Any = None,
    use_think: bool = False
) -> float:
    """
    Compute multi-signal reward score (API-compatible function).
    
    This is the main entry point for the reward manager.
    
    Args:
        data_source: Data source identifier (unused, for API compatibility)
        solution_str: The model's generated solution
        ground_truth: The expected answer
        extra_info: Additional info (unused, for API compatibility)
        use_think: Whether thinking tags are used (unused, for API compatibility)
    
    Returns:
        Total multi-signal reward score: R = α·R_verify + β·R_format + γ·R_reflect
        
    Example:
        >>> score = compute_score("one_shot_rlvr", solution, "42")
        >>> print(score)  # e.g., 2.0 if all signals are satisfied
    """
    calculator = get_reward_calculator()
    total_reward, _ = calculator.compute_total_reward(solution_str, ground_truth)
    return total_reward


def compute_score_with_breakdown(
    solution_str: str, 
    ground_truth: str
) -> Tuple[float, Dict[str, float]]:
    """
    Compute multi-signal reward with full breakdown of components.
    
    Useful for debugging and analysis.
    
    Args:
        solution_str: The model's generated solution
        ground_truth: The expected answer
        
    Returns:
        Tuple of (total_score, component_dict)
    """
    calculator = get_reward_calculator()
    return calculator.compute_total_reward(solution_str, ground_truth)
