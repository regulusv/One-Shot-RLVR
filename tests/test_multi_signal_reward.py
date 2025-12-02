import unittest
from verl.utils.reward_score.multi_signal_math import compute_score

class TestMultiSignalReward(unittest.TestCase):
    
    def test_correct_confident(self):
        # Correct answer + confident reflection
        solution = r"<think>...</think>\boxed{5}<reflection>I am confident this is correct.</reflection>"
        ground_truth = "5"
        score = compute_score(solution, ground_truth)
        # 1.0 (verify) + 1.0 (format) + 1.0 (reflect) = 1.0 + 0.5 + 0.5 = 2.0
        self.assertEqual(score, 2.0, "Score should be 2.0 for correct and confident answer")

    def test_correct_unconfident(self):
         # Correct answer + no confident keywords
        solution = r"<think>...</think>\boxed{5}<reflection>I think this might be okay.</reflection>"
        ground_truth = "5"
        score = compute_score(solution, ground_truth)
        # 1.0 (verify) + 1.0 (format) + 0.0 (reflect) = 1.0 + 0.5 = 1.5
        self.assertEqual(score, 1.5, "Score should be 1.5 for correct but unconfident answer")

    def test_wrong_mistake_acknowledged(self):
        # Incorrect answer + acknowledged mistake
        solution = r"<think>...</think>\boxed{4}<reflection>I made a mistake calculation.</reflection>"
        ground_truth = "5"
        score = compute_score(solution, ground_truth)
        # 0.0 (verify) + 1.0 (format) + 1.0 (reflect) = 0.5 + 0.5 = 1.0
        self.assertEqual(score, 1.0, "Score should be 1.0 for incorrect but acknowledged mistake")
    
    def test_wrong_confident(self):
        # Incorrect answer + confident (delusion)
        solution = r"<think>...</think>\boxed{4}<reflection>I am confident this is correct.</reflection>"
        ground_truth = "5"
        score = compute_score(solution, ground_truth)
        # 0.0 (verify) + 1.0 (format) + 0.0 (reflect) = 0.5
        self.assertEqual(score, 0.5, "Score should be 0.5 for incorrect and confident answer")

    def test_bad_format(self):
        # Missing tags
        solution = r"The answer is \boxed{5}"
        ground_truth = "5"
        score = compute_score(solution, ground_truth)
        # 1.0 (verify) + 0.0 (format) + 0.0 (reflect) = 1.0
        self.assertEqual(score, 1.0, "Score should be 1.0 for correct answer with bad format")

if __name__ == '__main__':
    unittest.main()

