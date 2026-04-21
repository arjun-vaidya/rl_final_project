import unittest
from src.rewards.outcome import extract_answer_from_trajectory, outcome_reward

class TestRewards(unittest.TestCase):
    def test_extract_answer_from_trajectory(self):
        self.assertEqual(extract_answer_from_trajectory("<answer>42</answer>"), 42)
        self.assertEqual(extract_answer_from_trajectory("The result is <answer> -7 </answer>."), -7)
        self.assertEqual(extract_answer_from_trajectory("blah blah #### 100"), 100)
        self.assertEqual(extract_answer_from_trajectory("I think the answer is 50"), 50)
        self.assertIsNone(extract_answer_from_trajectory("No answer here"))

    def test_outcome_reward(self):
        self.assertEqual(outcome_reward("<answer>42</answer>", 42), 1.0)
        self.assertEqual(outcome_reward("<answer>42</answer>", 43), 0.0)
        self.assertEqual(outcome_reward("Maybe 10 or 20 #### 30", 30), 1.0)

if __name__ == "__main__":
    unittest.main()
