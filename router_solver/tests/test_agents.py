import unittest
from src.utils.parsing import parse_plan_json, extract_code_block
from src.rewards.router import router_reward
from src.rewards.solver import solver_step_reward
from src.env.python_tool import ToolResult

class TestPeter(unittest.TestCase):
    def test_parse_plan_json(self):
        valid_log = 'Plan: {"plan": [{"subgoal": "add 2+2", "tool": "python"}]}'
        plan = parse_plan_json(valid_log)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["plan"][0]["subgoal"], "add 2+2")

        invalid_log = 'Plan: { broken json }'
        self.assertIsNone(parse_plan_json(invalid_log))

    def test_extract_code_block(self):
        text = "I should use code: <code>4 * 2</code>... result: 8"
        self.assertEqual(extract_code_block(text), "4 * 2")
        self.assertIsNone(extract_code_block("No code here"))

    def test_router_reward_basic(self):
        # Valid plan, correct trajectory
        plan_out = '{"plan": [{"s": "1", "t": "p"}]}'
        traj = "The answer is <answer>4</answer>"
        # Using a mock outcome_reward check
        res = router_reward(plan_out, traj, 4)
        self.assertEqual(res, 1.0)

        # Invalid JSON
        self.assertEqual(router_reward("{ bad }", traj, 4), 0.0)

        # Wrong plan length (0 or >8)
        self.assertEqual(router_reward('{"plan": []}', traj, 4), 0.0)

    def test_solver_reward_basic(self):
        # Clean execution, Correct final outcome
        res = solver_step_reward(ToolResult("8", False, 10.0), 1.0)
        self.assertEqual(res, 1.0) # 0.3 + 0.2 + 0.5

        # Error
        res = solver_step_reward(ToolResult("err", True, 10.0), 1.0)
        self.assertEqual(res, 0.0)

        # Clean execution, Wrong final outcome
        res = solver_step_reward(ToolResult("8", False, 10.0), 0.0)
        self.assertEqual(res, 0.5) # 0.3 + 0.2 + 0.0

if __name__ == "__main__":
    unittest.main()
