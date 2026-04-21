import unittest
from src.env.gsm8k_loader import extract_numeric_answer, load_gsm8k_train
from src.env.python_tool import run_python

class TestEnv(unittest.TestCase):
    def test_extract_numeric_answer(self):
        self.assertEqual(extract_numeric_answer("The answer is #### 42"), 42)
        self.assertEqual(extract_numeric_answer("#### -7"), -7)
        self.assertIsNone(extract_numeric_answer("No answer here"))

    def test_gsm8k_loader(self):
        # Only load a few to verify it works
        problems = load_gsm8k_train()
        self.assertGreater(len(problems), 0)
        self.assertIsInstance(problems[0].numeric_answer, int)
        self.assertIn(str(problems[0].numeric_answer), problems[0].answer)

    def test_python_tool_simple(self):
        res = run_python("4 * 2")
        self.assertEqual(res.output, "8")
        self.assertFalse(res.is_error)

    def test_python_tool_complex(self):
        res = run_python("import math; math.sqrt(16)")
        # Depending on how eval/exec works, math.sqrt(16) might return 4.0
        self.assertIn(res.output, ["4.0", "4"])
        self.assertFalse(res.is_error)

    def test_python_tool_error(self):
        res = run_python("1/0")
        self.assertTrue(res.is_error)
        self.assertIn("ZeroDivisionError", res.output)

    def test_python_tool_timeout(self):
        res = run_python("while True: pass", timeout=0.1)
        self.assertTrue(res.is_error)
        self.assertEqual(res.output, "TimeoutExpired")

if __name__ == "__main__":
    unittest.main()
