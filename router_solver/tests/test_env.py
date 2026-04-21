"""Tests for src/env/*."""


def test_gsm8k_loader_shapes():
    """Train excludes val; val is 200; test is 1319."""
    raise NotImplementedError


def test_ground_truth_extraction():
    """'...#### 42' -> 42."""
    raise NotImplementedError


def test_python_tool_basic():
    """run_python('4*2').output == '8'."""
    raise NotImplementedError


def test_python_tool_timeout():
    """Infinite loop terminates within timeout."""
    raise NotImplementedError


def test_python_tool_error():
    """1/0 returns is_error=True with ZeroDivisionError in output."""
    raise NotImplementedError
