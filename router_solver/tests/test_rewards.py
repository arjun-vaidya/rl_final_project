"""Tests for src/rewards/*."""


def test_outcome_reward_match():
    """Exact match → 1.0; mismatch → 0.0."""
    raise NotImplementedError


def test_outcome_reward_no_answer():
    """Trajectory without extractable answer → 0.0."""
    raise NotImplementedError


def test_router_reward_invalid_json():
    """Malformed JSON plan → 0.0 regardless of outcome."""
    raise NotImplementedError


def test_router_reward_too_many_subgoals():
    """Plan with > max_subgoals → 0.0."""
    raise NotImplementedError


def test_router_reward_valid_and_correct():
    """Valid plan + correct answer → 1.0."""
    raise NotImplementedError


def test_solver_step_reward_all_three_components():
    """tool-valid + sensible + outcome=1 → 1.0 with default weights."""
    raise NotImplementedError


def test_solver_step_reward_error_short_circuits():
    """is_error=True → 0.0, regardless of other signals."""
    raise NotImplementedError
