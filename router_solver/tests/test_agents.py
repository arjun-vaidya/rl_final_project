"""Tests for src/agents/*.

These use a MockLM that returns canned strings so we can test parsing/flow without GPU.
"""


def test_flat_agent_parses_answer_tag():
    """Flat agent extracts N from <answer>N</answer>."""
    raise NotImplementedError


def test_flat_agent_interleaves_tool_calls():
    """Each <code>...</code> in generation triggers one tool call."""
    raise NotImplementedError


def test_router_solver_plan_parsing():
    """Valid JSON plan becomes list[Subgoal]; invalid becomes empty list."""
    raise NotImplementedError


def test_router_solver_step_sequence():
    """With a 3-subgoal plan, Solver runs 3 times and scratchpad grows."""
    raise NotImplementedError


def test_router_solver_respects_max_subgoals():
    """Plans with >max_subgoals are truncated or rejected."""
    raise NotImplementedError


def test_router_solver_memory_disabled_is_identical():
    """Agent with memory=None produces identical trajectories to agent with no memory kwarg."""
    raise NotImplementedError
