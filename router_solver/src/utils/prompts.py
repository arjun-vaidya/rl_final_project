# src/utils/prompts.py
import json
from typing import List, Sequence, Union

ROUTER_HEADER = "You are a planner. Output a JSON plan of subgoals for this math problem."
ROUTER_FORMAT_LINE = 'Format: {"plan": [{"subgoal": "<desc>", "tool": "python"}, ...]}'

SOLVER_PROMPT_TEMPLATE = """You are solving one subgoal of a larger problem.

Original problem:  {question}
Full plan:         {plan_json}
Previous results:  {scratchpad}
Current subgoal:   {current_subgoal}

Write Python code (one expression or short block) wrapped in <code>...</code>.
"""


def _render_memory_block(entries: Sequence) -> str:
    """Render retrieved (question, plan) pairs per docs/04_design.md and 07_plan_memory.md."""
    if not entries:
        return ""
    lines = ["Similar problems you have solved before:"]
    for i, entry in enumerate(entries, start=1):
        # Accept either a dict {"question": ..., "plan": ...} or a (question, plan) tuple.
        if isinstance(entry, dict):
            past_q = entry.get("question", "")
            past_plan = entry.get("plan", entry)
        else:
            past_q, past_plan = entry
        if not isinstance(past_plan, str):
            past_plan = json.dumps(past_plan, ensure_ascii=False)
        lines.append(f"{i}. Problem: {past_q}")
        lines.append(f"   Plan:    {past_plan}")
    return "\n".join(lines) + "\n\n"


def build_router_prompt(question: str, past_memory: Union[str, Sequence, None] = None) -> str:
    """Builds the router prompt, optionally including retrieved past plans.

    `past_memory` may be:
      - None / empty: no memory block (core design, docs/04).
      - a list of {"question","plan"} dicts or (question, plan) tuples (memory-on path).
      - a pre-rendered string (accepted for back-compat).
    """
    if past_memory is None or past_memory == "":
        memory_block = ""
    elif isinstance(past_memory, str):
        memory_block = f"Similar problems you have solved before:\n{past_memory}\n\n"
    else:
        memory_block = _render_memory_block(past_memory)

    # Assemble by concatenation (not str.format) so memory content containing
    # "{" or "}" cannot break format-string parsing.
    return (
        f"{ROUTER_HEADER}\n\n"
        f"{memory_block}"
        f"Problem: {question}\n\n"
        f"{ROUTER_FORMAT_LINE}\n\n"
        f"Plan:\n"
    )


def build_solver_prompt(question: str, plan_json: str, scratchpad: str, current_subgoal: str) -> str:
    """Builds the solver prompt."""
    return SOLVER_PROMPT_TEMPLATE.format(
        question=question,
        plan_json=plan_json,
        scratchpad=scratchpad,
        current_subgoal=current_subgoal,
    )
