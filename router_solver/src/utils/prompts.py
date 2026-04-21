"""Prompt templates.

Spec: docs/04_design.md §Prompts, docs/07_plan_memory.md §"Router prompt (with memory)".
Keep these as module-level strings so they're easy to diff and ablate.
"""


ROUTER_PROMPT_NO_MEMORY = """\
You are a planner. Output a JSON plan of subgoals for this math problem.

Problem: {question}

Format: {{"plan": [{{"subgoal": "<desc>", "tool": "python"}}, ...]}}

Plan:
"""


ROUTER_PROMPT_WITH_MEMORY = """\
You are a planner. Output a JSON plan of subgoals for this math problem.

Similar problems you have solved before:
{retrieved_plans_block}

Problem: {question}

Format: {{"plan": [{{"subgoal": "<desc>", "tool": "python"}}, ...]}}

Plan:
"""


SOLVER_PROMPT = """\
You are solving one subgoal of a larger problem.

Original problem:  {question}
Full plan:         {plan_json}
Previous results:  {scratchpad}
Current subgoal:   {current_subgoal}

Write Python code (one expression or short block) wrapped in <code>...</code>.
"""


FLAT_PROMPT = """\
Solve this math problem. You may call Python by writing <code>...</code>.
When you are done, output your final answer as <answer>N</answer>.

Problem: {question}
"""


def build_router_prompt(question: str, retrieved: list[dict] | None = None) -> str:
    """Return the memory-off or memory-on variant depending on `retrieved`."""
    raise NotImplementedError


def build_solver_prompt(question: str, plan_json: str, scratchpad: str, current_subgoal: str) -> str:
    raise NotImplementedError


def build_flat_prompt(question: str) -> str:
    raise NotImplementedError
