# src/utils/prompts.py

ROUTER_PROMPT_TEMPLATE = """You are a planner. Output a JSON plan of subgoals for this math problem.

Problem: {question}

Format: {{"plan": [{{"subgoal": "<desc>", "tool": "python"}}, ...]}}

Plan:
"""

SOLVER_PROMPT_TEMPLATE = """You are solving one subgoal of a larger problem.

Original problem:  {question}
Full plan:         {plan_json}
Previous results:  {scratchpad}
Current subgoal:   {current_subgoal}

Write Python code (one expression or short block) wrapped in <code>...</code>.
"""

def build_router_prompt(question: str, past_memory: str = "") -> str:
    """Builds the router prompt, optionally including past memory."""
    if past_memory:
        # According to 04_design.md, memory block is optional
        memory_block = f"Similar problems you have solved before:\n{past_memory}\n\n"
        prompt = ROUTER_PROMPT_TEMPLATE.replace("Problem:", f"{memory_block}Problem:")
        return prompt.format(question=question)
    return ROUTER_PROMPT_TEMPLATE.format(question=question)

def build_solver_prompt(question: str, plan_json: str, scratchpad: str, current_subgoal: str) -> str:
    """Builds the solver prompt."""
    return SOLVER_PROMPT_TEMPLATE.format(
        question=question,
        plan_json=plan_json,
        scratchpad=scratchpad,
        current_subgoal=current_subgoal
    )
