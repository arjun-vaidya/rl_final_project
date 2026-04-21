# 04 · Technical Design

## Stack

| Component | Choice | Why |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct | Fits on one 24 GB GPU; strong tool-use priors; standard RLVR baseline |
| Fine-tuning | LoRA (rank 16, α 32) | Cheap; composable (two adapters on one base) |
| RL algorithm | GRPO | No critic; current default for LLM RLVR |
| RL framework | TRL ≥ 0.11 (`GRPOTrainer`) | Maintained, HF-native |
| Rollout acceleration | vLLM or Unsloth | 2–5× faster generation |
| Tool sandbox | Local Python subprocess + timeout | Cheapest option |
| Tracking | Weights & Biases | Free tier |

Two LoRA adapters on one frozen base model; swap active adapter between Router and Solver calls.

## Prompts

### Router

```
You are a planner. Output a JSON plan of subgoals for this math problem.

Problem: {question}

Format: {"plan": [{"subgoal": "<desc>", "tool": "python"}, ...]}

Plan:
```

### Solver

```
You are solving one subgoal of a larger problem.

Original problem:  {question}
Full plan:         {plan_json}
Previous results:  {scratchpad}
Current subgoal:   {current_subgoal}

Write Python code (one expression or short block) wrapped in <code>...</code>.
```

## Reward functions

### Outcome (used by both agents)

```python
def outcome_reward(trajectory, gt) -> float:
    return 1.0 if extract_numeric_answer(trajectory) == gt else 0.0
```

### Router

```python
def router_reward(plan_json, trajectory, gt) -> float:
    if not is_valid_json(plan_json):        return 0.0
    if "plan" not in plan:                  return 0.0
    if not (1 <= len(plan["plan"]) <= 8):   return 0.0
    return outcome_reward(trajectory, gt)   # gated downstream credit
```

The Router reward is the outcome, **gated** by structural validity. We deliberately don't use an LLM judge on plan quality — that would confound the experiment with reward-model quality.

### Solver (per step)

```python
def solver_step_reward(code, tool_output, final_outcome) -> float:
    r = 0.0
    if is_error(tool_output):   return 0.0
    r += 0.3   # tool executed cleanly
    if looks_sensible(tool_output):  r += 0.2   # non-empty, numeric/string, not absurd
    if final_outcome == 1.0:    r += 0.5
    return r
```

Weights (0.3 / 0.2 / 0.5) are a first guess — swept in ablation A.

### Flat baseline

```python
def flat_reward(trajectory, gt) -> float:
    return outcome_reward(trajectory, gt)
```

One signal applied to all tokens.

## Training loop (Router–Solver)

```python
for step in range(num_steps):
    questions = sample_batch(train_set, B=8)

    plans = [router.generate(q, n=G) for q in questions]        # B × G plans

    trajectories = [
        solver.execute(q, plan)
        for q, plan_group in zip(questions, plans)
        for plan in plan_group
    ]

    router_rewards = [router_reward(p, t, gt) for ...]
    solver_rewards = [solver_step_reward(...)  for ...]

    router_loss = grpo_loss(router_rewards, grouping=by_question)
    solver_loss = grpo_loss(solver_rewards, grouping=by_question)

    (router_loss + solver_loss).backward()
    optimizer.step()
```

Both adapters updated in one step with summed losses. Alternative: alternating updates — keep as fallback if joint is unstable.

## GRPO hyperparameters (starting point)

| Param | Value |
|---|---|
| Group size G | 8 |
| Batch B | 8 questions (64 rollouts/step) |
| Learning rate | 5e-6 |
| KL coefficient β | 0.04 |
| Max gen tokens (Router) | 256 |
| Max gen tokens (Solver per step) | 128 |
| Max subgoals | 6 |
| Total steps | ~500 (≈ 1 epoch at B=8) |

## Compute budget

**Target:** 1× 24 GB GPU (RTX 4090 / A10 / A5000), ~24 h per run. Qwen2.5-1.5B + rank-16 LoRA fits in bf16 with gradient checkpointing; vLLM handles rollouts in a separate process.

**Fallback:** Colab free tier → drop to Qwen2.5-0.5B. Absolute numbers go down but the flat-vs-hierarchical comparison still holds.

## Code layout (proposed — no code yet)

```
router_solver/
├── README.md
├── docs/
├── src/
│   ├── env/                  # gsm8k_loader.py, python_tool.py
│   ├── agents/               # flat_agent.py, router_solver_agent.py
│   ├── rewards/              # outcome.py, router.py, solver.py
│   ├── training/             # train_flat.py, train_router_solver.py
│   └── eval/                 # evaluate.py
├── configs/                  # flat.yaml, router_solver.yaml
├── experiments/              # (gitignored) runs + logs
└── tests/
```
