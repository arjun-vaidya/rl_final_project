# Next Steps & Implementation Logic

This guide explains how to run the project and what each component does.

## Setup
First, prepare your environment on a GPU-enabled machine (4090, A100, etc.):
```bash
# Run the setup script to create a venv and install dependencies
bash scripts/setup_env.sh

# Activate the environment
source .venv/bin/activate
```

## 2. Running the Experiments

### Condition #2: Flat Baseline (Gate 2)
To train the standard model with sparse internal rewards:
```bash
python3 src/training/train_flat.py --config configs/flat.yaml
```

### Condition #4: Router-Solver (Gate 3)
To train the hierarchical model with decomposed rewards:
```bash
python3 src/training/train_router_solver.py --config configs/router_solver.yaml
```

### Evaluation
To evaluate any saved checkpoint:
```bash
bash scripts/run_eval.sh experiments/flat_baseline/final_model val
```

---

## Project Structure

### `configs/`
Holds the "Brains" of the experiments. Change weights, learning rates, or memory modes here.
- `flat.yaml`: Baseline config.
- `router_solver.yaml`: Core hierarchical config.
- `router_solver_memory_sim.yaml`: Memory-enabled experiment.

### `src/env/`
- `gsm8k_loader.py`: Handles data ingestion and answer parsing.
- `python_tool.py`: The sandboxed Python executor (the agent's "calculator").

### `src/agents/`
- `flat_agent.py`: Simple one-shot reasoning model.
- `router_solver_agent.py`: The complex agent that switches between Planning (Router) and Execution (Solver) specialized adapters.

### `src/rewards/`
- `outcome.py`: Binary 0/1 success signal.
- `router.py`: Rewards the "Manager" for structural plan quality.
- `solver.py`: Rewards the "Specialist" for clean tool usage even if the final result is wrong.

### `src/memory/`
- `retrieval.py`: The long-context memory layer that allows the agent to recall past similar plans.

### `tests/`
- Verified suite of 20 tests. Run `PYTHONPATH=. python3 -m unittest discover tests` before any major push.

---

## Immediate Next Steps
1. **GPU Training:** Peter and Arjun should pick up the code and start the `train_flat.py` run to hit the Apr 25 milestone.
2. **Diagnostic implementation:** We still need to finalize the "Gradient Conflict" plot generator in `evaluate.py`.
3. **Plan Memory:** Once the hierarchy is stable, enable memory in `router_solver_memory_sim.yaml` to test Pillar #2 of the pitch.
