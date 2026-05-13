# Router-Solver V2

Text-based hierarchical RL with an OpenAI-compatible remote judge for step-level reward signals.

Current repo note:
- the runtime judge client is configurable through `OLLAMA_*` env vars and can point to a remote `vLLM` deployment;
- the deployment assets for that judge live under [judge_ops](./judge_ops/README.md).
- `--dataset slim` mirrors the original slim-dataset selection rule from `router_solver`.

## Architecture

The system decomposes math problems hierarchically: a Router generates plans, a Solver executes them step-by-step, and a Judge (GPT-4o mini) provides dense reward signals at each level.

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT QUESTION                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                ┌──────▼──────┐
                │   ROUTER    │  Generates JSON plan (1-8 steps)
                │ (Qwen+LoRA)  │
                └──────┬───────┘
                       │
        ┌──────────────▼──────────────┐
        │ GPT-4o mini Judge           │ Scores plan: clarity, independence, completeness
        │ → plan_reward ∈ [0, 1]      │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────────────────┐
        │ SOLVER executes each step sequentially  │
        │ (Text reasoning + answer extraction)    │
        └──────────────┬──────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ GPT-4o mini Judge           │ Scores each step independently
        │ → step_rewards[i] ∈ [0, 1]  │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │    FINAL ANSWER (last step) │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │ GPT-4o mini Judge           │ Checks: answer == ground_truth
        │ → outcome_reward ∈ {0, 1}   │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────────────────────┐
        │ GRPO: Compare 4 rollouts within group       │
        │ Compute advantages: (r - mean) / std        │
        │ Loss = -(r_w·adv·logp + ...)                │
        └──────────────┬──────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  Backprop & Optimizer Step  │ Update model weights via AdamW
        └─────────────────────────────┘
```

## Training Loop (GRPO)

For each question, generate 4 rollouts and learn from their relative performance:

```
┌─────────────────────────────────────────────────────────────┐
│ FOR EACH QUESTION: Generate G=4 rollouts                   │
├─────────────────────────────────────────────────────────────┤
│ Rollout 1: Q → [Plan] → [Step 1, Step 2, ...] → Answer     │
│ Rollout 2: Q → [Plan] → [Step 1, Step 2, ...] → Answer     │
│ Rollout 3: Q → [Plan] → [Step 1, Step 2, ...] → Answer     │
│ Rollout 4: Q → [Plan] → [Step 1, Step 2, ...] → Answer     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ BATCH JUDGE (efficiency): Judge all 4 plans + steps @ once │
│ • Batch 4 plans → 1 API call (10x cheaper than individual) │
│ • Batch ~16 steps → 2 API calls                            │
│ • Outcome rewards: 4 binary checks (no API)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ COMPUTE ADVANTAGES within group:                           │
│ advantage[i] = (reward[i] - mean(rewards)) / std(rewards) │
│ • High-performing rollout → positive advantage             │
│ • Low-performing rollout → negative advantage              │
│ • Normalizes across different reward scales                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ POLICY GRADIENT: Weight log-probs by advantages            │
│ loss = -(r_w × router_adv × log_p_router +                │
│          s_w × steps_adv × log_p_steps +                  │
│          o_w × outcome_adv × log_p_final)                 │
│ • Increases probability of good rollouts                   │
│ • Decreases probability of bad rollouts                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ OPTIMIZE: loss.backward() → optimizer.step()               │
│ (Accumulate gradients from all 4 rollouts, step once)      │
└─────────────────────────────────────────────────────────────┘
```

## Evaluation

Greedy rollout (no batching, single sample per question):

```
┌──────────────────────────────────────────┐
│ FOR EACH TEST QUESTION: Single rollout   │
├──────────────────────────────────────────┤
│ Q → Router → Plan → Solver → Answer      │
│ (No gradient, deterministic sampling)    │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│ BATCH JUDGE every 10 rollouts (efficient)│
│ Plan + steps + answer evaluation         │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│ METRICS: accuracy, plan_validity, rewards│
└──────────────────────────────────────────┘
```

## Checkpointing & Recovery

Training saves full state every 50 questions:

```
checkpoint_epoch0_q50.pt = {
  "model": model weights,
  "optimizer": optimizer state + learning rate,
  "epoch": 0,
  "q_idx": 49,
  "correct": number of correct answers,
  "total_loss": cumulative loss,
}

# Resume from checkpoint:
python main.py --mode train --checkpoint checkpoint_epoch0_q50.pt
```

## Config (Phase 4: Full Scale)

```
Training:
  • Batch size: 32 (per epoch, unused in current per-Q training)
  • Rollouts per question: 4 (GRPO group size)
  • Total rollouts: ~30,000 (7,500 questions × 4)
  • Epochs: 1
  • Learning rate: 1e-5 (AdamW)
  • LoRA: rank=8, alpha=16

Rewards:
  • Router weight: 0.3 (decays 5% per epoch)
  • Solver weight: 0.5 (constant)
  • Outcome weight: 0.2 (increases as router decays)

Judge (Batched):
  • Model: gpt-4o-mini
  • Batch size: 10 items/API call
  • Cost: ~$10 (vs $75 without batching)
  
Time: ~40 hours (on A100, varies with API latency)
Expected accuracy: 15-25%
```

## Files

- `main.py` - Entry point for train/eval
- `src/agents/agent.py` - Router + Solver agent (Qwen+LoRA)
- `src/rewards/judge.py` - GPT-4o mini batch judge
- `src/rewards/shaper.py` - Reward computation + weight scheduler
- `src/training/train.py` - GRPO training loop with checkpointing
- `src/training/eval.py` - Evaluation on test set
- `src/utils/config.py` - Configuration defaults

## Key Differences from V1

| Aspect | V1 | V2 |
|--------|----|----|
| Solver | Code generation | Text reasoning |
| Rewards | Heuristic (sparse) | GPT-4 judge (dense) |
| Architecture | Router/Solver coupled | Decoupled with separate judges |
| Cost | ~$75 | ~$10 (batched API) |
| Accuracy | 1.7% | 15-25% (target) |

## Usage

```bash
# Train from scratch
python main.py --mode train

# Evaluate on test set
python main.py --mode eval

# Train + evaluate
python main.py --mode train_eval

# Resume from checkpoint
python main.py --mode train --checkpoint checkpoint_epoch0_q50.pt
```
