# linear_reasoning

The single-pass chain-of-thought policy and GRPO training scaffolding for this project. Every other RL experiment in the repo (math-base GRPO, DAPO, ablations) reuses the agent, reward, and eval modules from here via `sys.path`.

## What this directory does

A LoRA adapter on top of Qwen2.5-1.5B-Instruct is trained with GRPO and a verifiable reward to solve GSM8K. The policy emits one chain of thought ending in `\boxed{N}`. There is no router, no separate solver, and no judge. The reward is +1.0 for a numerically-correct boxed answer, +0.5 format bonus for any parseable boxed token, and 0 otherwise. The full training recipe and the held-out test result (71.6%, +1.9 over the 69.7% base) is what Section 3 of the report describes.

## Why this design

The earlier hierarchical Router-Solver attempt (see `../router_solver/`, `../router_solver_v2/`) plateaued around 20-35% accuracy on GSM8K. The V2 failure taxonomy in `../notes/2026-05-09_peter_pov_answer_target_vs_core_reasoning.md` showed that 51.7% of failures were architectural (plan parse, last-step-equals-final-answer, intermediate-as-final), not reasoning. The same model with a single-pass CoT and a verifiable reward has none of those failure modes, and the GSM8K ground truth is already a perfectly clean reward signal. So we dropped the hierarchy and used GRPO with RLVR, the same recipe DeepSeekMath and DeepSeek-R1 use at larger scale.

LoRA only because of compute. Rank 8 on `q_proj, v_proj` is around 1M trainable parameters, which is enough on the general base to produce a measurable lift and small enough that one training pass fits in 7 hours on a single A100. The reward is intentionally minimal (no learned RM, no PRM, no partial credit) to remove every source of reward-hacking that does not have to be there.

## How to run

Training and eval are wired through `main.py`.

```bash
cd linear_reasoning

# Train + eval on the general Qwen base, full GSM8K train split (~7 hours on A100).
python main.py --mode train_eval

# Eval only, from a saved checkpoint.
python main.py --mode eval --checkpoint experiments/<run_name>/checkpoint_q2000.pt

# Train only.
python main.py --mode train

# Quick smoke run (50 questions, ~10 min).
python main.py --mode train_eval --train_questions 50 --eval_questions 200
```

Common overrides:

| Flag | Default | What it changes |
|---|---|---|
| `--base_model` | `Qwen/Qwen2.5-1.5B-Instruct` | swap base |
| `--rollouts_per_q` | 6 | GRPO group size G |
| `--learning_rate` | 1e-5 | AdamW LR |
| `--kl_coef` | 0.04 | KL penalty against the frozen base |
| `--train_microbatch_size` | 3 | trajectories per forward/backward |
| `--no_wandb` | off | disable W&B logging |

## Important files

| Path | Purpose |
|---|---|
| `main.py` | argparse wiring, model + LoRA load, calls `train()` and `evaluate()` |
| `src/agent.py` | `LinearReasoningAgent`, chat-template prompt, batched generate, trim-at-EOS |
| `src/reward.py` | `compute_reward` with the +1.0 / +0.5 / 0 schedule and `\boxed{}` extraction |
| `src/train.py` | GRPO loop: per-question rollouts, group-relative advantage, microbatched policy update with reference logp under disabled adapter |
| `src/eval.py` | greedy eval over the test set, writes `eval_results.json` |
| `src/config.py` | the `Config` dataclass that all four scripts read |
| `assets/` | W&B chart exports used in the report |
| `requirements.txt` | torch, transformers, peft, datasets, wandb |

## Notes and caveats

- The reward is strict: a generation that gets the right number but does not wrap it in `\boxed{}` scores 0. This is intentional; without the format constraint the model wanders and the answer-extraction regex becomes a second adversary.
- The KL term uses the k3 unbiased estimator from DeepSeek-R1, computed against the same model with the LoRA adapter disabled (via `model.disable_adapter()`), not against a separately-loaded reference model. This halves memory.
- `eval_temperature` defaults to 0 (greedy). Switching to nonzero adds run-to-run noise on the order of +/-1 percentage point on the 1319-question test set.
- Checkpoints save the full model state dict, including frozen base weights. That makes them self-contained but ~3 GB each. Switch to PEFT's `save_pretrained` if disk is tight.
- The reward function uses a fallback extractor on the last numeric token only for the eval `predicted` field, not for the reward itself. Logged predictions can therefore be non-numeric (`null`, `Infinity`) when the model goes off the rails; these still receive 0 reward.
