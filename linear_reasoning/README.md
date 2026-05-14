# linear_reasoning

Single-pass CoT policy trained with GRPO on GSM8K. Result: 71.6% accuracy (+1.9pp improvement).

Core scaffolding reused by math-base GRPO, DAPO, and other experiments.

## Setup

- Base: Qwen2.5-1.5B-Instruct
- LoRA: rank 8 on q_proj, v_proj (~1M trainable parameters)
- Training: 2000 GSM8K questions, G=6 rollouts, KL penalty 0.04
- Reward: +1.0 for correct `\boxed{N}`, +0.5 format bonus, 0 otherwise
- Inference: Greedy decoding on full test set (1319 questions)

## Running

```bash
python main.py --mode train_eval                    # Full training + eval
python main.py --mode eval --checkpoint <path>      # Eval only
python main.py --mode train --no_wandb               # Training without W&B
```

Key flags:
- `--base_model`: swap base (default: Qwen/Qwen2.5-1.5B-Instruct)
- `--rollouts_per_q`: G (default: 6)
- `--learning_rate`: AdamW LR (default: 1e-5)
- `--kl_coef`: KL penalty (default: 0.04)
- `--train_questions`: limit train set (default: all 2000)

## Key files

- `main.py` — Entry point, model loading, trains and evaluates
- `src/agent.py` — `LinearReasoningAgent`, batch generation, prompt formatting
- `src/reward.py` — `compute_reward`, boxed answer extraction
- `src/train.py` — GRPO training loop
- `src/eval.py` — Test set evaluation
- `requirements.txt` — Dependencies

## Special considerations

- **Reward strictness**: Only `\boxed{}` answers get reward. Unboxed correct numbers score 0 to prevent wandering.
- **KL reference**: Uses model with LoRA disabled (`model.disable_adapter()`) instead of separate reference model.
- **Eval noise**: Eval at T=0 (greedy). Nonzero temperature adds ±1pp variance on 1319-question test set.
- **Checkpoints**: Save full state dict (~3GB). Use PEFT `save_pretrained` if space is tight.
- **Memory**: ~7 hours on L4 for full training pass (2000 questions, G=6).
