# grpo_linear_math_version

GRPO on math-specialized base (Qwen2.5-Math-1.5B-Instruct). Result: 84.8% accuracy (zero RL gain, same as base).

Same recipe as `../linear_reasoning/` but:
- Base: Qwen2.5-Math-1.5B-Instruct (not general base)
- Training set: ~1000 questions from probe-derived buckets (mixed/hard/trivial)
- Result: Policy produces different text but no improvement

## Key finding

The math base is already strong (84.8% base accuracy). ~70% of GRPO groups are all-correct (zero variance), producing zero gradients. KL penalty saturates within 5 steps. LoRA-rank8 cannot move within the 0.04 KL ball. Documented in report Section 4.

## Running

```bash
python main.py --mode train_eval                  # Train + eval
python main.py --mode train_eval --bucket mixed_hard   # Harder questions only
python main.py --mode eval --checkpoint <path>    # Eval saved checkpoint
```

Time: ~7 hours on L4 (1000 train questions, G=6).

## Special considerations

- **Dead-gradient regime**: ~70% of groups uninformative on math base
- **Probe buckets**: partition.json defines mixed (583q), hard (286q), trivial (131q)
- **LoRA size**: 1.09M params (~0.07% of model) — controlled to isolate the base model effect
- Eval at greedy decoding (T=0) as in report
