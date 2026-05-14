# baseline

Eval harness for three models on full GSM8K test set (1319 questions). Greedy decoding, matched prompting. Source of Table 1 in report.

## Models

1. Qwen2.5-1.5B-Instruct (general base)
2. Qwen2.5-1.5B-Instruct + RL (from `../linear_reasoning/main.py`)
3. Qwen2.5-Math-1.5B-Instruct (math-specialized base)

## Running

```bash
python eval_all_baselines.py                    # Generate all predictions
python compare_models.py                        # Print comparison table

# Subset:
python eval_all_baselines.py --num_questions 100
```

Outputs: `baseline_results_all.json` (predictions + scores) and `comparison_report.md` (table).

Time: ~30-40 minutes on L4 for full set.

## Special considerations

- RL checkpoint: uses `../linear_reasoning/checkpoints/final_model.pt`. Update if retraining.
- Answer extraction: last `\boxed{N}` or last numeric token (same as reward function)
- System message: "You are a math tutor..." is critical — removing it hurts general base by 5-8pp
- Eval at greedy (T=0). Self-consistency evals in `../dapo_linear_math/`.
