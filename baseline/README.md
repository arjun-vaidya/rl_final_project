# baseline

The three-model GSM8K eval harness that produces the headline table in the report. Greedy decoding, matched prompting, same chat template across all three models. Output of this folder is the source of truth for Table 1 in the report.

## What this directory does

Runs the full 1319-question GSM8K test set against three models and writes a single JSON with every prediction, plus a small comparison report. The three models are:

- Qwen2.5-1.5B-Instruct (off-the-shelf general base)
- Qwen2.5-1.5B-Instruct + RLVR (the checkpoint produced by `../linear_reasoning/`)
- Qwen2.5-Math-1.5B-Instruct (off-the-shelf math-specialized base)

The harness mirrors the prompt and decoding from training (same system message, greedy, 512 max new tokens, batched at 32 with left-padding) so off-the-shelf baselines are not penalized by prompt-format drift.

## How to run

```bash
cd ..   # the eval scripts use absolute paths relative to the repo root
python baseline/eval_all_baselines.py
python baseline/compare_models.py

# Subset (fast check):
python baseline/eval_all_baselines.py --num_questions 100
python baseline/compare_models.py --num_questions 100
```

`eval_all_baselines.py` is the heavy step; it generates and scores all 1319 questions for each model and writes `baseline_results_all.json`. `compare_models.py` reads that JSON and emits a Markdown comparison table.

## Important files

| Path | Purpose |
|---|---|
| `eval_all_baselines.py` | runs all three evals and dumps per-question predictions to `baseline_results_all.json` |
| `compare_models.py` | reads the JSON, prints per-model accuracy and writes `comparison_report.md` |
| `baseline_results_all.json` | the committed reference run on the full test set (used by the report) |
| `eval.log` | log from the reference run |
| `qa_log_general.txt`, `qa_log_math_specialized.txt`, `qa_log_rl_final_model.txt` | per-model trace dumps for spot-checking model behavior |

## Notes and caveats

- The RL-trained checkpoint that the third row uses is the `final_model.pt` produced by `../linear_reasoning/main.py`. If you retrain, point `eval_all_baselines.py` at the new path.
- Answer extraction takes the last `\boxed{N}` match and falls back to the last numeric token. Numeric matching is float-based with a 1e-6 tolerance. Same logic as `../linear_reasoning/src/reward.py`.
- All three models share the same system message ("You are a math tutor. Solve the problem step by step..."). Removing the system message hurts the general base by 5-8 points and is the most likely way to accidentally make the comparison look better than it is.
- Greedy decoding (`do_sample=False`) is used throughout. Self-consistency / majority-vote evaluations live in `../dapo_linear_math/src/eval_sc.py`, not here.
