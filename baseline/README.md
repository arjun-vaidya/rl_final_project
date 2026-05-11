# Baseline Evaluation

Compare three models on GSM8K:
- Qwen2.5-1.5B-Instruct (general baseline)
- Qwen2.5-Math-1.5B-Instruct (math-specialized baseline)
- Router-Solver V2 (your RL-trained model)

## Usage

```bash
cd /Users/vaidya/Documents/Spring\ 2026/orcs6529/final_project

# Evaluate baselines (full test set)
python baseline/eval_all_baselines.py

# Generate comparison report
python baseline/compare_models.py
```

To test on a subset instead of the full 1319 questions:
```bash
python baseline/eval_all_baselines.py --num_questions 100
python baseline/compare_models.py --num_questions 100
```

Outputs:
- `baseline_results_all.json` — baseline accuracies
- `comparison_report.md` — three-way comparison table
