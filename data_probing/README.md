# data_probing

Per-question difficulty probe of the GSM8K-train split with Qwen2.5-Math-1.5B-Instruct (G=4 rollouts per question, temperature 0.8).

## Analysis output

```
Total questions: 7473
G = 4 rollouts/question

n_correct    count      pct  meaning
  0/4          286    3.83%  unsolvable for this model
  1/4          101    1.35%  useful (variance)
  2/4          142    1.90%  useful (variance)
  3/4          340    4.55%  useful (variance)
  4/4         6604   88.37%  trivial - model always gets it

At least one correct rollout: 7187  (96.17%)
Zero correct rollouts (hard): 286  (3.83%)
All 4 correct (trivial):       6604  (88.37%)
Mixed 1-3/4 (useful RL):       583  (7.80%)
No boxed answer in any rollout: 27  (0.36%)
```

## Why we did this

GRPO computes within-group advantages as `(reward - group_mean) / group_std`. When all G rollouts in a group return the same reward, the std collapses and every advantage is zero — that group contributes no policy gradient.

For Qwen2.5-Math-1.5B-Instruct on GSM8K-train:
- **88.37% trivial** (all 4 rollouts correct) → advantage = 0
- **3.83% hard** (0 of 4 correct) → advantage = 0
- **Only 7.80% mixed** (1-3 of 4 correct) → non-zero advantage, real gradient

Without filtering, ~92% of GRPO compute would be spent generating rollouts that produce zero gradient. Probing tells us exactly which questions are in the useful band so we can build a focused training subset (see `../grpo_linear_math_version/`). It also surfaces secondary issues — e.g. 27 questions where no rollout produced a `\boxed{}` answer at all, which are format failures rather than reasoning failures.

## Files

- `probe.py` — runs the probe (resumable; appends to outputs)
- `analysis.py` — prints the distribution above from `probe_summary.csv`
- `probe_summary.csv` — one row per question: idx, n_correct, acc, max_reward, any_boxed
- `probe_rollouts.jsonl` — full rollout text per question (used for SFT seeds / inspection)
- `probe.log` — VM run log
