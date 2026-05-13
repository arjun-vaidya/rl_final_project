# data_probing

Per-question difficulty probe of the GSM8K-train split. We use it to construct the trivial / mixed / hard partition that drives the math-base RL experiments in `../grpo_linear_math_version/` and `../dapo_linear_math/`.

## What this directory does

For each of the 7473 GSM8K-train questions, draw G=4 rollouts at temperature 0.8 from Qwen2.5-Math-1.5B-Instruct (no LoRA, no RL, just the base) and record how many are correct. The output is a CSV with one row per question and a JSONL with the raw rollout text. The CSV is the input to `make_partition.py` which then writes the `partition.json` used by the math-base experiments.

## Why this matters

GRPO computes within-group advantages as `(reward - group_mean) / group_std`. When every rollout in a group gets the same reward the std collapses and every advantage is zero, so that group contributes no policy gradient. On the math-specialized base, this is the dominant failure mode of vanilla GRPO. The probe quantifies it:

```
n_correct    count      pct  meaning
  0/4          286    3.83%  unsolvable for this model (no gradient)
  1/4          101    1.35%  useful (variance)
  2/4          142    1.90%  useful (variance)
  3/4          340    4.55%  useful (variance)
  4/4         6604   88.37%  trivial (no gradient)

At least one correct rollout: 7187  (96.17%)
Mixed 1-3 of 4 (useful RL):    583  (7.80%)
```

So only ~8% of GSM8K-train is in the band where vanilla GRPO can learn on the math base at step 0. The probe gives us the indices of that 8% so we can build a focused training subset, and the indices of the all-correct mass so we can sample a "trivial anchor" subset to keep KL coverage on easy problems.

## How to run

```bash
cd data_probing

# Probe (resumable; appends to outputs if interrupted). Default base is Qwen2.5-Math-1.5B-Instruct.
python probe.py

# Probe with a different base.
python probe.py --base_model Qwen/Qwen2.5-1.5B-Instruct

# Print the bucket distribution from the CSV.
python analysis.py
```

The probe writes `probe_summary.csv` (one row per question) and `probe_rollouts.jsonl` (one record per rollout) incrementally so a crash does not lose the work. To rebuild `partition.json` after a fresh probe, run `../grpo_linear_math_version/make_partition.py`.

## Important files

| Path | Purpose |
|---|---|
| `probe.py` | the probe loop, resumable, appends to outputs |
| `analysis.py` | reads `probe_summary.csv` and prints the n_correct distribution above |
| `probe_summary.csv` | one row per question: `idx, n_correct, acc, max_reward, any_boxed` |
| `probe_rollouts.jsonl` | one record per rollout, includes full generated text and pred / correct flags |
| `probe.log` | VM run log from the reference probe |

## Notes and caveats

- The probe uses the same chat template and `\boxed{}` extraction as training. If you change the prompt or reward in `../linear_reasoning/src/`, re-probe so the buckets stay in sync with what training will see.
- `any_boxed = False` rows are format failures, not reasoning failures. There were 27 of these in the reference probe. They get 0 reward but are not interesting to train on.
- The G=4 / T=0.8 setting matches the training-time temperature. Probing at T=0 gives a different (and less useful) partition since most questions become deterministic.
- Probing is single-GPU and roughly linear in question count. The reference 7473-question probe takes ~3 hours on an A100. Use `--limit N` for a quick smoke run.
