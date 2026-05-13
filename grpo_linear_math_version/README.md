# grpo_linear_math_version

The same RL recipe from `../linear_reasoning/` applied to the math-specialized base Qwen2.5-Math-1.5B-Instruct. Trained on a 1000-question probe-derived subset of GSM8K-train. The result is exactly the base-model accuracy (84.8% on the 1319-question test set), with the policy producing different reasoning text on every problem but no net change in correctness. Section 4 of the report unpacks why.

## What this directory does

Identical pipeline to `../linear_reasoning/`: chain-of-thought policy, LoRA-r8 on `q_proj, v_proj`, GRPO with verifiable rewards, KL=0.04 against the frozen base. Only three things differ:

1. base model is Qwen2.5-Math-1.5B-Instruct,
2. training set is `partition.json` (probe-derived buckets, ~1000 questions), not the full 7473,
3. eval and training go through this folder's `main.py` so the run name and output dir are independent.

The eval, agent, reward, and GRPO loop are imported from `../linear_reasoning/src/` via `sys.path`.

## Why this design

After the +1.9 point gain on the general base, the natural question was whether stacking the same recipe on a math-pretrained base would compound. We picked the math base because it was the strongest 1.5B GSM8K solver we had available, and we built the partition.json subset using the probe in `../data_probing/` so we would not waste compute on questions where every rollout is identical.

The training dynamics are the negative-result content. KL pinned at the 0.04 penalty within five steps; rolling accuracy oscillated in [0.45, 0.65] with no upward trend; cumulative accuracy flatlined at ~55% after 150 steps. On the test set, the post-training policy and the base agree on 1291 of 1319 problems, with 14 newly-correct and 14 newly-wrong (consistent with sampling noise at T > 0).

The mechanism is in Section 4 of the report. Briefly: GRPO produces gradient only when a group of G rollouts has within-group variance, and on the math base roughly 70% of training groups are all-correct (zero variance) and another 15% are all-wrong, so most steps contributed only KL drag toward the reference. LoRA-r8 on q,v inside the 0.04 KL ball did not have the capacity or the freedom to move.

## How to run

```bash
cd grpo_linear_math_version

# Default: 1-epoch GRPO on bucket=all (mixed + hard + trivial), eval on the full test set.
python main.py --mode train_eval

# Train only on the harder questions (the probed mixed + hard bucket, no trivial).
python main.py --mode train_eval --bucket mixed_hard

# Quick smoke (50 train questions, 200 eval questions).
python main.py --mode train_eval --eval_questions 200

# Run only the eval against a saved checkpoint.
python main.py --mode eval --checkpoint runs/<run_name>/final_model.pt
```

Bucket selection comes from `partition.json` which is produced by `../data_probing/probe.py`. The mapping `mixed / hard / trivial / mixed_hard / all` is documented in `main.py`.

## Important files

| Path | Purpose |
|---|---|
| `main.py` | argparse wiring, swaps `cfg.base_model` to the math base, loads `partition.json`, calls the shared `train()` and `evaluate()` |
| `make_partition.py` | one-shot script that rebuilds `partition.json` from `../data_probing/probe_summary.csv` |
| `partition.json` | per-bucket question indices into the GSM8K train split (counts: mixed 583, hard 286, trivial 131) |
| `assets/` | W&B chart exports used in the report (cumulative accuracy, KL, etc.) |
| `runs/` | per-run training checkpoints, `train.log`, `eval_results.json` |
| `train.log` | the run log from the documented experiment in the report |

## Notes and caveats

- The trainable-parameter count is small (~1.09M, 0.07% of the model) by design. Bigger LoRA ranks would change the experiment; see `../dapo_linear_math/` for the follow-up that does change them.
- `eval_results.json` truncates the `output` field at 400 characters and the `question` at 200. This is intentional, set in `../linear_reasoning/src/eval.py`. If you need the full traces, edit those slice limits and re-run eval.
- Two trajectories in our recorded run produced `"predicted": Infinity` in the eval JSON because the model emitted a several-hundred-character run of the same digit and the float-cast overflowed. The reward function still scores those as 0, so accuracy is unaffected.
- The probed mixed bucket (583 questions where 1-3 of 4 probe rollouts were correct) is the only bucket that produces non-zero advantages on the math base at step 0. If you want a faster sanity-check run with maximum signal per step, use `--bucket mixed`.
- The follow-up in `../dapo_linear_math/` attacks the dead-gradient problem documented here directly, via DAPO-style dynamic sampling and a bigger LoRA.
