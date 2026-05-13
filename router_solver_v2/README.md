# router_solver_v2

The second hierarchical attempt. Text-reasoning Solver, GPT-4o-mini as a remote judge for dense step-level rewards, three-term weighted GRPO advantage. Reached approximately 35% rollout accuracy on a fresh traced eval with answer-synthesis on. Section 5.2 of the report unpacks why this is still 50 points below the linear-policy baselines.

## What this directory does

The pipeline at training time:

1. **Router** (Qwen + LoRA) generates a JSON plan with 1-8 subgoal steps for the input question.
2. A remote **GPT-4o-mini Judge** scores the plan (clarity, independence, completeness) and returns `plan_reward in [0, 1]`.
3. **Solver** (same Qwen base, separate LoRA head) executes each step sequentially, producing text reasoning.
4. The Judge scores each Solver step independently, returning `step_rewards[i] in [0, 1]`.
5. The final Solver step's answer is checked against the ground truth, returning a binary `outcome_reward in {0, 1}`.
6. GRPO is computed on a weighted advantage `r_w * A_router + s_w * A_steps + o_w * A_outcome`, with the router weight decaying 5% per epoch and the outcome weight increasing correspondingly.

Judge calls are batched 10 items per API call. The full training pass costs around $10 of judge time vs ~$75 without batching.

## Why this design

V1 (`../router_solver/`) failed because outcome-only credit assignment over a multi-step Python-tool pipeline produced gradient conflict. V2 attacks that directly by adding a dense per-step reward signal and switching the Solver from code generation to text reasoning. The judge is a stronger model than the Solver so the per-step labels are at least directionally correct.

What still does not work, with quantification, is in `../notes/2026-05-09_peter_pov_answer_target_vs_core_reasoning.md`: 51.7% of failures on the traced 20Q x G=6 sample are non-core (plan parse, last-step-equals-final-answer, intermediate copied as final), not core reasoning failures. Section 5.2 of the report walks through this and the ablations.

## How to run

```bash
cd router_solver_v2

# Train + eval on the slim probe-derived subset.
python main.py --mode train --dataset slim

# Eval-only against a saved checkpoint.
python main.py --mode eval --checkpoint runs/<run_name>/checkpoint_epoch0_q50.pt

# Resume from checkpoint.
python main.py --mode train --checkpoint runs/<run_name>/checkpoint_epoch0_q50.pt

# Ablation suites used in the report (see notes/2026-05-10_peter_ablation_report.md).
bash ablation_suite/run_answer_target_suite.sh
bash ablation_suite/run_parser_split_10q.sh
bash ablation_suite/run_credit_assignment_10q.sh
```

The remote judge is configured through env vars (`OLLAMA_API_BASE`, `OPENAI_API_KEY` for the OpenAI-compatible endpoint). The deployment assets for the GCP-hosted judge VM are in `judge_ops/`.

## Important files

| Path | Purpose |
|---|---|
| `main.py` | CLI entrypoint, `--mode train / eval / train_eval`, ablation toggles |
| `src/agents/agent.py` | Router and Solver LoRA agents, plan-parse repair, answer-synthesis path, repeated-prompt fast path |
| `src/rewards/judge.py` | batched GPT-4o-mini judge client |
| `src/rewards/shaper.py` | reward computation, weight scheduler (router decay + outcome ramp) |
| `src/training/train.py` | GRPO loop with the three-term weighted advantage |
| `src/training/eval.py` | greedy eval with optional answer-synthesis |
| `src/training/taxonomy.py` | per-rollout failure-mode classifier used to build the taxonomy in the report |
| `src/utils/config.py` | the configuration defaults referenced from `main.py` |
| `ablation_suite/` | shell scripts for the answer-target, parser-split, credit-assignment, and robust-matrix ablations |
| `judge_ops/` | judge VM deployment manifests, bootstrap scripts, vLLM systemd unit |
| `experiments/` | per-ablation rollout traces and taxonomy reports |

## Notes and caveats

- The reference Phase-4 config is roughly `G=4`, ~30000 rollouts over 7500 questions for one epoch, with the reward weights `router=0.3 -> 0.0`, `solver=0.5`, `outcome=0.2 -> 0.5`. End-to-end training is around 40 hours on a single A100.
- `--dataset slim` mirrors the original slim-dataset selection rule from V1 (`../router_solver/slim_dataset_provenance.md` documented this before that file was removed; the rule lives in `src/utils/config.py` now).
- The judge runs on a separate VM. If `OLLAMA_API_BASE` is unreachable the rollout phase fails fast. `judge_ops/scripts/smoke_test_remote_judge.sh` is the canonical health check.
- Two interventions tested in the report did not generalize: router prompt hardening (20% -> 3.3% catastrophic regression) and `outcome_credit_all_steps` (improved train, hurt held-out eval). The intervention that did help is the inference-side answer-synthesis re-prompt, enabled with `--use-answer-synthesis`.
- Some ablation outputs in `experiments/` are quite large (rollout traces JSONL). When publishing, consider gzipping them; we kept them uncompressed for ease of `grep`.
