# Issue 05: Judge Cost Caps Training at ~50 Questions

## Why it matters
Router-Solver V2 uses a GPT-4o mini judge to score plans, steps, and outcomes. Each rollout costs API calls, which limited us to training on **50-120 questions** out of GSM8K's 7,473 available training problems. That's ~0.7% of the dataset. The model has nowhere near enough exposure to converge.

## What's happening
`router_solver_v2/src/rewards/judge.py` batches judge calls but still racks up cost roughly linearly with `train_questions × G × steps_per_rollout`. Even with batching:
- 50 questions × G=6 × ~3 steps each ≈ 900 step-judgments + 50 plan-judgments + 50 outcomes
- Scaling to 7,473 questions would cost ~$1,500+ per run

The infrastructure (remote vLLM judge, GCP load balancer) was built specifically because the cost was a binding constraint.

## Example
Best Router-Solver run (`outcome_heavy_50q_remotejudge_20260510_100507`):
- Training accuracy: 25.7% after 50 questions
- The accuracy curve was still climbing at the end — model is data-starved, not capacity-limited

## Suggested fix
Replace the judge entirely with a **verifiable reward**: GSM8K provides ground-truth numeric answers, so `reward = 1 if extracted_answer == ground_truth else 0` requires zero API calls. This is what `linear_reasoning/src/reward.py` does. Training on the full 7,473 questions becomes free (only GPU time matters).
