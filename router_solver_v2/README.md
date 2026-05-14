# router_solver_v2

Hierarchical Router-Solver V2: text Solver, GPT-4o-mini judge for step-level rewards. Result: ~35% rollout accuracy. Kept for reference; see `router_solver_hierarchical_pivot/` for V3.

## Architecture

1. **Router** (Qwen + LoRA): emits JSON plan with subgoals
2. **Judge** (GPT-4o-mini): scores plan (0-1) and each step (0-1)
3. **Solver** (Qwen + LoRA): executes steps sequentially, text reasoning
4. Outcome: binary if final answer correct
5. **GRPO**: weighted advantage: `r_w * A_router + s_w * A_steps + o_w * A_outcome`, weights decay over epochs

Judge calls batched 10/request → ~$10 total vs ~$75 unbatched.

## Why it failed

V1's outcome-only signal led to gradient conflict. V2 adds dense step rewards. But 51.7% of failures (from traced 20Q x G=6 sample) are architectural, not reasoning:
- plan parse failures (13.3%)
- copied intermediate as final (15.8%)
- answer target mismatch (10.0%)
- other (12.7%)

Only 37.5% are actual wrong-answer failures.

Verdict: interface design, not reasoning capability, is the bottleneck.

## Running

```bash
python main.py --mode train --dataset slim      # Train on small subset
python main.py --mode eval --checkpoint <path>  # Eval from checkpoint
```

Judge VM setup in `judge_ops/`. Requires OpenAI API key.

## Special considerations

- Judge reliability: GPT-4o-mini labels directional but not ground-truth
- Architecture failures dominate: plan parse, final-answer contract
- See `router_solver_hierarchical_pivot/` for V3 (simplified contracts)
