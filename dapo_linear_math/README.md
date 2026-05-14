# dapo_linear_math

DAPO-style training on Qwen2.5-Math-1.5B-Instruct. Improves over vanilla GRPO by filtering uninformative groups and increasing LoRA rank.

## Changes from vanilla GRPO

- **Sampling**: Drop all-correct / all-wrong groups, refill batch (DAPO)
- **LoRA**: Rank 32 on all linear layers (~25M params, vs 1.1M)
- **KL**: 0.015 (vs 0.04)
- **Eval**: Self-consistency K=8 majority vote (vs greedy)
- **Bucket**: mixed_hard only (vs all)

Result: 88.5% majority-vote (vs 88.2% base). Within noise.

## Running

Without vLLM (HF generation):
```bash
python main.py --mode train_eval --bucket mixed_hard --rollouts_per_q 8
```

With vLLM (5-10x faster):
```bash
python main.py --mode train_eval --bucket mixed_hard --rollouts_per_q 8 --use_vllm
```

Time: ~3-4 hours (train + eval) without vLLM, ~1 hour with vLLM on L4.

## Special considerations

- **vLLM memory**: Use `--vllm_gpu_memory_utilization 0.45` on L4. Adjust if OOM.
- **DAPO knobs**:
  - `--dapo_groups_per_step`: Target informative groups per step (default: 2)
  - `--dapo_max_resamples`: Max resamples before giving up per step (default: 8)
- **Eval variance**: With 1319 test examples, ±0.9pp variance at 95% CI.
