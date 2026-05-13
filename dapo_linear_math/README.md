# dapo_linear_math

DAPO-style RL on top of the math-specialized base. This is the response to the negative result in `grpo_linear_math_version` (math base + vanilla GRPO produced zero RL gain).

## What is different vs `grpo_linear_math_version`

| | grpo_linear_math_version | dapo_linear_math |
|---|---|---|
| Sampling | every prompt contributes | groups with all-correct or all-wrong rollouts are dropped, batch is refilled |
| LoRA | rank 8 on `q,v` (~1.1M params) | rank 32 on `q,k,v,o,gate,up,down` (~25M params) |
| KL coef | 0.04 | 0.015 |
| Loss | per-trajectory sum, divided by N | per-token sum, divided by total step tokens |
| Bucket default | `all` (mixed + hard + trivial) | `mixed_hard` (only the buckets that produce mixed-outcome groups on the math base) |
| Eval | greedy, single rollout | self-consistency K=8 at T=0.6, majority vote |

## How to run

### Without vLLM (HF transformers only)

```bash
cd dapo_linear_math
python main.py \
  --mode train_eval \
  --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
  --bucket mixed_hard \
  --rollouts_per_q 8 \
  --dapo_groups_per_step 2 \
  --dapo_rollout_batch 2 \
  --dapo_max_resamples 8 \
  --lora_rank 32 \
  --kl_coef 0.015 \
  --sc_K 8 \
  --sc_temperature 0.6 \
  --output_dir runs/v1 \
  --wandb_run_name dapo_math_mixedhard_g8
```

Expected wall clock on a single A100:
- Train (~870 questions through dynamic sampling, G=8): ~2.5-4 hours
- Eval (K=8 self-consistency on the 1319-question test set): ~15-25 minutes

### With vLLM (5-10x faster rollouts)

```bash
pip install vllm   # if not already installed

python main.py \
  --mode train_eval \
  --base_model Qwen/Qwen2.5-Math-1.5B-Instruct \
  --bucket mixed_hard \
  --rollouts_per_q 8 \
  --dapo_groups_per_step 2 \
  --dapo_rollout_batch 2 \
  --lora_rank 32 \
  --kl_coef 0.015 \
  --sc_K 8 --sc_temperature 0.6 \
  --use_vllm \
  --vllm_gpu_memory_utilization 0.45 \
  --vllm_sync_every 1 \
  --output_dir runs/v1_vllm \
  --wandb_run_name dapo_math_mixedhard_g8_vllm
```

Expected wall clock on a single A100 with vLLM:
- Train: ~45-90 minutes
- Eval: ~3-5 minutes

#### vLLM memory tuning

Both vLLM and HF (the trainable LoRA + optimizer + activations) live on the same GPU. `--vllm_gpu_memory_utilization` controls how much VRAM vLLM may use for its weights and KV cache; the rest is for HF training. Starting points:

| GPU | Suggested `--vllm_gpu_memory_utilization` |
|---|---|
| A100 40GB | 0.40-0.45 |
| A100 80GB | 0.55-0.60 |
| H100 80GB | 0.55-0.60 |

If HF training OOMs, lower this value. If vLLM rollout latency is bad, raise it.

#### How the vLLM sync works

After every gradient step the PEFT adapter is saved to a versioned directory and registered as a fresh `LoRARequest` with vLLM. The next rollout call uses that LoRA. The save+register costs a few seconds per step; net wall-clock with G=8 is still 3-5x faster than HF generation. Use `--vllm_sync_every N` to sync only every N steps if you want to amortize the save cost (at the cost of slightly stale rollouts).

## DAPO knobs in plain language

- `dapo_groups_per_step` is the target number of *informative* groups (1 < n_correct < G) per gradient step. Bigger = more stable gradient, slower steps.
- `dapo_rollout_batch` is how many questions we draw per resample attempt. Set it equal to `dapo_groups_per_step` to start; bigger speeds things up if many groups are uninformative.
- `dapo_max_resamples` caps how many questions we are willing to burn per step before giving up and moving on.

## Notes

- The DAPO loop logs `kept_groups`, `dropped_all_correct`, and `dropped_all_wrong` to W&B so you can monitor the informative-group rate in real time. If `dropped_all_correct` dominates, the bucket is too easy. If `dropped_all_wrong` dominates, drop temperature or increase G.
- The self-consistency eval reports both majority-vote accuracy and pass@K. Pass@K is the upper bound that majority-vote can reach with a better aggregator.
