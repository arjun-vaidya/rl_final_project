# grpo_linear_math_version

GRPO training on a 1,000-question subset of GSM8K-train, selected from probe results on Qwen2.5-Math-1.5B-Instruct (G=4 rollouts, temperature 0.8).

| Bucket  | n_correct (of 4) | Count | Why selected |
|---------|------------------|------:|---|
| mixed   | 1-3 / 4          |   583 | Within-group reward variance is non-zero, so GRPO produces a real policy gradient from step 0. This is the core learning signal. |
| hard    | 0 / 4            |   286 | Zero gradient at step 0 (all rollouts wrong), but flips to useful as the policy improves. Reserves room for the model to grow into harder problems. |
| trivial | 4 / 4            |   131 | Distribution anchor: keeps KL coverage on easy problems so the policy doesn't drift on questions it already solves. Sampled deterministically (seed=0) from 6,604 trivial questions. |
| **all** |                  | **1,000** | |
