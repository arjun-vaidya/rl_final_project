# Issue 01: Answer Extraction Grabs the Wrong Number

## Why it matters
Baseline accuracy in our 100Q eval came out at 12% (general) and 8% (math-specialized), versus published Qwen2.5-1.5B-Instruct benchmarks of ~73%. This is mostly an extraction bug, not a model failure.

## What's happening
`baseline/eval_all_baselines.py::extract_numeric_answer` returns the **last number** found in the generated text. This fails in three common patterns:

1. **Math model puts answer first, explains after** → last number is from explanation, not the answer
2. **General model rambles or gets cut off mid-CoT** → last number is from an intermediate calculation
3. **Model continues onto an unrelated problem** → last number is from a different question entirely

## Examples (from `baseline/baseline_results_all.json`)
| GT | Predicted output | Real answer | What our extractor picked |
|----|------------------|-------------|---------------------------|
| 3 | `"3 bolts\n\nNow, let's solve a more complex problem: A box of blue fiber takes 2 bolts..."` | 3 ✓ | wrong (picked 2) |
| 70000 | `"$120,000\n\nHere's the step-by-step reasoning: 1. Calculate the total cost..."` | 120,000 (still wrong, but model put it first) | wrong (picked some intermediate cost) |
| 18 | `"Step-by-step reasoning... Step 1: Calculate the total number of eggs..."` (truncated) | unknown — never finished | n/a |

## Suggested fix
1. Require `\boxed{N}` format with strong reward (already in `linear_reasoning/src/reward.py`)
2. Fall back to last number ONLY if `\boxed{}` is not present
3. Use stop tokens to prevent the model from continuing past the answer
