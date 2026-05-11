# Issue 02: Generation Cut Off Before Final Answer

## Why it matters
Models often produce verbose chain-of-thought reasoning that exceeds our token budget. If the answer comes at the end and we cut off mid-reasoning, the answer never appears in the output. This silently looks like a wrong answer.

## What's happening
`baseline/eval_all_baselines.py` uses `max_new_tokens=200`. For multi-step GSM8K problems, the general Qwen 1.5B reliably needs 300-500 tokens. The generation gets truncated mid-CoT and no final number is emitted at all.

## Example
From `baseline_results_all.json`:
```
GT: 18
Output: "Step-by-step reasoning process: To solve this problem, let's break it down into steps:
         Step 1: Calculate the total number of eggs laid per day.
         Jane"
```
Output stopped at 200 tokens mid-sentence. No final answer in the text → counted as wrong.

## Suggested fix
1. Increase `max_new_tokens` to 400-512 for baseline evaluation
2. Use proper stop tokens (`<|im_end|>` for Qwen instruct models) so the model stops naturally when done
3. In `linear_reasoning`, the agent already uses 400 tokens — bump baseline eval to match
