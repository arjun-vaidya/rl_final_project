# Issue 03: Zero-Shot vs Published 4-Shot Benchmarks

## Why it matters
Published Qwen2.5-1.5B-Instruct GSM8K accuracy (~73%) uses **4-shot prompting** with worked examples. Our baseline runs 0-shot with a generic "solve this problem" instruction. The 60+ percentage point gap to the published number isn't real model weakness — it's a prompting mismatch.

## What's happening
`baseline/eval_all_baselines.py` builds the prompt as:
```python
prompt = f"Solve this math problem step by step:\n\n{question}\n\nAnswer:"
```

No worked examples, no format demonstration, no `\boxed{}` template. The model has to guess both the reasoning style and the output format on its own.

## Example
0-shot: model produces free-form text, often without a clean numeric ending.
4-shot: model is primed by 4 example Q/A pairs showing the exact expected format → consistently emits a single number.

## Suggested fix
For an apples-to-apples baseline:
1. Use the 4-shot prompt from the GSM8K paper (or Qwen's eval harness)
2. Or use the model's official chat template via `tokenizer.apply_chat_template`
3. For our trained model: skip few-shot since RL training takes the place of in-context examples

A more honest comparison: report both 0-shot (current 12%) AND 4-shot (~73%) baselines.
