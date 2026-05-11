# Issue 06: No Enforced Output Format

## Why it matters
Without a consistent output format, the model has no obligation to put its final answer anywhere extractable. We can't reliably grade outputs that vary between "The answer is 18", "= 18 dollars", "18", "We get \boxed{18}", or "Therefore, 18 is correct." Extraction heuristics break differently for each.

## What's happening
The Router-Solver and baseline scripts ask for "step-by-step reasoning" but give no format spec. Different models default to different conventions:

- **Qwen2.5-1.5B-Instruct (general)**: free-form prose, answer often buried mid-paragraph
- **Qwen2.5-Math-1.5B-Instruct**: trained to use `\boxed{}` — but if our extractor doesn't parse `\boxed{}`, we miss correct answers
- **DeepSeek-R1 style**: long reasoning then `\boxed{}` at the end

Three models, three formats, one fragile extractor.

## Example
Both models in the baseline eval produced *correct* answers in their natural format that our last-number extractor missed. Math-specialized scored 8% — strictly worse than general's 12% — because it actually used `\boxed{}` correctly and our parser ignored it.

## Suggested fix
1. Specify `\boxed{N}` in the system prompt (already done in `linear_reasoning/src/agent.py`)
2. Reward `\boxed{}` format with +0.5 (already in `linear_reasoning/src/reward.py`)
3. Parse `\boxed{}` first, fall back to last number only if missing
4. RL training pushes the model toward consistent use of the format across rollouts
