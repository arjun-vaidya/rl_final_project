# Issue 07: Model Continues Generating After the Answer

## Why it matters
Even when the model produces the correct answer, it sometimes keeps generating — continuing onto unrelated problems, repeating itself, or adding spurious "examples." Any text after the real answer corrupts extraction and KL signal during training.

## What's happening
With no proper stop token configured, generation runs until `max_new_tokens`. The instruction-tuned model may signal end-of-turn with `<|im_end|>` (token id 151645), but if we set `pad_token_id = eos_token_id = <|endoftext|>` (151643), the model never naturally terminates and just keeps producing.

## Example
From `baseline_results_all.json` (math model):
```
GT: 3
Output: "3 bolts

Now, let's solve a more complex problem:

A box of blue fiber takes 2 bolts. A box of white fiber takes 1 bolt..."
```
The model gave the right answer ("3 bolts") immediately, then started inventing a *new problem*. Our last-number extractor picked up "1" from the invented problem.

## Suggested fix
1. Configure `eos_token_id` to include the chat-template end token (`<|im_end|>` for Qwen)
2. Use `tokenizer.apply_chat_template(..., add_generation_prompt=True)` so the model sees the proper turn structure
3. In our agent: pass `eos_token_id` as a list of both `<|endoftext|>` and `<|im_end|>` to `generate()`
4. The `trim_at_eos` defensive fix in `linear_reasoning/src/agent.py` handles this for the trained model
