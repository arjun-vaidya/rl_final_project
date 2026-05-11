# Issue 04: Router-Solver Hierarchy Introduces Failures

## Why it matters
Peter's failure taxonomy on Router-Solver V2 traces showed that **51.7% of failures are architectural**, not core reasoning. The hierarchy itself is adding error modes the base model wouldn't have made on its own.

## What's happening
From the 20-question taxonomy (`router_solver_v2/experiments/taxonomy_q20_20260509_222707/`):

| Failure | Share | Cause |
|---------|-------|-------|
| `wrong_numeric_final` | 37.5% | Core reasoning |
| `copied_intermediate_as_final` | 15.8% | Last-step extraction grabs an intermediate value |
| `plan_parse_failed` | 13.3% | Router output not valid JSON |
| `correct_number_in_trace_wrong_final` | 12.5% | Right answer was in the reasoning, but emitted scalar is wrong |
| `plan_endpoint_mismatch` | 10.0% | Last subgoal is not the actual answer target |

Each Router → Solver → final-step-extraction handoff is an independent failure interface.

## Example
- Router plans: `["Calculate eggs laid", "Subtract eggs eaten", "Convert to dollars"]`
- Solver does each step correctly, last step output: `"$18"`
- System emits "18" — correct
- But on another question, Router plans: `["Calculate cost", "Find total"]` where "Find total" was supposed to be the answer
- Solver's "Find total" step outputs `"The total is 18, so the answer is 18 dollars"` → extractor grabs the FIRST 18 (intermediate, wrong)

## Suggested fix
**Idea 1: Drop the Hierarchy (Flat CoT)**
Drop the hierarchy. Use single-pass CoT directly (implemented in `linear_reasoning/`). The base model already does chain-of-thought; forcing it through plan + step + extract just adds three points where things can break.

**Idea 2: Prompt-Based Dynamic Routing (Compute-Penalized Routing)**
If we want to retain the Router paradigm to save inference costs without the fragile subgoal handoffs, we can implement a compute-efficient routing strategy. The Router predicts problem difficulty and routes to one of two system prompts fed into the exact same model:
* **Option A (Fast):** System Prompt: *"Answer immediately without reasoning."* (Max tokens: 50)
* **Option B (Slow):** System Prompt: *"Let's think step by step rigorously."* (Max tokens: 512)

By adding a token-length penalty to the GRPO reward function, the Router learns to optimize the trade-off between accuracy and compute, providing a NeurIPS-level architectural novelty without the parsing errors of the original implementation.

**Idea 3: Strict Constrained Decoding (Infrastructure Fix)**
Fix the JSON and extraction failures at the infrastructure level, rather than changing the RL paradigm. Use a constrained decoding library like `outlines` or `guidance` (or vLLM's guided JSON decoding). This physically forces the model at the logits level to only output valid JSON plans or perfectly formatted `\boxed{}` final answers. It makes it mathematically impossible for the model to generate a syntax error, completely eliminating the 13.3% `plan_parse_failed` bucket.

**Idea 4: The "Pointer-Network" Extractor (Architectural Fix)**
To address the 15.8% of failures that are `copied_intermediate_as_final` (where the Python script grabs the wrong number from the text), stop using Python regex or string splitting to extract the answer. Instead, add a tiny Pointer Head (a single linear layer) on top of the Solver's neural network. During training, this head learns to point to the exact token index in its own output that represents the "final answer." You replace brittle, hand-coded heuristics with a learned extraction mechanism, creating an end-to-end differentiable system. This completely eliminates intermediate-copying errors without relying on strict formatting constraints, making for a very strong, NeurIPS-level architectural claim.
