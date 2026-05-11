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
Drop the hierarchy. Use single-pass CoT directly (implemented in `linear_reasoning/`). The base model already does chain-of-thought; forcing it through plan + step + extract just adds three points where things can break.
