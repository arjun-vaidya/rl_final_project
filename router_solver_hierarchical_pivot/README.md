# router_solver_hierarchical_pivot (V3)

Three-branch hierarchical agent attempting to fix V2's architectural failures. **Ongoing work.**

## Branches

- **easy**: Single-pass CoT (copy of `../linear_reasoning/`)
- **soft** (production candidate): Hierarchical Solver, no synthesis, answer-bearing final step, majority vote. Result: ~90% question-majority on 10Q diagnostic with T=0.7.
- **hard**: Hierarchical Solver + Hopfield memory retrieval. **Still experimental.**

## Key idea

V2 failed on interface design (51.7% of failures architectural). Route each question to cheapest branch that solves it. Hard branch uses memory retrieval: condition on structurally similar solved problems instead of reasoning from scratch.

## Current state

- Soft branch stabilized after fixing answer-target contract
- Hard branch retrieval improved substantially, but downstream accuracy poor
- Bottleneck: corpus quality, not retriever architecture
- See notes/ for detailed plan

## Running

```bash
python main.py --mode train                       # Three-branch training
python main.py --mode eval --diagnostic-rollouts-per-q 6  # Fixed 10Q eval
python scripts/build_retrieval_corpus.py           # Build memory
python scripts/train_contrastive_retriever.py      # Train retriever
```

Eval contract: fixed 10Q with G=6 rollouts (do not change for comparisons).

## Special considerations

- Hard branch corpus-dependent
- Router selection heuristic simple (currently based on question characteristics)
- Judge same as V2 (GPT-4o-mini)
