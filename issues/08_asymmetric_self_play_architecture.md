# Issue 08: Overcoming Dataset Limits and Judge Costs via Asymmetric Self-Play

## Why it matters
Our current training pipeline faces two existential bottlenecks:
1. **Dataset Starvation:** We are limited to the static GSM8K training set (7,473 questions), which is not enough diversity for complex RL policies to converge without overfitting.
2. **Judge Cost Cap:** As noted in Issue 05, evaluating complex hierarchical plans with an LLM-as-a-Judge (e.g., GPT-4o mini) is too expensive and slow, forcing us to train on a fraction (~0.7%) of even the limited data we have.

## Proposed Solution: Compute-Efficient Asymmetric Self-Play
We completely drop the static dataset and the LLM judge. Instead, we evolve the Router-Solver paradigm into a **Teacher-Student Co-Evolution Architecture** trained jointly via GRPO.

### How it Works
We split the system into two distinct roles (which can be two separate LoRA adapters on the same base model):

1. **The Teacher (Router):** 
   - **Task:** Prompted to generate a *novel* math word problem AND write a deterministic Python script that calculates its exact ground-truth numeric answer.
2. **The Student (Solver):**
   - **Task:** Prompted to read the Teacher's generated word problem and solve it using Chain-of-Thought reasoning.

### The Zero-Cost Execution Pipeline
To evaluate the Student, we do **not** use an LLM Judge. We run the Teacher's Python script through our existing `code_batcher.py` environment. 
- If the script crashes or throws an error, the Teacher receives a reward of `-1.0`. This forces the Teacher to generate logically sound, solvable mathematics.
- If the script succeeds, the output is used as the deterministic Ground-Truth Oracle to evaluate the Student's rollouts. **This reduces our judge cost to $0.**

### The "ZPD" (Zone of Proximal Development) Reward Metric
To prevent the Teacher from generating questions that are too easy (e.g., $1+1$) or impossibly hard (e.g., abstract algebra), we introduce an adversarial reward function targeting the Student's 50% accuracy boundary:

* **Student Reward:** Standard GRPO. +1.0 for matching the Ground-Truth Oracle, optimizing the Student to solve the problems.
* **Teacher Reward:** `1.0 - abs(Student_Accuracy - 0.5) * 2`
  * If the Student gets 100% (too easy), Teacher Reward = 0.
  * If the Student gets 0% (too hard), Teacher Reward = 0.
  * If the Student gets exactly 50% (the perfect learning boundary), Teacher Reward = +1.0.

### The NeurIPS Value Proposition
By implementing this, we eliminate the reliance on static datasets and costly API judges. We introduce a **Compute-Efficient, Oracle-Free Framework for Asymmetric Self-Play in LLMs**. The paper's core scientific contribution becomes demonstrating that a 1.5B parameter model can bootstrap its own mathematical capabilities from scratch via emergent, unsupervised curriculum learning.
