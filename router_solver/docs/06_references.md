# 06 · References and Prior Work

## Has this been done before?

**Yes, partly.** The broad idea — hierarchical planner/executor LLM agents trained with GRPO-style RL — has recent published work. What we own is the **clean, matched-compute comparison** isolating (a) architecture and (b) reward decomposition, plus an explicit **gradient-conflict diagnostic**.

## Most-related prior work

### Agent-as-Tool (Pan et al., 2025 · arXiv 2507.01489)
Closest to us. Splits an agent into a Planner and a Toolcaller, fine-tunes the Planner with GRPO, and masks Toolcaller outputs during credit assignment. Our differences: we train both components (not just the Planner), we compare against a flat GRPO baseline with matched compute (they compare against different agent scaffolds), and we include a gradient-conflict diagnostic.
→ https://arxiv.org/abs/2507.01489

### ArCHer (Zhou et al., ICML 2024 · arXiv 2402.19446)
Foundational hierarchical RL for LLM agents. High-level off-policy TD over utterances + low-level on-policy PG within each utterance. Reports ~100× sample-efficiency gains on multi-turn tasks up to 7B. Different algorithm (TD + PG, not GRPO) and different action abstraction (utterances, not plan/execute), but same core insight that credit assignment benefits from hierarchy.
→ https://arxiv.org/abs/2402.19446

### Tree-GRPO (Liu et al., ICLR 2026 · arXiv 2509.21240)
Tree-search rollouts in place of independent chain rollouts, with step-level nodes. Addresses credit assignment by structuring the rollout, not by structuring the policy. Complementary to our approach — could be combined with Router–Solver.
→ https://arxiv.org/abs/2509.21240

### GiGPO — Group-in-Group Policy Optimization (Feng et al., NeurIPS 2025)
Nests two levels of group-relative credit assignment. A methodological alternative to our architectural hierarchy: they do flat agents with hierarchical advantage estimation; we do hierarchical agents with flat advantage estimation. Worth discussing in related work.
→ https://personal.ntu.edu.sg/boan/papers/NeurIPS25_GiGPO.pdf

### AgentPRM (arXiv 2511.08325, 2025)
Process reward models for LLM agents, with step-wise promise and progress scoring. Closest process-reward analogue to our Solver shaped reward. They learn the PRM; we use a hand-coded rule-based proxy because (a) the experiment is cleaner and (b) we don't have PRM training data.
→ https://arxiv.org/abs/2511.08325

## Method building blocks

### GRPO (DeepSeek-AI, 2024)
Group Relative Policy Optimization. Policy-gradient variant that replaces the critic with group-relative advantages (mean/std within a group of rollouts on the same prompt). Introduced in DeepSeekMath and widely adopted for reasoning RL (DeepSeek-R1).
→ DeepSeekMath: https://arxiv.org/abs/2402.03300
→ DeepSeek-R1: https://arxiv.org/abs/2501.12948

### LoRA (Hu et al., ICLR 2022)
Low-rank adapters. Lets us attach two independent policies (Router, Solver) to one frozen base model cheaply.
→ https://arxiv.org/abs/2106.09685

### Let's Verify Step by Step (Lightman et al., ICLR 2024)
Foundational result that process supervision beats outcome supervision for math reasoning. Uses a trained PRM rather than rule-based rewards. Motivates the whole process-rewards line of work.
→ https://arxiv.org/abs/2305.20050

### Math-Shepherd (Wang et al., 2023)
Automatic process rewards via Monte Carlo estimation of step-wise correctness. Method for generating process labels without human annotation.
→ https://arxiv.org/abs/2312.08935

## Dataset / environment

### GSM8K (Cobbe et al., 2021)
Grade-school math word-problem benchmark. 7,473 train / 1,319 test.
→ https://arxiv.org/abs/2110.14168

### Qwen2.5 (Qwen team, 2024)
Base model family. `Qwen2.5-1.5B-Instruct` is our starting point.
→ https://arxiv.org/abs/2412.15115

## Framing and motivation

### Chain-of-Thought self-consistency (Wang et al., ICLR 2023)
Cited in the project's original pitch. Motivates multi-sample agent rollouts and the non-linear scaling of workflow complexity in $T$.
→ https://arxiv.org/abs/2203.11171

### Tool-R0 and Tool Zero
Referenced in the project pitch as evidence that pure-RL tool acquisition is viable. Show that self-play RL can induce complex tool-calling without curated SFT trajectories. (Add precise citations when writing the final report.)

## Our positioning

For the final report, the related-work section should:

1. **Acknowledge** that hierarchical planner/executor LLM agents exist (Agent-as-Tool, ArCHer).
2. **Distinguish** our contribution along three axes: matched-compute comparison, ablation isolating architecture vs reward decomposition, and the gradient-conflict diagnostic.
3. **Place** us in the process-reward literature (Math-Shepherd, AgentPRM): we use a simple rule-based proxy for process rewards, not a learned PRM, because the point is isolating the credit-assignment effect, not building the best possible reward model.

This is a course project, not a paper. The novelty bar is a clean, well-designed experiment with honest analysis — not a new method.
