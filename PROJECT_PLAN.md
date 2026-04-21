# Long-context Reasoning for Agentic LLMs — Project Plan

**Team:** Peter Vail Driscoll, Arjun Vaidya
**Course:** ORCSE6529 Advanced Reinforcement Learning (Spring 2026)

## Key Dates (today: 2026-04-20)

| Date | Deliverable | Days Left |
|---|---|---|
| May 3 | Upload presentation slides | 13 |
| May 4–6 | Deliver 10-min presentation | 14–16 |
| May 15 | 4–6 page report + code | 25 |

> **Implication:** pick a direction where an MVP runs end-to-end within ~10 days, leaving buffer for baselines, ablations, and writing.

---

## 1. Pitch Recap (from slides)

Three pillars from the initial pitch:

1. **RL for Reasoning (training time)** — hierarchical Router / Solver decomposition. Router proposes subgoals & tool plans; Solver executes. Motivated by the gradient-conflict problem: outcome-only rewards penalize good intermediate calls when the final answer is wrong.
2. **Graph-based Memory (inference time)** — external KG + internal working memory for multi-turn state/action sequences. Referenced KG-Agent template.
3. **Unified train + inference optimization** — RL shapes the general policy; graph memory adapts per workflow.

Cited work: Tool-R0 / Tool Zero (self-play, pure-RL tool use), KG-Agent, CoT self-consistency, RLVR-style verifiable rewards.

---

## 2. The Space of Paths We Could Explore

Grouped by which pillar they lean on. Not all are feasible in the remaining timeline — this is the menu.

### 2A. RL-for-reasoning directions

- **A1. Process-reward vs outcome-only RL on tool-use.** Train a small LLM with GRPO/PPO on a tool benchmark. Compare (i) outcome-only reward, (ii) tool-correctness shaping, (iii) step-level verifier reward. Measure gradient-conflict empirically (variance of advantages across sub-trajectories).
- **A2. Hierarchical Router–Solver with separate policies.** Two policies sharing a base model (LoRA adapters). Router action space = {subgoal, tool}; Solver action space = token-level generation of a tool call. Train jointly with credit assignment from outcome + per-subgoal verifier.
- **A3. Self-play / pure-RL tool acquisition (Tool-Zero-style).** No SFT cold start; start from instruction-tuned base and learn tool schemas from reward alone. Ambitious.
- **A4. RLVR curriculum.** Start with single-tool tasks, graduate to multi-tool, then multi-hop. Study whether the curriculum reduces gradient interference vs flat training.
- **A5. Offline RL on agent trajectories.** Collect trajectories from GPT-4-class teacher, apply DPO/KTO or offline Q-learning to a small student. Compare vs SFT on same trajectories.

### 2B. Graph-memory directions

- **B1. Graph memory as retrieval augmentation.** Build a triple-store KG from prior sessions; flat RAG vs graph-walk retrieval on a multi-hop QA or long-conversation benchmark (LoCoMo, MuSiQue, HotpotQA).
- **B2. RL-trained graph queries.** Treat Cypher-style queries over the KG as actions; RL-train the retriever to maximize downstream answer reward. Closest to the pitch's "graph retrieval as an RL action" framing.
- **B3. Graph memory for inference-time reflection.** After each episode, the agent writes a structured reflection (nodes/edges) into the KG; on the next episode it retrieves. Measure compounding improvement across repeated workflows.
- **B4. Working-memory + external-graph hybrid (KG-Agent replication).** Replicate a small version of the KG-Agent architecture, then ablate the internal vs external memory components.

### 2C. Evaluation / analysis directions

- **C1. Gradient-conflict diagnostic.** A short empirical/theoretical study of where outcome-only rewards fail in multi-turn tool use. Could stand alone as a "theory-lite" project.
- **C2. Memory scaling study.** How does answer quality degrade as the number of prior sessions grows, for flat context vs summary vs graph memory?
- **C3. Benchmark audit.** Run a fixed base model with several agent scaffolds (ReAct, Reflexion, tree search, graph-memory) on the same benchmark; isolate what actually helps.

### 2D. Candidate environments / benchmarks

| Env | Long-context? | Tool use? | Graph natural? | Cost |
|---|---|---|---|---|
| **GSM8K + calculator/python** | No | Yes (1 tool) | No | Cheap, fast iteration |
| **HotpotQA / MuSiQue** | Medium | Retrieval | Strong | Cheap |
| **LoCoMo** (long conversations) | Yes | No | Strong | Cheap |
| **BFCL / ToolBench** | Medium | Yes (many tools) | Weak | Medium |
| **AppWorld / WebArena** | Yes | Yes | Weak | Expensive, complex setup |
| **ALFWorld / ScienceWorld** | Medium | Yes | Medium | Cheap |
| **Custom synthetic multi-hop-tool env** | Tunable | Tunable | Tunable | You build it |

### 2E. Candidate algorithms / base models

- **Algorithms:** GRPO (DeepSeek-R1 style — no critic, group-relative), PPO, REINFORCE with baseline, DPO (offline), ReMax. GRPO is the current default for RLVR on reasoning tasks.
- **Base models:** Qwen2.5-1.5B / 3B / 7B-Instruct, Llama-3.2-1B / 3B, SmolLM2-1.7B. All runnable with LoRA on a single A100 / 4090. Qwen2.5 is the current RLVR go-to.
- **Infra:** `trl` (HF) or `verl` for GRPO; `unsloth` for fast LoRA; `networkx` or a dict-of-triples for the graph; `sentence-transformers` for embeddings.

---

## 3. Three Starting Directions (ranked by feasibility)

> Each is self-contained, has a clear baseline, and produces the result types the rubric wants (experiments + interpretation + future-work framing). Pick one as primary; any can be promoted to the full project.

### ⭐ Option 1 — "Process Rewards Beat Outcome Rewards on Tool-Use" (safest, most polished)

**Pitch tie-in:** directly attacks the gradient-conflict problem from slide 3.

**What we build:**
- A small tool-use environment: GSM8K / MATH extended with a Python-exec tool, or a toy multi-tool env (calculator + unit-converter + lookup).
- Three training runs on the same base (Qwen2.5-1.5B-Instruct + LoRA, GRPO):
  1. Outcome-only reward (0/1 on final answer).
  2. Outcome + tool-correctness shaping (did each tool call type-check / return a sensible result?).
  3. Outcome + step-level verifier (LLM-as-judge or rule-based on intermediate reasoning).
- Compare final accuracy, sample efficiency, and a **gradient-conflict metric** (e.g., variance of per-token advantages in correct-vs-incorrect trajectories, or a correlation between sub-trajectory quality and final reward).

**Why it's feasible:** no graph component, single env, ~1 week of training runs, all on existing infra (`trl` GRPO example is ~200 LOC).

**Risks:** not visibly novel — process rewards are a known idea. Mitigate by framing around the *gradient-conflict diagnostic* rather than "process rewards work."

**Deliverables for the report:**
- Figure: learning curves for the three reward schemes.
- Figure: gradient-conflict metric over training.
- Table: final accuracy, tool-call validity rate, avg steps/episode.
- Ablation: reward weight sweep.

**~Week plan:**
- W1 (Apr 21–27): env + GRPO pipeline working end-to-end on outcome-only. One training run completes.
- W2 (Apr 28–May 3): add shaping + verifier rewards, run comparisons, build slides.
- W3 (May 4–15): presentation, final runs, write report.

---

### Option 2 — "Graph Memory + RL Retriever for Multi-Hop QA" (medium scope, most novel-feeling)

**Pitch tie-in:** directly instantiates pillars 2 and 3 (graph memory + inference-time self-improvement).

**What we build:**
- Benchmark: **MuSiQue** or **HotpotQA** (2–4 hop questions, clear ground truth).
- Memory store: build an entity-triple graph from the passage corpus ahead of time (off-the-shelf with `rebel` or OpenIE, or just use the provided paragraph structure).
- Two retrievers on the same frozen base LLM (Qwen2.5-3B-Instruct):
  1. **Baseline:** flat dense retrieval (sentence-transformers) → answer.
  2. **Ours:** policy that emits a sequence of graph actions (`expand(entity)`, `follow(relation)`, `answer`) — trained with RL (GRPO) on outcome reward = answer match.
- Optional ablation: replace RL with a heuristic graph walk (BFS from entities in the question).

**Why it's feasible:** MuSiQue has a dev set; the graph can be small (per-question subgraph); the policy can be a small LoRA on top of a frozen base.

**Risks:** getting RL to work on variable-length graph traversal is fiddly. Mitigate by starting with a hard cap of N hops and a small action vocab.

**Deliverables:**
- Figure: hop-distribution accuracy (RL graph vs flat RAG vs heuristic graph).
- Figure: learning curve of the graph policy.
- Qualitative: 2-3 example trajectories showing graph walks vs flat retrieval.
- Ablation: policy trained with vs without outcome shaping; shared-vs-separate LoRA.

**~Week plan:**
- W1: build the graph + heuristic BFS baseline + flat RAG baseline. End with numbers for both.
- W2: RL policy training. At least one run converged.
- W3: slides, ablations, report.

---

### Option 3 — "Router–Solver Agent with Graph Reflection" (most ambitious, closest to pitch)

**Pitch tie-in:** the full pitch — hierarchical RL + graph memory + unified train/inference loop.

**What we build:**
- Environment: **ALFWorld** (text-based, cheap) or a custom multi-turn tool env. Needs repeated-workflow structure so memory matters.
- Router (small LoRA): outputs subgoal + tool plan.
- Solver (small LoRA, same base): executes tool calls for one subgoal.
- Graph memory: after each episode, a writer agent extracts `(state, action, outcome)` triples into the KG; Router conditions on retrieved subgraphs for new episodes.
- Training: GRPO on outcome reward, with per-subgoal shaping from the Router's own subgoal completion check.
- Claim: on repeated workflows, RL-trained agent + graph memory > RL-trained agent alone > SFT + graph memory > SFT alone.

**Why it's ambitious:** three moving parts (hierarchy, RL, memory). Any one of them bugging out burns a week.

**Risks:** high. The failure mode is "nothing converges and we have no baseline story." Strongly mitigate by building baselines *first* — the vanilla ReAct + flat memory baseline has to be running by end of W1.

**Deliverables:**
- 2x2 ablation (RL vs SFT) × (graph memory vs none).
- Figure: per-episode reward *within* a repeated workflow (does memory compound?).
- Case study: one workflow where memory clearly helps.

**~Week plan (aggressive):**
- W1: ALFWorld + ReAct baseline; graph memory writer/reader. Numbers for SFT + no-memory.
- W2: Router-Solver LoRA + GRPO; one end-to-end RL run.
- W3: ablations, write up; accept that some cells of the 2x2 may be weak.

---

## 4. Recommendation

**Start with Option 1 this week.** It's the lowest-risk path to a defensible submission and builds the GRPO/verifier infra that Option 2 and 3 both need. If it's working cleanly by end of W1, we can promote to Option 2 (add graph memory) or Option 3 (add Router-Solver) without throwing away any code. If it's shaky, we still have a complete, honest story to present.

**Immediate next steps (today/tomorrow):**
1. Pick base model + env (propose: Qwen2.5-1.5B-Instruct + GSM8K-with-python-tool).
2. Stand up `trl` GRPO example on the chosen env with a dummy reward → confirm it trains.
3. Define the three reward functions precisely.
4. Decide compute: local GPU? Colab? Lambda/Runpod?

## 5. Open Questions to Resolve with Peter

- Do we have GPU budget? (drives model size: 1.5B vs 7B)
- Is the intended novelty (a) the gradient-conflict framing, (b) graph memory as RL actions, or (c) the hierarchy? Pick one to lead with in the report.
- Are we OK with a "negative result" direction (e.g., "process rewards didn't help because X")? Per the rubric, this is acceptable and often scores well if the analysis is sharp.
