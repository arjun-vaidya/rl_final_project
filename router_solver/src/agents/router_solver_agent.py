from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch

from src.utils.prompts import build_router_prompt, build_solver_prompt
from src.utils.parsing import parse_plan_json, extract_code_block
from src.env.python_tool import run_python, ToolResult
from src.env.code_batcher import CodeBatcher
from src.utils.config import GlobalConfig
from src.rewards.router import RouterReward, HeuristicRouterReward


@dataclass
class SolverStepRecord:
    subgoal: str
    prompt_ids: torch.Tensor           # [L_p] — the solver prompt as tokenized
    completion_ids: torch.Tensor       # [L_c] — the solver's generated tokens
    output: str                        # decoded completion text
    tool_result: ToolResult


@dataclass
class Rollout:
    question: str
    router_prompt_ids: torch.Tensor
    router_completion_ids: torch.Tensor
    router_output: str
    plan_dict: Optional[Dict]
    steps: List[SolverStepRecord] = field(default_factory=list)
    final_answer: Optional[str] = None
    tool_error_count: int = 0


class RouterSolverAgent:
    """
    Hierarchical agent with a Router and a Solver.
    Uses two LoRA adapters on a shared base model.
    """
    def __init__(
        self,
        model,
        tokenizer,
        config: GlobalConfig,
        device: str = "cuda",
        router_reward_evaluator: Optional[RouterReward] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        self.router_adapter = config.model.router_adapter_name
        self.solver_adapter = config.model.solver_adapter_name

        # Router reward evaluator: defaults to heuristic, can be swapped for LLM-based
        if router_reward_evaluator is None:
            self.router_reward_evaluator = HeuristicRouterReward()
        else:
            self.router_reward_evaluator = router_reward_evaluator

    def _set_adapter(self, adapter_name: str):
        """Swaps the active LoRA adapter. No-op for non-peft models (tests)."""
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter(adapter_name)

    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Generate from the current adapter. Returns (text, prompt_ids, completion_ids)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_ids = inputs.input_ids[0]  # [L_p]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        full_ids = outputs[0]
        completion_ids = full_ids[prompt_ids.size(0):]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return text, prompt_ids.detach(), completion_ids.detach()

    def batch_rollouts(
        self,
        questions: List[str],
        num_rollouts: int = 1,
        memory=None,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> List[Rollout]:
        """
        Generate multiple rollouts for multiple questions in parallel.
        Returns B*G rollouts (B questions, G rollouts per question).
        """
        all_rollouts = []

        # 1. Batch router inference for all questions
        self._set_adapter(self.router_adapter)
        router_results = []  # List of (text, prompt_ids, completion_ids)

        for question in questions:
            past_memory_entries = []
            if memory is not None:
                past_memory_entries = memory.retrieve(question, k=self.config.memory.k)

            router_prompt = build_router_prompt(question, past_memory_entries)
            router_text, router_prompt_ids, router_completion_ids = self._generate(
                router_prompt,
                max_new_tokens=self.config.rollout.router_max_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            router_results.append((question, router_text, router_prompt_ids, router_completion_ids))

        # 2. Prepare all solver steps and batch code execution
        self._set_adapter(self.solver_adapter)
        code_batcher = CodeBatcher(num_workers=4, timeout=5.0)

        # Structure: (q_idx, rollout_idx, step_idx, subgoal, solver_text, prompt_ids, comp_ids, code, plan_dict)
        all_step_records = []
        code_task_map = {}  # task_id -> (q_idx, rollout_idx, step_idx)

        for q_idx, (question, router_text, router_prompt_ids, router_completion_ids) in enumerate(router_results):
            plan_dict = parse_plan_json(router_text)
            if not plan_dict:
                # Invalid plan - skip solver steps
                for rollout_idx in range(num_rollouts):
                    all_rollouts.append(Rollout(
                        question=question,
                        router_prompt_ids=router_prompt_ids,
                        router_completion_ids=router_completion_ids,
                        router_output=router_text,
                        plan_dict=None,
                        steps=[],
                        final_answer=None,
                        tool_error_count=0,
                    ))
                continue

            plan_steps = plan_dict["plan"][: self.config.rollout.max_subgoals]

            for rollout_idx in range(num_rollouts):
                scratchpad = ""
                for step_idx, step in enumerate(plan_steps):
                    subgoal = step.get("subgoal", "solve") if isinstance(step, dict) else str(step)
                    solver_prompt = build_solver_prompt(question, router_text, scratchpad, subgoal)
                    solver_text, s_prompt_ids, s_comp_ids = self._generate(
                        solver_prompt,
                        max_new_tokens=self.config.rollout.solver_max_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                    )

                    code = extract_code_block(solver_text)
                    if code:
                        task_id = code_batcher.queue_code(code)
                        code_task_map[task_id] = (q_idx, rollout_idx, step_idx)

                    all_step_records.append((q_idx, rollout_idx, step_idx, subgoal, solver_text, s_prompt_ids, s_comp_ids, code, plan_dict))
                    scratchpad += f"\n[Step {step_idx+1}] (placeholder)\n"

        # 3. Execute all code in one parallel batch
        code_results = code_batcher.execute_batch()

        # 4. Assemble rollouts
        rollout_data = {}  # (q_idx, rollout_idx) -> {steps, plan_dict, router_...}

        for q_idx, (question, router_text, router_prompt_ids, router_completion_ids) in enumerate(router_results):
            plan_dict = parse_plan_json(router_text)
            for rollout_idx in range(num_rollouts):
                rollout_data[(q_idx, rollout_idx)] = {
                    "question": question,
                    "router_text": router_text,
                    "router_prompt_ids": router_prompt_ids,
                    "router_completion_ids": router_completion_ids,
                    "plan_dict": plan_dict,
                    "steps": [],
                    "tool_error_count": 0,
                }

        # Build step records with code results
        for q_idx, rollout_idx, step_idx, subgoal, solver_text, s_prompt_ids, s_comp_ids, code, plan_dict in all_step_records:
            if code:
                task_id = None
                for tid, (qi, ri, si) in code_task_map.items():
                    if qi == q_idx and ri == rollout_idx and si == step_idx:
                        task_id = tid
                        break

                if task_id is not None and task_id in code_results:
                    tool_result = code_results[task_id]
                else:
                    tool_result = ToolResult("Code execution failed", True, 0.0)
            else:
                tool_result = ToolResult("no code block emitted", True, 0.0)

            if tool_result.is_error:
                rollout_data[(q_idx, rollout_idx)]["tool_error_count"] += 1

            rollout_data[(q_idx, rollout_idx)]["steps"].append(SolverStepRecord(
                subgoal=subgoal,
                prompt_ids=s_prompt_ids,
                completion_ids=s_comp_ids,
                output=solver_text,
                tool_result=tool_result,
            ))

        # Convert to Rollout objects
        for q_idx, rollout_idx in sorted(rollout_data.keys()):
            data = rollout_data[(q_idx, rollout_idx)]
            final = data["steps"][-1].tool_result.output if data["steps"] else None
            all_rollouts.append(Rollout(
                question=data["question"],
                router_prompt_ids=data["router_prompt_ids"],
                router_completion_ids=data["router_completion_ids"],
                router_output=data["router_text"],
                plan_dict=data["plan_dict"],
                steps=data["steps"],
                final_answer=final,
                tool_error_count=data["tool_error_count"],
            ))

        return all_rollouts

    def rollout(
        self,
        question: str,
        memory=None,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> Rollout:
        """
        Executes a full hierarchical rollout (Router → N × Solver+tool).
        Returns a Rollout with prompt/completion token tensors so the caller
        can recompute log-probs for GRPO.
        """
        # 1. Router creates plan
        self._set_adapter(self.router_adapter)
        past_memory_entries = []
        if memory is not None:
            past_memory_entries = memory.retrieve(question, k=self.config.memory.k)

        router_prompt = build_router_prompt(question, past_memory_entries)
        router_text, router_prompt_ids, router_completion_ids = self._generate(
            router_prompt,
            max_new_tokens=self.config.rollout.router_max_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        plan_dict = parse_plan_json(router_text)

        if not plan_dict:
            return Rollout(
                question=question,
                router_prompt_ids=router_prompt_ids,
                router_completion_ids=router_completion_ids,
                router_output=router_text,
                plan_dict=None,
                steps=[],
                final_answer=None,
                tool_error_count=0,
            )

        # 2. Solver executes subgoals with batch code execution
        self._set_adapter(self.solver_adapter)
        scratchpad = ""
        steps: List[SolverStepRecord] = []
        code_batcher = CodeBatcher(num_workers=4, timeout=5.0)

        # First pass: generate all solver steps and queue code
        step_records = []  # (step_idx, subgoal, solver_text, prompt_ids, comp_ids)
        code_to_step_idx = {}  # Maps code task_id to step index

        plan_steps = plan_dict["plan"][: self.config.rollout.max_subgoals]

        for step_idx, step in enumerate(plan_steps):
            subgoal = step.get("subgoal", "solve") if isinstance(step, dict) else str(step)
            solver_prompt = build_solver_prompt(question, router_text, scratchpad, subgoal)
            solver_text, s_prompt_ids, s_comp_ids = self._generate(
                solver_prompt,
                max_new_tokens=self.config.rollout.solver_max_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

            code = extract_code_block(solver_text)
            if code:
                # Queue code for batch execution
                task_id = code_batcher.queue_code(code)
                code_to_step_idx[task_id] = step_idx

            step_records.append((step_idx, subgoal, solver_text, s_prompt_ids, s_comp_ids, code))

        # Execute all code in parallel on CPU cores
        code_results = code_batcher.execute_batch()

        # Second pass: build step records with results
        tool_errs = 0
        for step_idx, subgoal, solver_text, s_prompt_ids, s_comp_ids, code in step_records:
            if code:
                # Look up result from batch execution
                task_id = None
                for tid, step_id in code_to_step_idx.items():
                    if step_id == step_idx:
                        task_id = tid
                        break

                if task_id is not None and task_id in code_results:
                    tool_result = code_results[task_id]
                else:
                    tool_result = ToolResult("Code execution failed", True, 0.0)
            else:
                # No code extracted → no tool was executed
                tool_result = ToolResult("no code block emitted", True, 0.0)

            if tool_result.is_error:
                tool_errs += 1

            scratchpad += f"\n[Step {step_idx+1}] Output: {tool_result.output}\n"
            steps.append(SolverStepRecord(
                subgoal=subgoal,
                prompt_ids=s_prompt_ids,
                completion_ids=s_comp_ids,
                output=solver_text,
                tool_result=tool_result,
            ))

        final = steps[-1].tool_result.output if steps else None
        return Rollout(
            question=question,
            router_prompt_ids=router_prompt_ids,
            router_completion_ids=router_completion_ids,
            router_output=router_text,
            plan_dict=plan_dict,
            steps=steps,
            final_answer=final,
            tool_error_count=tool_errs,
        )
