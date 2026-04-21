from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch

from src.utils.prompts import build_router_prompt, build_solver_prompt
from src.utils.parsing import parse_plan_json, extract_code_block
from src.env.python_tool import run_python, ToolResult
from src.utils.config import GlobalConfig


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
    def __init__(self, model, tokenizer, config: GlobalConfig, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        self.router_adapter = config.model.router_adapter_name
        self.solver_adapter = config.model.solver_adapter_name

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

        # 2. Solver executes subgoals
        self._set_adapter(self.solver_adapter)
        scratchpad = ""
        steps: List[SolverStepRecord] = []
        tool_errs = 0

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
                tool_result = run_python(code)
            else:
                # No code extracted → no tool was executed. Mark as error so
                # solver_step_reward correctly assigns 0 (docs/04_design.md).
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
