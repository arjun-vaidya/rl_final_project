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
        # We wrap the single question into batched_rollout
        return self.batched_rollout([question], memory, do_sample, temperature, batch_size=1)[0]

    @torch.no_grad()
    def _batched_generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 1.0,
        batch_size: int = 8,
        vllm_engine=None,
        lora_request=None,
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """Batched generation using vLLM if provided, else HF with left-padding."""
        if vllm_engine is not None:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=temperature if do_sample else 0.0,
                max_tokens=max_new_tokens,
            )
            # vLLM handles batching optimally, so we just submit everything
            outputs = vllm_engine.generate(
                prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
                lora_request=lora_request
            )
            all_texts = []
            all_prompt_ids = []
            all_comp_ids = []
            for out in outputs:
                all_texts.append(out.outputs[0].text)
                all_prompt_ids.append(torch.tensor(out.prompt_token_ids, device="cpu"))
                all_comp_ids.append(torch.tensor(out.outputs[0].token_ids, device="cpu"))
            return all_texts, all_prompt_ids, all_comp_ids

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        all_texts = []
        all_prompt_ids = []
        all_comp_ids = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            comp_slice = outputs[:, inputs.input_ids.size(1):]
            
            for j in range(len(batch_prompts)):
                mask = inputs.attention_mask[j] == 1
                p_ids = inputs.input_ids[j][mask]
                all_prompt_ids.append(p_ids.detach().cpu() if p_ids.device.type == "cuda" else p_ids.detach())
                
                c_ids = comp_slice[j]
                eos_idx = (c_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_idx) > 0:
                    c_ids = c_ids[:eos_idx[0] + 1]
                all_comp_ids.append(c_ids.detach().cpu() if c_ids.device.type == "cuda" else c_ids.detach())
                
                text = self.tokenizer.decode(c_ids, skip_special_tokens=True)
                all_texts.append(text)
                
        return all_texts, all_prompt_ids, all_comp_ids

    def batched_rollout(
        self,
        questions: List[str],
        memory=None,
        do_sample: bool = True,
        temperature: float = 1.0,
        batch_size: int = 8,
        vllm_engine=None,
        lora_base_path=None,
    ) -> List[Rollout]:
        """
        Executes a batched hierarchical rollout (Router → N × Solver+tool).
        Generates plans for all questions concurrently, then executes solvers concurrently
        layer by layer (step 1 for all, step 2 for all, etc.).
        """
        router_lora_req = None
        solver_lora_req = None
        if vllm_engine is not None and lora_base_path is not None:
            import os
            from vllm.lora.request import LoRARequest
            router_lora_req = LoRARequest(self.router_adapter, 1, os.path.join(lora_base_path, self.router_adapter, self.router_adapter))
            solver_lora_req = LoRARequest(self.solver_adapter, 2, os.path.join(lora_base_path, self.solver_adapter, self.solver_adapter))

        self._set_adapter(self.router_adapter)
        
        router_prompts = []
        for q in questions:
            past_mem = memory.retrieve(q, k=self.config.memory.k) if memory is not None else []
            router_prompts.append(build_router_prompt(q, past_mem))
            
        r_texts, r_p_ids, r_c_ids = self._batched_generate(
            router_prompts,
            max_new_tokens=self.config.rollout.router_max_tokens,
            do_sample=do_sample,
            temperature=temperature,
            batch_size=batch_size,
            vllm_engine=vllm_engine,
            lora_request=router_lora_req,
        )
        
        rollouts = []
        active_solvers = []
        
        for i, (q, r_text, p_ids, c_ids) in enumerate(zip(questions, r_texts, r_p_ids, r_c_ids)):
            plan_dict = parse_plan_json(r_text)
            ro = Rollout(
                question=q,
                router_prompt_ids=p_ids.to(self.device),
                router_completion_ids=c_ids.to(self.device),
                router_output=r_text,
                plan_dict=plan_dict,
                steps=[],
                tool_error_count=0
            )
            rollouts.append(ro)
            if plan_dict and "plan" in plan_dict:
                plan_steps = plan_dict["plan"][:self.config.rollout.max_subgoals]
                if plan_steps:
                    active_solvers.append({"idx": i, "steps": plan_steps, "scratchpad": "", "step_idx": 0})
                
        self._set_adapter(self.solver_adapter)
        
        while active_solvers:
            solver_prompts = []
            for solver in active_solvers:
                ro = rollouts[solver["idx"]]
                step = solver["steps"][solver["step_idx"]]
                subgoal = step.get("subgoal", "solve") if isinstance(step, dict) else str(step)
                prompt = build_solver_prompt(ro.question, ro.router_output, solver["scratchpad"], subgoal)
                solver_prompts.append(prompt)
                
            s_texts, s_p_ids, s_c_ids = self._batched_generate(
                solver_prompts,
                max_new_tokens=self.config.rollout.solver_max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                batch_size=batch_size,
                vllm_engine=vllm_engine,
                lora_request=solver_lora_req,
            )
            
            # Execute all tool calls concurrently via multiprocessing pool
            codes = [extract_code_block(s_texts[i]) for i in range(len(active_solvers))]
            tool_results = []
            from src.env.python_tool import _get_pool
            pool = _get_pool()
            futures = []
            for code in codes:
                if code:
                    futures.append(pool.apply_async(run_python, (code,)))
                else:
                    futures.append(None)
            for j, fut in enumerate(futures):
                if fut is not None:
                    try:
                        tool_results.append(fut.get(timeout=5.0))
                    except Exception:
                        tool_results.append(ToolResult("TimeoutExpired", True, 5000.0))
                else:
                    tool_results.append(ToolResult("no code block emitted", True, 0.0))
            
            next_active = []
            for i, solver in enumerate(active_solvers):
                text = s_texts[i]
                res = tool_results[i]
                    
                ro = rollouts[solver["idx"]]
                if res.is_error:
                    ro.tool_error_count += 1
                    
                solver["scratchpad"] += f"\n[Step {solver['step_idx']+1}] Output: {res.output}\n"
                
                step = solver["steps"][solver["step_idx"]]
                subgoal = step.get("subgoal", "solve") if isinstance(step, dict) else str(step)
                
                ro.steps.append(SolverStepRecord(
                    subgoal=subgoal,
                    prompt_ids=s_p_ids[i].to(self.device),
                    completion_ids=s_c_ids[i].to(self.device),
                    output=text,
                    tool_result=res
                ))
                
                solver["step_idx"] += 1
                if solver["step_idx"] < len(solver["steps"]):
                    next_active.append(solver)
                else:
                    ro.final_answer = res.output
                    
            active_solvers = next_active
            
        return rollouts
