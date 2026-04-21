import torch
from typing import List, Dict, Optional
from src.utils.prompts import build_router_prompt, build_solver_prompt
from src.utils.parsing import parse_plan_json, extract_code_block
from src.env.python_tool import run_python, ToolResult
from src.utils.config import GlobalConfig

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
        """Swaps the active LoRA adapter."""
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter(adapter_name)
        else:
            # Dummy implementation for tests without peft
            pass

    def rollout(self, question: str, memory=None) -> Dict:
        """
        Executes a full hierarchical rollout.
        Returns a trajectory log.
        """
        # 1. Router creates plan
        self._set_adapter(self.router_adapter)
        past_memory = ""
        if memory:
            mem_results = memory.retrieve(question, k=self.config.memory.k)
            # Format memory results as past examples
            for i, p in enumerate(mem_results):
                past_memory += f"{i+1}. Problem: ... -> Plan: {p}\n"
        
        router_prompt = build_router_prompt(question, past_memory)
        router_output = self._generate(router_prompt, max_new_tokens=self.config.rollout.router_max_tokens)
        
        plan_dict = parse_plan_json(router_output)
        if not plan_dict:
            return {"question": question, "error": "Invalid plan", "trajectory": router_output}

        # 2. Solver executes subgoals
        self._set_adapter(self.solver_adapter)
        scratchpad = ""
        trajectory_steps = []
        
        # Limit steps to config
        plan_steps = plan_dict["plan"][:self.config.rollout.max_subgoals]
        
        for step_idx, step in enumerate(plan_steps):
            subgoal = step.get("subgoal", "solve")
            solver_prompt = build_solver_prompt(question, router_output, scratchpad, subgoal)
            
            solver_output = self._generate(solver_prompt, max_new_tokens=self.config.rollout.solver_max_tokens)
            
            code = extract_code_block(solver_output)
            tool_result = ToolResult("", False, 0.0)
            if code:
                tool_result = run_python(code)
            
            scratchpad += f"\n[Step {step_idx+1}] Output: {tool_res.output}\n"
            trajectory_steps.append({
                "subgoal": subgoal,
                "solver_output": solver_output,
                "tool_output": tool_result.output,
                "is_error": tool_result.is_error
            })

        return {
            "question": question,
            "router_output": router_output,
            "steps": trajectory_steps,
            "final_answer": trajectory_steps[-1]["tool_output"] if trajectory_steps else None
        }

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Helper to generate text from the current model+adapter."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
