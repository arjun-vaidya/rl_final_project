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

    def rollout(self, question: str, memory=None, max_tokens: int = 1024) -> Dict:
        """
        Executes a full hierarchical rollout.
        Returns a trajectory log.
        """
        # 1. Router creates plan
        self._set_adapter(self.router_adapter)
        past_memory = ""
        if memory:
            past_memory = memory.retrieve(question)
        
        router_prompt = build_router_prompt(question, past_memory)
        # router_output = self._generate(router_prompt)
        router_output = '{"plan": [{"subgoal": "test", "tool": "python"}]}' # Placeholder
        
        plan_dict = parse_plan_json(router_output)
        if not plan_dict:
            return {"question": question, "error": "Invalid plan", "trajectory": router_output}

        # 2. Solver executes subgoals
        self._set_adapter(self.solver_adapter)
        scratchpad = ""
        trajectory_steps = []
        
        for step in plan_dict["plan"]:
            subgoal = step["subgoal"]
            solver_prompt = build_solver_prompt(question, router_output, scratchpad, subgoal)
            
            # solver_output = self._generate(solver_prompt)
            solver_output = "<code>4*2</code>" # Placeholder
            
            code = extract_code_block(solver_output)
            tool_result = ToolResult("8", False, 0.0)
            if code:
                tool_result = run_python(code)
            
            scratchpad += f"\nSubgoal: {subgoal}\nTool Output: {tool_result.output}\n"
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
        # This will be implemented with actual model.generate() in the training loop
        pass
