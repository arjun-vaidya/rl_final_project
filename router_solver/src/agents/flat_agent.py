import re
from src.env.python_tool import run_python

class FlatAgent:
    """
    A flat CoT agent that uses Python tools. 
    It interleaves thought and code blocks.
    """
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def execute(self, question: str, max_tokens: int = 1024) -> str:
        """
        Runs the full agentic loop for a single question.
        Returns the full trajectory text.
        """
        # Starting prompt
        prompt = f"Question: {question}\nReasoning and tool use:"
        trajectory = prompt
        
        remaining_tokens = max_tokens
        
        # Simple loop: generate until </code> or <answer>
        # Note: In a real GRPO training setup, this loop needs to be very efficient (vLLM).
        # For the implementation, we'll follow the logical flow.
        
        while remaining_tokens > 0:
            # We look for <code> or <answer>
            # For simplicity, let's assume the model uses <code>...</code> blocks
            # and we stop whenever </code> is generated.
            
            # This is a placeholder for actual batched generation logic used in training.
            # In src/training/train_flat.py, we might use a more optimized version.
            
            # 1. Generate until </code> or <answer>
            # (In reality, we'd use stop strings)
            
            # For now, let's just implement the pattern described in the docs.
            # We will use this logic in the evaluate.py script.
            break # Implementation depends on whether we are in training or eval loop.

        return trajectory

# Actually, the task list says:
# "src/agents/flat_agent.py — flat CoT+tool agent (Condition #2)."
# In the context of GRPOTrainer, we might just need the prompting logic 
# and a custom rollout function.

def get_flat_prompt(question: str) -> str:
    return (
        "You are a math assistant. Solve the following problem step by step. "
        "Use <code>...</code> blocks for python code to perform calculations. "
        "Every code block will be executed and the output will be shown to you. "
        "End your answer with <answer>RESULT</answer>.\n\n"
        f"Question: {question}\n\n"
        "Reasoning:"
    )
