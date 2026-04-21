# src/rewards/solver.py
from src.env.python_tool import ToolResult, looks_sensible

def solver_step_reward(tool_result: ToolResult, final_outcome: float) -> float:
    """
    Decomposed per-step Solver reward.
    Weights: 0.3 (no error) + 0.2 (sensible output) + 0.5 (final outcome)
    """
    if tool_result.is_error:
        return 0.0
        
    r = 0.0
    r += 0.3  # Tool executed cleanly
    
    if looks_sensible(tool_result.output):
        r += 0.2  # Output is non-empty and reasonably sized
        
    if final_outcome == 1.0:
        r += 0.5  # Final answer was correct
        
    return r
