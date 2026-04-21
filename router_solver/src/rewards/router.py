# src/rewards/router.py
from src.utils.parsing import parse_plan_json
from src.rewards.outcome import outcome_reward

def router_reward(plan_output, trajectory, gt) -> float:
    """
    Rewards the router based on plan structure and final outcome.
    Gated by structural validity.
    """
    plan_dict = parse_plan_json(plan_output)
    if plan_dict is None:
        return 0.0
    
    plan_list = plan_dict.get("plan", [])
    if not isinstance(plan_list, list):
        return 0.0
        
    if not (1 <= len(plan_list) <= 8):
        return 0.0
        
    # Gated downstream credit: outcome of the trajectory that followed this plan
    return outcome_reward(trajectory, gt)
