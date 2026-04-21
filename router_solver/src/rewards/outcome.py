import re

def extract_answer_from_trajectory(trajectory: str) -> int:
    """
    Extracts the numeric answer from a model's trajectory.
    Looks for <answer>X</answer> or the last number in the text as a fallback.
    """
    # Try <answer> pattern first
    match = re.search(r"<answer>\s*(-?\d+)\s*</answer>", trajectory, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Fallback to looking for "#### X" which models often use on GSM8K
    match = re.search(r"####\s*(-?\d+)", trajectory)
    if match:
        return int(match.group(1))
    
    # Last fallback: just find the last integer in the string
    numbers = re.findall(r"-?\d+", trajectory)
    if numbers:
        return int(numbers[-1])
    
    return None

def outcome_reward(trajectory: str, ground_truth: int) -> float:
    """
    Returns 1.0 if the extracted answer matches the ground truth, else 0.0.
    """
    extracted = extract_answer_from_trajectory(trajectory)
    if extracted is not None and extracted == ground_truth:
        return 1.0
    return 0.0
