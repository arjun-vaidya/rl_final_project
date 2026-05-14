import re
from typing import Optional


BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def extract_boxed_answer(text: str) -> Optional[str]:
    # Extract answer from \\boxed{} format.
    if not text:
        return None
    matches = BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def extract_last_number(text: str) -> Optional[float]:
    # Extract last number in text (fallback).
    if not text:
        return None
    cleaned = text.replace(",", "")
    matches = NUMBER_RE.findall(cleaned)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def numeric_match(predicted: Optional[str], ground_truth: str) -> bool:
    # Check numeric match.
    if predicted is None:
        return False

    try:
        pred_num = float(str(predicted).replace(",", "").strip())
    except (ValueError, TypeError):
        pred_num = extract_last_number(str(predicted))

    try:
        gt_num = float(str(ground_truth).replace(",", "").strip())
    except (ValueError, TypeError):
        gt_num = extract_last_number(str(ground_truth))

    if pred_num is None or gt_num is None:
        return False

    return abs(pred_num - gt_num) < 1e-6


def compute_reward(generated_text: str, ground_truth: str, cfg) -> tuple:
    # Verifiable reward: correct_reward + format_reward if boxed and matches GT,
    # format_reward if boxed but wrong, 0 otherwise.
    boxed = extract_boxed_answer(generated_text)
    has_format = boxed is not None

    # is_correct requires BOTH a boxed answer AND a numeric match
    is_correct = has_format and numeric_match(boxed, ground_truth)

    reward = cfg.incorrect_reward
    if is_correct:
        reward = cfg.correct_reward
    if has_format:
        reward += cfg.format_reward

    # For logging/eval: show what the model actually emitted (boxed first, last-number otherwise)
    predicted = boxed if boxed is not None else extract_last_number(generated_text)

    return reward, is_correct, predicted
