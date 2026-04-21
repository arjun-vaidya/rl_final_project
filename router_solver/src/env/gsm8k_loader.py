import re
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class GSM8KProblem:
    question: str
    answer: str
    numeric_answer: int

def extract_numeric_answer(ans_str: str) -> int:
    """Extracts the number after #### in the GSM8K answer string."""
    if "####" not in ans_str:
        return None
    ans = ans_str.split("####")[-1].strip()
    ans = ans.replace(",", "")  # Handle commas in numbers like 1,000
    # Match integer (including negative)
    match = re.search(r"(-?\d+)", ans)
    if match:
        return int(match.group(1))
    return None

def load_gsm8k(split: str = "train") -> list[GSM8KProblem]:
    """Loads GSM8K and returns a list of GSM8KProblem objects."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    problems = []
    for item in ds:
        num_ans = extract_numeric_answer(item["answer"])
        problems.append(GSM8KProblem(
            question=item["question"],
            answer=item["answer"],
            numeric_answer=num_ans
        ))
    return problems

def load_gsm8k_train():
    return load_gsm8k("train")

def load_gsm8k_test():
    return load_gsm8k("test")
