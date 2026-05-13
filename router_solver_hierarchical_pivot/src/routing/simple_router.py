import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

BRANCH_ORDER: Tuple[str, ...] = ("easy", "soft", "hard")
BRANCH_COSTS: Dict[str, float] = {
    "easy": 0.0,
    "soft": 0.15,
    "hard": 0.30,
}


KEYWORD_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("total", "altogether", "combined", "in all", "sum"),
    ("left", "remain", "remaining", "difference", "more"),
    ("each", "per", "rate", "hour", "minute", "day", "week"),
    ("half", "twice", "double", "triple", "percent", "%"),
    ("fraction", "ratio", "/", "decimal"),
)

NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def extract_question_features(question: str) -> torch.Tensor:
    lowered = question.lower()
    tokens = question.split()
    numbers = NUMBER_RE.findall(question)
    features: List[float] = [
        float(len(question)),
        float(len(tokens)),
        float(len(numbers)),
        float(question.count("?")),
        float(question.count(",")),
        float(question.count("$")),
    ]

    for keywords in KEYWORD_GROUPS:
        features.append(float(any(keyword in lowered for keyword in keywords)))

    if numbers:
        numeric_values = []
        for item in numbers:
            try:
                numeric_values.append(abs(float(item)))
            except ValueError:
                continue
        if numeric_values:
            features.extend(
                [
                    float(min(numeric_values)),
                    float(max(numeric_values)),
                    float(sum(numeric_values) / len(numeric_values)),
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0, 0.0, 0.0])

    return torch.tensor(features, dtype=torch.float32)


@dataclass
class RouterExample:
    question: str
    ground_truth: str
    branch_summaries: Dict[str, Dict]

    @property
    def label(self) -> int:
        available = self.available_branches
        best_branch = max(
            available,
            key=lambda branch: (branch_score(self.branch_summaries[branch], branch_cost=BRANCH_COSTS[branch]), -BRANCH_ORDER.index(branch)),
        )
        return BRANCH_ORDER.index(best_branch)

    @property
    def available_branches(self) -> List[str]:
        return [branch for branch in BRANCH_ORDER if branch in self.branch_summaries]


def branch_score(summary: Dict, branch_cost: float) -> float:
    majority = 1.0 if summary.get("majority_relaxed_match", False) else 0.0
    any_relaxed = 1.0 if summary.get("any_relaxed_match", False) else 0.0
    vote_support = float(summary.get("majority_vote_fraction", 0.0))
    return 2.0 * majority + 0.5 * any_relaxed + 0.25 * vote_support - branch_cost


class QuestionBranchRouter(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_dataset(examples: Sequence[RouterExample]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.stack([extract_question_features(example.question) for example in examples])
    ys = torch.tensor([example.label for example in examples], dtype=torch.long)
    return xs, ys


def train_multiclass_router(
    train_examples: Sequence[RouterExample],
    eval_examples: Sequence[RouterExample],
    epochs: int = 200,
    learning_rate: float = 1e-2,
) -> Dict:
    if not train_examples:
        raise ValueError("train_examples must be non-empty")

    train_x, train_y = build_dataset(train_examples)
    input_dim = train_x.shape[1]
    router = QuestionBranchRouter(input_dim, num_classes=len(BRANCH_ORDER))
    optimizer = torch.optim.Adam(router.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = router(train_x)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()

    metrics = {"router": router}
    metrics.update(evaluate_multiclass_router(router, train_examples, split_name="train"))
    metrics.update(evaluate_multiclass_router(router, eval_examples, split_name="eval"))
    return metrics


def evaluate_multiclass_router(router: QuestionBranchRouter, examples: Sequence[RouterExample], split_name: str) -> Dict:
    if not examples:
        metrics = {
            f"{split_name}_router_accuracy": 0.0,
            f"{split_name}_routed_accuracy": 0.0,
            f"{split_name}_oracle_accuracy": 0.0,
            f"{split_name}_count": 0,
        }
        for branch in BRANCH_ORDER:
            metrics[f"{split_name}_{branch}_accuracy"] = 0.0
        return metrics

    x, y = build_dataset(examples)
    with torch.no_grad():
        preds = router(x).argmax(dim=-1)

    router_correct = float((preds == y).float().mean().item())
    routed_correct = 0
    oracle_correct = 0
    branch_correct = {branch: 0 for branch in BRANCH_ORDER}

    for pred, gold, example in zip(preds.tolist(), y.tolist(), examples):
        pred_branch = BRANCH_ORDER[pred]
        gold_branch = BRANCH_ORDER[gold]
        if pred_branch not in example.branch_summaries:
            pred_branch = example.available_branches[0]
        chosen = example.branch_summaries[pred_branch]
        oracle = example.branch_summaries[gold_branch]
        routed_correct += int(bool(chosen.get("majority_relaxed_match", False)))
        oracle_correct += int(bool(oracle.get("majority_relaxed_match", False)))
        for branch in example.available_branches:
            branch_correct[branch] += int(bool(example.branch_summaries[branch].get("majority_relaxed_match", False)))

    count = len(examples)
    metrics = {
        f"{split_name}_router_accuracy": router_correct,
        f"{split_name}_routed_accuracy": routed_correct / count,
        f"{split_name}_oracle_accuracy": oracle_correct / count,
        f"{split_name}_count": count,
    }
    for branch in BRANCH_ORDER:
        metrics[f"{split_name}_{branch}_accuracy"] = branch_correct[branch] / count
    return metrics


def train_binary_router(
    train_examples: Sequence[RouterExample],
    eval_examples: Sequence[RouterExample],
    epochs: int = 200,
    learning_rate: float = 1e-2,
) -> Dict:
    return train_multiclass_router(train_examples, eval_examples, epochs=epochs, learning_rate=learning_rate)
