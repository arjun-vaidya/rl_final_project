from typing import Dict, List
from src.rewards.judge import Judge


class RewardShaper:
    def __init__(self, use_judge: bool = True):
        self.judge = Judge() if use_judge else None

    def compute(self, question: str, ground_truth: str, plan: List[str], steps: List, final_answer: str) -> Dict:
        if not plan:
            return {
                "router": 0.0,
                "steps": [],
                "outcome": 0.0,
            }

        router_reward = self._router_reward(question, plan)
        step_rewards = [
            self._step_reward(question, plan, idx, step.reasoning)
            for idx, step in enumerate(steps)
        ]
        outcome_reward = self._outcome_reward(final_answer, ground_truth)

        return {
            "router": router_reward,
            "steps": step_rewards,
            "outcome": outcome_reward,
        }

    def _router_reward(self, question: str, plan: List[str]) -> float:
        if self.judge:
            plan_score = self.judge.judge_plan(question, plan)
        else:
            plan_score = min(len(plan) / 8.0, 1.0)

        structural = min(len(plan) / 8.0, 1.0)
        reward = 0.4 * structural + 0.4 * plan_score + 0.2
        return min(max(reward, 0.0), 1.0)

    def _step_reward(self, question: str, plan: List[str], idx: int, reasoning: str) -> float:
        if self.judge:
            return self.judge.judge_step(question, plan, idx, reasoning)
        return 0.3 if len(reasoning) > 50 else 0.1

    def _outcome_reward(self, answer: str, ground_truth: str) -> float:
        if self.judge:
            return self.judge.judge_answer("", answer, ground_truth)
        if str(answer).strip().lower() == str(ground_truth).strip().lower():
            return 1.0
        try:
            return 1.0 if abs(float(answer) - float(ground_truth)) < 1e-6 else 0.0
        except:
            return 0.0


class Scheduler:
    def __init__(self, router_w: float = 0.3, solver_w: float = 0.5, outcome_w: float = 0.2, decay: float = 0.95):
        self.router_w = router_w
        self.solver_w = solver_w
        self.outcome_w = outcome_w
        self.decay = decay

    def get_weights(self, epoch: int):
        decayed_router = self.router_w * (self.decay ** epoch)
        total = decayed_router + self.solver_w + self.outcome_w
        return decayed_router / total, self.solver_w / total, self.outcome_w / total
