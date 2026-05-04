# src/rewards/router.py
from abc import ABC, abstractmethod
from src.utils.parsing import parse_plan_json
from src.rewards.outcome import outcome_reward
import anthropic
import json


class RouterReward(ABC):
    """Base class for router reward evaluators."""

    @abstractmethod
    def compute_reward(self, plan_output: str, trajectory: str, ground_truth, question: str = None) -> float:
        """
        Compute router reward (0.0-1.0).

        Args:
            plan_output: JSON string containing the router's plan
            trajectory: Solver's execution trajectory/output
            ground_truth: Correct answer for the problem
            question: Optional problem question for context

        Returns:
            Reward score 0.0-1.0
        """
        pass


class HeuristicRouterReward(RouterReward):
    """Hand-crafted heuristics: evaluate plan quality based on structure."""

    def compute_reward(self, plan_output: str, trajectory: str, ground_truth, question: str = None) -> float:
        """
        Hybrid reward: 0.5 × plan_quality + 0.5 × outcome_correctness.
        Evaluates both planning structure and final answer.
        """
        plan_dict = parse_plan_json(plan_output)
        if plan_dict is None:
            return 0.0

        plan_list = plan_dict.get("plan", [])
        if not isinstance(plan_list, list):
            return 0.0

        if not (1 <= len(plan_list) <= 8):
            return 0.0

        # Process signal: evaluate plan quality independently
        plan_quality = self._evaluate_plan_quality(plan_list, question)

        # Outcome signal: is final answer correct?
        outcome_signal = outcome_reward(trajectory, ground_truth)

        # Hybrid: balance process (structure) with outcome (correctness)
        return 0.5 * plan_quality + 0.5 * outcome_signal

    @staticmethod
    def _evaluate_plan_quality(plan_list: list, question: str = None) -> float:
        """
        Score plan quality (0.0–1.0) based on structure, clarity, logical order.
        Independent of execution or final answer.
        """
        if not plan_list or len(plan_list) == 0:
            return 0.0

        # Convert plan steps to strings (handle both dict and string formats)
        steps_as_strings = []
        for step in plan_list:
            if isinstance(step, dict):
                # If step is a dict, extract subgoal or convert to string
                step_str = step.get("subgoal", str(step))
            else:
                step_str = str(step)
            steps_as_strings.append(step_str)

        score = 0.0

        # 1. Concreteness: each step should mention numbers/operations (0.33 weight)
        concrete_steps = 0
        for step in steps_as_strings:
            step_lower = step.lower()
            if any(c.isdigit() for c in step) or \
               any(keyword in step_lower for keyword in ['add', 'subtract', 'multiply', 'divide', 'calculate', 'compute', 'total', 'sum', 'product']):
                concrete_steps += 1

        concreteness_score = concrete_steps / len(steps_as_strings)
        score += 0.33 * concreteness_score

        # 2. Logical ordering: steps shouldn't contradict or repeat (0.33 weight)
        if HeuristicRouterReward._is_logically_ordered(steps_as_strings):
            score += 0.33

        # 3. Specificity: steps should be detailed (> 10 chars) not vague (0.34 weight)
        specific_steps = sum(1 for step in steps_as_strings if len(step.strip()) > 10)
        specificity_score = specific_steps / len(steps_as_strings)
        score += 0.34 * specificity_score

        return min(score, 1.0)

    @staticmethod
    def _is_logically_ordered(plan_list: list) -> bool:
        """
        Heuristic: steps shouldn't repeat the same topic twice.
        """
        keywords_seen = set()

        for step in plan_list:
            text = step.get("subgoal", "") if isinstance(step, dict) else str(step)
            words = text.lower().split()
            step_keywords = {w for w in words if len(w) > 3 and (any(c.isdigit() for c in w) or w.isalpha())}

            overlap = keywords_seen & step_keywords
            if len(overlap) > 3:
                return False

            keywords_seen.update(step_keywords)

        return True


class LLMJudgeRouterReward(RouterReward):
    """Use Claude to evaluate plan quality as a judge."""

    def __init__(self, model: str = "claude-opus-4-1"):
        """
        Initialize LLM-based reward evaluator.

        Args:
            model: Claude model ID to use for evaluation
        """
        self.client = anthropic.Anthropic()
        self.model = model

    def compute_reward(self, plan_output: str, trajectory: str, ground_truth, question: str = None) -> float:
        """
        Hybrid reward using Claude: 0.5 × llm_plan_quality + 0.5 × outcome_correctness.
        """
        plan_dict = parse_plan_json(plan_output)
        if plan_dict is None:
            return 0.0

        plan_list = plan_dict.get("plan", [])
        if not isinstance(plan_list, list):
            return 0.0

        if not (1 <= len(plan_list) <= 8):
            return 0.0

        # Get Claude's evaluation of plan quality
        plan_quality = self._evaluate_plan_with_llm(plan_list, question)

        # Outcome signal: is final answer correct?
        outcome_signal = outcome_reward(trajectory, ground_truth)

        # Hybrid: balance process (Claude's eval) with outcome (correctness)
        return 0.5 * plan_quality + 0.5 * outcome_signal

    def _evaluate_plan_with_llm(self, plan_list: list, question: str = None) -> float:
        """
        Use Claude to score plan quality 0-10, return as 0.0-1.0.
        """
        try:
            formatted_steps = []
            for i, step in enumerate(plan_list):
                text = step.get("subgoal", "") if isinstance(step, dict) else str(step)
                formatted_steps.append(f"{i+1}. {text}")
            plan_text = "\n".join(formatted_steps)

            prompt = f"""Evaluate the quality of this mathematical reasoning plan.

Question: {question if question else "Unknown"}

Plan:
{plan_text}

Score 0-10 based on:
- Clarity: Is each step explicit and understandable?
- Completeness: Does it decompose the problem fully?
- Feasibility: Would this plan lead to the correct answer if executed perfectly?

Respond with ONLY a number from 0 to 10 (no explanation)."""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )

            score_str = response.content[0].text.strip()
            score = float(score_str) / 10.0
            return min(max(score, 0.0), 1.0)

        except Exception as e:
            print(f"Warning: LLM evaluation failed ({e}), returning 0.5")
            return 0.5


# Backward compatibility: default to heuristic
def router_reward(plan_output, trajectory, gt, question: str = None) -> float:
    """
    Legacy function for backward compatibility.
    Uses heuristic-based reward evaluation.
    """
    evaluator = HeuristicRouterReward()
    return evaluator.compute_reward(plan_output, trajectory, gt, question)

