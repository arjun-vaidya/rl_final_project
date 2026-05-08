import json
import os
import logging
import time
from openai import OpenAI, RateLimitError
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _retry_with_backoff(func, max_retries=5, initial_delay=1.0):
    """Retry with exponential backoff for rate limit errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logger.warning(f"Rate limited. Retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                logger.error(f"Rate limit exceeded after {max_retries} retries")
                raise
        except Exception as e:
            raise


class Judge:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def judge_plan(self, question: str, plan: list) -> float:
        if not plan or len(plan) > 8:
            return 0.0
        scores = self.batch_judge_plans([(question, plan)])
        return scores[0] if scores else 0.0

    def batch_judge_plans(self, items: List[Tuple[str, list]]) -> List[float]:
        """Judge multiple plans in one API call. items: [(question, plan), ...]"""
        if not items:
            return []

        batch_prompt = "Rate each plan 0-10. Respond with JSON array: [{\"idx\": 0, \"score\": <0-10>}, ...]\n\n"
        for idx, (question, plan) in enumerate(items):
            plan_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(plan)])
            batch_prompt += f"PLAN {idx}:\nQuestion: {question}\nPlan:\n{plan_text}\n\n"

        batch_prompt += "Rate based on: clarity, independence, completeness.\n"
        batch_prompt += "Respond: [{\"idx\": <0-" + str(len(items)-1) + ">, \"score\": <0-10>}, ...]"

        def api_call():
            return client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": batch_prompt}],
                temperature=0.0,
            )

        try:
            response = _retry_with_backoff(api_call, max_retries=5, initial_delay=1.0)
            text = response.choices[0].message.content

            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx == -1 or end_idx == -1:
                logger.warning(f"No JSON array found in response: {text[:100]}")
                return [0.0] * len(items)

            json_str = text[start_idx:end_idx+1]
            results = json.loads(json_str)

            if not isinstance(results, list):
                logger.error(f"Expected list, got {type(results)}")
                return [0.0] * len(items)

            scores = [0.0] * len(items)
            for result in results:
                idx = result.get("idx", -1)
                if 0 <= idx < len(items):
                    score = result.get("score", 0)
                    if isinstance(score, (int, float)):
                        scores[idx] = min(max(score / 10.0, 0.0), 1.0)
            return scores
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in batch_judge_plans: {e}")
            return [0.0] * len(items)
        except Exception as e:
            logger.error(f"Batch judge API error: {e}")
            return [0.0] * len(items)

    def judge_step(self, question: str, plan: list, step_idx: int, reasoning: str) -> float:
        scores = self.batch_judge_steps([(question, plan, step_idx, reasoning)])
        return scores[0] if scores else 0.0

    def batch_judge_steps(self, items: List[Tuple[str, list, int, str]]) -> List[float]:
        """Judge multiple steps in one API call. items: [(question, plan, step_idx, reasoning), ...]"""
        if not items:
            return []

        batch_prompt = "Rate each step 0-10. Respond with JSON array: [{\"idx\": 0, \"score\": <0-10>}, ...]\n\n"
        for idx, (question, plan, step_idx, reasoning) in enumerate(items):
            batch_prompt += f"STEP {idx}:\nQuestion: {question}\nStep {step_idx+1}: {plan[step_idx]}\nReasoning: {reasoning}\n\n"

        batch_prompt += "Respond: [{\"idx\": <0-" + str(len(items)-1) + ">, \"score\": <0-10>}, ...]"

        def api_call():
            return client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": batch_prompt}],
                temperature=0.0,
            )

        try:
            response = _retry_with_backoff(api_call, max_retries=5, initial_delay=1.0)
            text = response.choices[0].message.content

            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx == -1 or end_idx == -1:
                logger.warning(f"No JSON array found in response: {text[:100]}")
                return [0.0] * len(items)

            json_str = text[start_idx:end_idx+1]
            results = json.loads(json_str)

            if not isinstance(results, list):
                logger.error(f"Expected list, got {type(results)}")
                return [0.0] * len(items)

            scores = [0.0] * len(items)
            for result in results:
                idx = result.get("idx", -1)
                if 0 <= idx < len(items):
                    score = result.get("score", 0)
                    if isinstance(score, (int, float)):
                        scores[idx] = min(max(score / 10.0, 0.0), 1.0)
            return scores
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in batch_judge_steps: {e}")
            return [0.0] * len(items)
        except Exception as e:
            logger.error(f"Batch judge API error: {e}")
            return [0.0] * len(items)

    def judge_answer(self, question: str, answer: str, ground_truth: str) -> float:
        if str(answer).strip().lower() == str(ground_truth).strip().lower():
            return 1.0
        try:
            return 1.0 if abs(float(answer) - float(ground_truth)) < 1e-6 else 0.0
        except:
            return 0.0
