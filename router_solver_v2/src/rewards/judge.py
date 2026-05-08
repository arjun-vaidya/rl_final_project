import logging
import os
from typing import Iterable, List, Tuple
from dotenv import load_dotenv

from src.utils.openai_compat_client import OpenAICompatClient

load_dotenv()
logger = logging.getLogger(__name__)

class Judge:
    def __init__(self, model: str = "qwen:7b"):
        self.client = OpenAICompatClient.from_env(
            prefix="OLLAMA",
            default_base_url="http://localhost:11434/v1/chat/completions",
            default_model=model,
        )
        self.model = self.client.model
        self.plan_batch_size = max(1, int(os.getenv("JUDGE_PLAN_BATCH_SIZE", "3")))
        self.step_batch_size = max(1, int(os.getenv("JUDGE_STEP_BATCH_SIZE", "3")))
        self.max_question_chars = max(64, int(os.getenv("JUDGE_MAX_QUESTION_CHARS", "400")))
        self.max_reasoning_chars = max(128, int(os.getenv("JUDGE_MAX_REASONING_CHARS", "800")))
        self.max_step_label_chars = max(64, int(os.getenv("JUDGE_MAX_STEP_LABEL_CHARS", "200")))

    @staticmethod
    def _chunked(items: List, chunk_size: int) -> Iterable[List]:
        for start_idx in range(0, len(items), chunk_size):
            yield items[start_idx : start_idx + chunk_size]

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        text = str(text).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _extract_score_list(self, text: str, expected_len: int) -> List[float]:
        start_idx = text.find("[")
        end_idx = text.rfind("]")
        if start_idx == -1 or end_idx == -1:
            logger.warning("No JSON array found in response: %s", text[:100])
            return [0.0] * expected_len

        json_str = text[start_idx : end_idx + 1]
        try:
            import json

            scores = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.error("JSON decode error in judge response: %s", exc)
            return [0.0] * expected_len

        if not isinstance(scores, list):
            logger.error("Expected list of %d scores, got %s", expected_len, scores)
            return [0.0] * expected_len

        normalized = [min(max(float(score) / 10.0, 0.0), 1.0) for score in scores]
        if len(normalized) != expected_len:
            logger.warning(
                "Judge returned %d scores for %d items; padding/truncating response",
                len(normalized),
                expected_len,
            )
            if len(normalized) < expected_len:
                normalized.extend([0.0] * (expected_len - len(normalized)))
            else:
                normalized = normalized[:expected_len]

        return normalized

    def judge_plan(self, question: str, plan: list) -> float:
        if not plan or len(plan) > 8:
            return 0.0
        scores = self.batch_judge_plans([(question, plan)])
        return scores[0] if scores else 0.0

    def batch_judge_plans(self, items: List[Tuple[str, list]]) -> List[float]:
        """Judge multiple plans in one API call. items: [(question, plan), ...]"""
        if not items:
            return []

        all_scores: List[float] = []
        for chunk in self._chunked(items, self.plan_batch_size):
            batch_prompt = (
                f"Rate each plan 0-10 on clarity, independence, completeness.\n"
                f"You must return exactly {len(chunk)} integers in a JSON array.\n"
                f"Respond ONLY with JSON like: [7, 9, 6]\n\n"
            )
            for idx, (question, plan) in enumerate(chunk):
                safe_question = self._clip(question, self.max_question_chars)
                plan_text = "\n".join(
                    [f"{i+1}. {self._clip(step, self.max_step_label_chars)}" for i, step in enumerate(plan)]
                )
                batch_prompt += f"PLAN {idx}:\nQ: {safe_question}\n{plan_text}\n\n"

            try:
                response = self.client.chat_completion([{"role": "user", "content": batch_prompt}])
                text = response["choices"][0]["message"]["content"]
                logger.info("Judge response (plans): %s", text)
                all_scores.extend(self._extract_score_list(text, len(chunk)))
            except Exception as e:
                logger.error(f"Batch judge error: {e}")
                all_scores.extend([0.0] * len(chunk))
        return all_scores

    def judge_step(self, question: str, plan: list, step_idx: int, reasoning: str) -> float:
        scores = self.batch_judge_steps([(question, plan, step_idx, reasoning)])
        return scores[0] if scores else 0.0

    def batch_judge_steps(self, items: List[Tuple[str, list, int, str]]) -> List[float]:
        """Judge multiple steps in one API call. items: [(question, plan, step_idx, reasoning), ...]"""
        if not items:
            return []

        all_scores: List[float] = []
        for chunk in self._chunked(items, self.step_batch_size):
            batch_prompt = (
                f"Rate each reasoning step 0-10 for relevance and usefulness.\n"
                f"You must return exactly {len(chunk)} integers in a JSON array.\n"
                f"Respond ONLY with JSON like: [4, 8, 5]\n\n"
            )
            for idx, (question, plan, step_idx, reasoning) in enumerate(chunk):
                safe_question = self._clip(question, self.max_question_chars)
                safe_label = self._clip(plan[step_idx], self.max_step_label_chars)
                safe_reasoning = self._clip(reasoning, self.max_reasoning_chars)
                batch_prompt += (
                    f"STEP {idx}:\n"
                    f"Q: {safe_question}\n"
                    f"Step {step_idx+1}: {safe_label}\n"
                    f"R: {safe_reasoning}\n\n"
                )

            try:
                response = self.client.chat_completion([{"role": "user", "content": batch_prompt}])
                text = response["choices"][0]["message"]["content"]
                logger.info("Judge response (steps): %s", text)
                all_scores.extend(self._extract_score_list(text, len(chunk)))
            except Exception as e:
                logger.error(f"Batch judge error: {e}")
                all_scores.extend([0.0] * len(chunk))
        return all_scores

    def judge_answer(self, question: str, answer: str, ground_truth: str) -> float:
        if str(answer).strip().lower() == str(ground_truth).strip().lower():
            return 1.0
        try:
            return 1.0 if abs(float(answer) - float(ground_truth)) < 1e-6 else 0.0
        except:
            return 0.0
