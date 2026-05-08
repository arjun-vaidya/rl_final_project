import json
import os
import logging
import time
import requests
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1/chat/completions")
MODEL = os.getenv("OLLAMA_MODEL", "qwen:7b")


def _call_ollama(messages, max_retries=3, initial_delay=1.0):
    """Call Ollama API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": MODEL, "messages": messages, "temperature": 0.0},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logger.warning(f"Ollama connection failed. Retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay)
            else:
                logger.error("Could not connect to Ollama after retries")
                raise
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise


class Judge:
    def __init__(self, model: str = "qwen:7b"):
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

        batch_prompt = "Rate each plan 0-10 on clarity, independence, completeness.\nRespond ONLY with JSON array of numbers: [<score1>, <score2>, ...]\n\n"
        for idx, (question, plan) in enumerate(items):
            plan_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(plan)])
            batch_prompt += f"PLAN {idx}:\nQ: {question}\n{plan_text}\n\n"

        try:
            response = _call_ollama([{"role": "user", "content": batch_prompt}])
            text = response["choices"][0]["message"]["content"]
            logger.info(f"Ollama response (plans): {text}")

            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx == -1 or end_idx == -1:
                logger.warning(f"No JSON array found in response: {text[:100]}")
                return [0.0] * len(items)

            json_str = text[start_idx:end_idx+1]
            scores = json.loads(json_str)

            if not isinstance(scores, list) or len(scores) != len(items):
                logger.error(f"Expected list of {len(items)}, got {scores}")
                return [0.0] * len(items)

            return [min(max(float(s) / 10.0, 0.0), 1.0) for s in scores]
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in batch_judge_plans: {e}")
            return [0.0] * len(items)
        except Exception as e:
            logger.error(f"Batch judge error: {e}")
            return [0.0] * len(items)

    def judge_step(self, question: str, plan: list, step_idx: int, reasoning: str) -> float:
        scores = self.batch_judge_steps([(question, plan, step_idx, reasoning)])
        return scores[0] if scores else 0.0

    def batch_judge_steps(self, items: List[Tuple[str, list, int, str]]) -> List[float]:
        """Judge multiple steps in one API call. items: [(question, plan, step_idx, reasoning), ...]"""
        if not items:
            return []

        batch_prompt = "Rate each step 0-10. Respond ONLY with JSON array of numbers: [<score1>, <score2>, ...]\n\n"
        for idx, (question, plan, step_idx, reasoning) in enumerate(items):
            batch_prompt += f"STEP {idx}:\nQ: {question}\nStep {step_idx+1}: {plan[step_idx]}\nR: {reasoning}\n\n"

        try:
            response = _call_ollama([{"role": "user", "content": batch_prompt}])
            text = response["choices"][0]["message"]["content"]
            logger.info(f"Ollama response (steps): {text}")

            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx == -1 or end_idx == -1:
                logger.warning(f"No JSON array found in response: {text[:100]}")
                return [0.0] * len(items)

            json_str = text[start_idx:end_idx+1]
            scores = json.loads(json_str)

            if not isinstance(scores, list) or len(scores) != len(items):
                logger.error(f"Expected list of {len(items)}, got {scores}")
                return [0.0] * len(items)

            return [min(max(float(s) / 10.0, 0.0), 1.0) for s in scores]
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in batch_judge_steps: {e}")
            return [0.0] * len(items)
        except Exception as e:
            logger.error(f"Batch judge error: {e}")
            return [0.0] * len(items)

    def judge_answer(self, question: str, answer: str, ground_truth: str) -> float:
        if str(answer).strip().lower() == str(ground_truth).strip().lower():
            return 1.0
        try:
            return 1.0 if abs(float(answer) - float(ground_truth)) < 1e-6 else 0.0
        except:
            return 0.0
