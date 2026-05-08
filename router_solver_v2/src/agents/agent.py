import json
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Step:
    idx: int
    subgoal: str
    reasoning: str
    answer: str
    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor


@dataclass
class Rollout:
    question: str
    ground_truth: str
    router_prompt_ids: torch.Tensor
    router_completion_ids: torch.Tensor
    plan: Optional[List[str]]
    steps: List[Step] = field(default_factory=list)
    final_answer: Optional[str] = None
    _router_reward: float = 0.0
    _step_rewards: List[float] = field(default_factory=list)
    _outcome_reward: float = 0.0

    def is_valid(self) -> bool:
        """Check if rollout has valid plan and steps."""
        return self.plan is not None and len(self.steps) > 0


class Agent:
    def __init__(
        self,
        model,
        tokenizer,
        router_adapter: str = "router",
        solver_adapter: str = "solver",
        router_max_tokens: int = 300,
        solver_max_tokens: int = 200,
        router_temperature: float = 1.0,
        solver_temperature: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.router_adapter = router_adapter
        self.solver_adapter = solver_adapter
        self.router_max_tokens = router_max_tokens
        self.solver_max_tokens = solver_max_tokens
        self.router_temperature = router_temperature
        self.solver_temperature = solver_temperature

    def _set_adapter(self, name: str):
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter(name)

    @torch.no_grad()
    def _generate(self, prompt: str, max_tokens: int, temp: float = 1.0) -> Tuple[str, torch.Tensor, torch.Tensor]:
        device = getattr(self.model, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_ids = inputs.input_ids[0]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        full_ids = outputs[0]
        completion_ids = full_ids[prompt_ids.size(0):]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        return text, prompt_ids.detach(), completion_ids.detach()

    def _parse_plan(self, text: str) -> Optional[List[str]]:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1 or end == 0:
                return None
            parsed = json.loads(text[start:end])
            plan = parsed.get("plan", [])
            if not isinstance(plan, list):
                return None
            plan = [str(s).strip() for s in plan if len(str(s).strip()) > 5]
            return plan[:8] if plan else None
        except:
            return None

    def _extract_answer(self, text: str) -> str:
        lines = text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if any(line.startswith(prefix) for prefix in ["Answer:", "Final answer:", "The answer is", "So"]):
                if ':' in line:
                    return line.split(':', 1)[1].strip()
        for line in reversed(lines):
            line = line.strip()
            if line:
                return line
        return text.strip()

    def rollout(self, question: str, ground_truth: str) -> Rollout:
        self._set_adapter(self.router_adapter)
        router_prompt = f"""Decompose this math problem into clear steps.

Problem: {question}

Respond with JSON: {{"plan": ["step 1", "step 2", ...]}}

JSON:"""

        router_text, router_prompt_ids, router_comp_ids = self._generate(
            router_prompt,
            max_tokens=self.router_max_tokens,
            temp=self.router_temperature,
        )
        plan = self._parse_plan(router_text)

        rollout = Rollout(
            question=question,
            ground_truth=ground_truth,
            router_prompt_ids=router_prompt_ids,
            router_completion_ids=router_comp_ids,
            plan=plan,
        )

        if not plan:
            return rollout

        self._set_adapter(self.solver_adapter)
        previous_answers = []

        for step_idx, subgoal in enumerate(plan):
            history = ""
            if previous_answers:
                history = "Previous steps:\n"
                for i, (s, a) in enumerate(zip(plan[:step_idx], previous_answers)):
                    history += f"Step {i+1}: {s}\nAnswer: {a}\n"

            prompt = f"""Solve this step.

Question: {question}

Plan:
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(plan)])}

{history}

Step {step_idx+1}: {subgoal}

Solve this step. Show work, then state your answer.

Answer:"""

            solver_text, s_prompt_ids, s_comp_ids = self._generate(
                prompt,
                max_tokens=self.solver_max_tokens,
                temp=self.solver_temperature,
            )
            answer = self._extract_answer(solver_text)
            previous_answers.append(answer)

            step = Step(
                idx=step_idx,
                subgoal=subgoal,
                reasoning=solver_text,
                answer=answer,
                prompt_ids=s_prompt_ids,
                completion_ids=s_comp_ids,
            )
            rollout.steps.append(step)

        if rollout.steps:
            rollout.final_answer = rollout.steps[-1].answer

        return rollout
