import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple


SYSTEM_PROMPT = """You are a math tutor. Solve the problem step by step, showing your reasoning clearly. End with the final answer in \\boxed{} format."""


def trim_at_eos(completion_ids: torch.Tensor, stop_token_ids) -> torch.Tensor:
    """
    Trim a completion at the first stop token (inclusive).

    `stop_token_ids` can be a single int or an iterable of ints. The first
    occurrence of ANY listed stop token marks the legitimate end of the
    generation. Tokens after that point are padding from batched generation
    and would otherwise contaminate the policy gradient (training the model
    to predict runs of EOS / im_end). The first stop is kept — it's a real
    prediction the model should learn.
    """
    if isinstance(stop_token_ids, int):
        stop_token_ids = [stop_token_ids]

    mask = torch.zeros_like(completion_ids, dtype=torch.bool)
    for tid in stop_token_ids:
        if tid is None:
            continue
        mask = mask | (completion_ids == tid)

    positions = mask.nonzero(as_tuple=False)
    if positions.numel() > 0:
        first_stop = positions[0].item()
        return completion_ids[: first_stop + 1]
    return completion_ids


@dataclass
class Trajectory:
    """A single rollout trajectory."""
    question: str
    ground_truth: str

    prompt_ids: torch.Tensor = None
    completion_ids: torch.Tensor = None
    text: str = ""

    final_answer: Optional[str] = None
    is_correct: bool = False
    reward: float = 0.0


class LinearReasoningAgent:
    """
    Single-pass CoT agent.

    No router, no solver, no reflection prompt. Just:
        Question -> Chain-of-thought reasoning -> \\boxed{N}
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_cot_tokens: int = 512,
        temperature: float = 0.8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_cot_tokens = max_cot_tokens
        self.temperature = temperature

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.use_cache = True

        # Build the list of stop tokens. For Qwen2.5-Instruct, the chat template
        # ends each assistant turn with <|im_end|>. If we only stop on <|endoftext|>
        # the model never terminates naturally and pads until max_new_tokens, often
        # producing junk after the real answer.
        self.stop_token_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id and im_end_id not in self.stop_token_ids:
            self.stop_token_ids.append(im_end_id)

    def _build_prompt(self, question: str) -> str:
        """Format the prompt using the model's chat template."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Generate a single completion from a prompt."""
        device = getattr(self.model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["do_sample"] = True
        else:
            generate_kwargs["do_sample"] = False

        generate_kwargs["eos_token_id"] = self.stop_token_ids

        outputs = self.model.generate(**inputs, **generate_kwargs)
        prompt_width = inputs.input_ids.size(1)
        prompt_ids = inputs.input_ids[0].detach()
        completion_ids = outputs[0][prompt_width:].detach()
        completion_ids = trim_at_eos(completion_ids, self.stop_token_ids)
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        return text, prompt_ids, completion_ids

    @torch.inference_mode()
    def _generate_batch(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        """Generate completions for a batch of prompts (for G rollouts)."""
        device = getattr(self.model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        original_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["do_sample"] = True
        else:
            generate_kwargs["do_sample"] = False

        generate_kwargs["eos_token_id"] = self.stop_token_ids

        outputs = self.model.generate(**inputs, **generate_kwargs)
        prompt_width = inputs.input_ids.size(1)

        results = []
        for i in range(len(prompts)):
            prompt_mask = inputs.attention_mask[i].bool()
            prompt_ids = inputs.input_ids[i][prompt_mask].detach()
            completion_ids = outputs[i][prompt_width:].detach()
            # Strip padding introduced by batched generation: anything after the
            # first stop token (EOS or <|im_end|>) is padding from later-finishing rows.
            completion_ids = trim_at_eos(completion_ids, self.stop_token_ids)
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            results.append((text, prompt_ids, completion_ids))

        self.tokenizer.padding_side = original_padding
        return results

    def rollout(self, question: str, ground_truth: str, temperature: Optional[float] = None) -> Trajectory:
        """Generate one trajectory."""
        temp = temperature if temperature is not None else self.temperature

        prompt = self._build_prompt(question)
        text, prompt_ids, completion_ids = self._generate(prompt, self.max_cot_tokens, temp)

        return Trajectory(
            question=question,
            ground_truth=ground_truth,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            text=text,
        )

    def rollout_group(self, question: str, ground_truth: str, G: int, temperature: Optional[float] = None) -> List[Trajectory]:
        """Generate G trajectories for one question (GRPO group)."""
        temp = temperature if temperature is not None else self.temperature

        prompt = self._build_prompt(question)
        results = self._generate_batch([prompt] * G, self.max_cot_tokens, temp)

        return [
            Trajectory(
                question=question,
                ground_truth=ground_truth,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                text=text,
            )
            for text, prompt_ids, completion_ids in results
        ]

    def rollout_batch(self, questions: List[str], ground_truths: List[str], temperature: Optional[float] = None) -> List[Trajectory]:
        """Generate one trajectory per (question, ground_truth) pair, batched together."""
        temp = temperature if temperature is not None else self.temperature

        prompts = [self._build_prompt(q) for q in questions]
        results = self._generate_batch(prompts, self.max_cot_tokens, temp)

        return [
            Trajectory(
                question=q,
                ground_truth=gt,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                text=text,
            )
            for q, gt, (text, prompt_ids, completion_ids) in zip(questions, ground_truths, results)
        ]
