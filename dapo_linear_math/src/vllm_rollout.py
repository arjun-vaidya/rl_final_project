import os
import shutil
import tempfile
import torch
from typing import List, Optional

from agent import Trajectory

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class VLLMRollout:
    def __init__(
        self,
        base_model: str,
        tokenizer,
        max_lora_rank: int = 32,
        max_cot_tokens: int = 512,
        gpu_memory_utilization: float = 0.45,
        dtype: str = "bfloat16",
    ):
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed. `pip install vllm`")

        self.tokenizer = tokenizer
        self.max_cot_tokens = max_cot_tokens
        self.llm = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=1,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )

        self._lora_dir = tempfile.mkdtemp(prefix="dapo_lora_")
        self._lora_counter = 0
        self._current_lora: Optional[LoRARequest] = None

        self.stop_token_ids = [tokenizer.eos_token_id]
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != tokenizer.unk_token_id and im_end_id not in self.stop_token_ids:
            self.stop_token_ids.append(im_end_id)

    def sync_lora_from_peft(self, peft_model):
        """Save the PEFT adapter to a fresh dir and register it with vLLM."""
        self._lora_counter += 1
        path = os.path.join(self._lora_dir, f"v{self._lora_counter}")
        peft_model.save_pretrained(path)
        self._current_lora = LoRARequest(
            lora_name=f"policy_v{self._lora_counter}",
            lora_int_id=self._lora_counter,
            lora_path=path,
        )

    def _sampling_params(self, temperature: float):
        return SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=self.max_cot_tokens,
            stop_token_ids=self.stop_token_ids,
        )

    def generate_groups(self, agent, questions: List[str], ground_truths: List[str], G: int, temperature: float) -> List[List[Trajectory]]:
        """Generate G rollouts per question. Returns one list of G Trajectories per question."""
        prompts = []
        for q in questions:
            prompts.extend([agent._build_prompt(q)] * G)

        sp = self._sampling_params(temperature)
        kwargs = {}
        if self._current_lora is not None:
            kwargs["lora_request"] = self._current_lora

        outputs = self.llm.generate(prompts, sp, **kwargs)

        grouped = []
        for qi, (q, gt) in enumerate(zip(questions, ground_truths)):
            group = []
            for gi in range(G):
                o = outputs[qi * G + gi]
                prompt_ids = torch.tensor(o.prompt_token_ids, dtype=torch.long)
                completion_ids = torch.tensor(list(o.outputs[0].token_ids), dtype=torch.long)
                text = o.outputs[0].text
                group.append(Trajectory(
                    question=q,
                    ground_truth=gt,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    text=text,
                ))
            grouped.append(group)
        return grouped

    def generate_flat(self, agent, questions: List[str], K: int, temperature: float) -> List[List[str]]:
        """K rollouts per question, returning just text. Used by eval_sc."""
        prompts = []
        for q in questions:
            prompts.extend([agent._build_prompt(q)] * K)

        sp = self._sampling_params(temperature)
        kwargs = {}
        if self._current_lora is not None:
            kwargs["lora_request"] = self._current_lora

        outputs = self.llm.generate(prompts, sp, **kwargs)

        per_question = []
        for qi in range(len(questions)):
            per_question.append([outputs[qi * K + ki].outputs[0].text for ki in range(K)])
        return per_question

    def cleanup(self):
        if os.path.isdir(self._lora_dir):
            shutil.rmtree(self._lora_dir, ignore_errors=True)
