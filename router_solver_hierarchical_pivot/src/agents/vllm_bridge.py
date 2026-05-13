import torch
from typing import Dict, List, Sequence, Tuple

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class RouterSolverVLLMBridge:
    def __init__(
        self,
        base_model: str,
        tokenizer,
        adapter_paths: Dict[str, str],
        max_lora_rank: int = 32,
        gpu_memory_utilization: float = 0.65,
        dtype: str = "bfloat16",
    ):
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not installed")
        self.tokenizer = tokenizer
        self.llm = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=max(1, len(adapter_paths)),
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )
        self.stop_token_ids = [tokenizer.eos_token_id]
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != tokenizer.unk_token_id and im_end_id not in self.stop_token_ids:
            self.stop_token_ids.append(im_end_id)
        self.lora_requests = {
            name: LoRARequest(
                lora_name=f"{name}_adapter",
                lora_int_id=idx,
                lora_path=path,
            )
            for idx, (name, path) in enumerate(adapter_paths.items(), start=1)
        }

    def _sampling_params(self, max_tokens: int, temperature: float):
        kwargs = {
            "max_tokens": max_tokens,
            "stop_token_ids": self.stop_token_ids,
        }
        if temperature is not None and temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = 1.0
        else:
            kwargs["temperature"] = 0.0
        return SamplingParams(**kwargs)

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: int,
        temperature: float,
        adapter_name: str,
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        params = self._sampling_params(max_tokens, temperature)
        kwargs = {}
        if adapter_name in self.lora_requests:
            kwargs["lora_request"] = self.lora_requests[adapter_name]
        outputs = self.llm.generate(list(prompts), params, **kwargs)
        rows: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for out in outputs:
            prompt_ids = torch.tensor(out.prompt_token_ids, dtype=torch.long)
            completion_ids = torch.tensor(list(out.outputs[0].token_ids), dtype=torch.long)
            rows.append((out.outputs[0].text, prompt_ids, completion_ids))
        return rows
