import json
import re
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence, Dict
from src.utils.answer_utils import extract_final_answer, extract_numeric_value


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
    invalid_reason: Optional[str] = None
    router_raw_text: Optional[str] = None
    final_answer_source: Optional[str] = None
    synthesis_reasoning: Optional[str] = None
    synthesis_prompt_ids: Optional[torch.Tensor] = None
    synthesis_completion_ids: Optional[torch.Tensor] = None
    candidate_rerank_candidates: List[str] = field(default_factory=list)
    candidate_rerank_metadata: List[Dict[str, object]] = field(default_factory=list)
    candidate_selector_output: Optional[str] = None
    answer_bearing_step_idx: Optional[int] = None
    synthesis_rejected_by_consistency: bool = False
    synthesis_vote_answers: List[str] = field(default_factory=list)
    heuristic_selected_candidate: Optional[str] = None

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
        synthesis_max_tokens: int = 64,
        router_temperature: float = 1.0,
        solver_temperature: float = 1.0,
        use_answer_synthesis: bool = False,
        constrained_final_answer_decoding: bool = False,
        candidate_rerank: bool = False,
        trace_consistency_guard: bool = False,
        answer_bearing_step_hint: bool = False,
        heuristic_final_selector: bool = False,
        heuristic_final_selector_refined: bool = False,
        guarded_heuristic_fallback: bool = False,
        synthesis_self_consistency: bool = False,
        synthesis_self_consistency_samples: int = 3,
        router_prompt_hardening: bool = False,
        plan_parse_repair: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.router_adapter = router_adapter
        self.solver_adapter = solver_adapter
        self.router_max_tokens = router_max_tokens
        self.solver_max_tokens = solver_max_tokens
        self.synthesis_max_tokens = synthesis_max_tokens
        self.router_temperature = router_temperature
        self.solver_temperature = solver_temperature
        self.use_answer_synthesis = use_answer_synthesis
        self.constrained_final_answer_decoding = constrained_final_answer_decoding
        self.candidate_rerank = candidate_rerank
        self.trace_consistency_guard = trace_consistency_guard
        self.answer_bearing_step_hint = answer_bearing_step_hint
        self.heuristic_final_selector = heuristic_final_selector
        self.heuristic_final_selector_refined = heuristic_final_selector_refined
        self.guarded_heuristic_fallback = guarded_heuristic_fallback
        self.synthesis_self_consistency = synthesis_self_consistency
        self.synthesis_self_consistency_samples = max(1, synthesis_self_consistency_samples)
        self.router_prompt_hardening = router_prompt_hardening
        self.plan_parse_repair = plan_parse_repair
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.use_cache = True

    def _set_adapter(self, name: str):
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter(name)

    @torch.inference_mode()
    def _generate(self, prompt: str, max_tokens: int, temp: float = 1.0) -> Tuple[str, torch.Tensor, torch.Tensor]:
        return self._generate_batch([prompt], max_tokens=max_tokens, temp=temp)[0]

    @torch.inference_mode()
    def _generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: int,
        temp: float = 1.0,
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        device = getattr(self.model, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        prompts = list(prompts)
        if not prompts:
            return []

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temp is not None and temp > 0:
            generate_kwargs["temperature"] = temp
            generate_kwargs["do_sample"] = True
        else:
            generate_kwargs["do_sample"] = False

        outputs = self.model.generate(**inputs, **generate_kwargs)
        prompt_width = inputs.input_ids.size(1)
        results: List[Tuple[str, torch.Tensor, torch.Tensor]] = []

        for row_idx in range(len(prompts)):
            prompt_mask = inputs.attention_mask[row_idx].bool()
            prompt_ids = inputs.input_ids[row_idx][prompt_mask].detach()
            full_ids = outputs[row_idx]
            completion_ids = full_ids[prompt_width:].detach()
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            results.append((text, prompt_ids, completion_ids))

        self.tokenizer.padding_side = original_padding_side
        return results

    @torch.inference_mode()
    def _generate_same_prompt(
        self,
        prompt: str,
        num_return_sequences: int,
        max_tokens: int,
        temp: float = 1.0,
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        if num_return_sequences <= 0:
            return []

        if temp is None or temp <= 0:
            return self._generate_batch([prompt] * num_return_sequences, max_tokens=max_tokens, temp=temp)

        device = getattr(self.model, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=temp,
            do_sample=True,
            num_return_sequences=num_return_sequences,
        )
        prompt_width = inputs.input_ids.size(1)
        prompt_mask = inputs.attention_mask[0].bool()
        prompt_ids = inputs.input_ids[0][prompt_mask].detach()

        results: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for row_idx in range(num_return_sequences):
            full_ids = outputs[row_idx]
            completion_ids = full_ids[prompt_width:].detach()
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            results.append((text, prompt_ids.clone(), completion_ids))

        self.tokenizer.padding_side = original_padding_side
        return results

    def _build_router_prompt(self, question: str) -> str:
        if not self.router_prompt_hardening:
            return f"""Decompose this math problem into clear steps.

Problem: {question}

Respond with JSON: {{"plan": ["step 1", "step 2", ...]}}

JSON:"""

        return f"""Decompose this math problem into clear steps.

Problem: {question}

Respond with valid JSON only. Use this exact schema:
{{"plan": ["Step 1: ...", "Step 2: ..."]}}

Rules:
- Output only JSON. No markdown, no prose, no code fences.
- Make the final step explicitly compute the final answer to the original problem.
- Keep the plan between 2 and 6 steps.

JSON:"""

    @staticmethod
    def _build_solver_prompt(question: str, plan: List[str], previous_answers: List[str], step_idx: int, subgoal: str) -> str:
        history = ""
        if previous_answers:
            history = "Previous steps:\n"
            for i, (step_text, answer_text) in enumerate(zip(plan[:step_idx], previous_answers)):
                history += f"Step {i+1}: {step_text}\nAnswer: {answer_text}\n"

        return f"""Solve this step.

Question: {question}

Plan:
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(plan)])}

{history}

Step {step_idx+1}: {subgoal}

Solve this step. Show work, then end with a final line exactly in the form:
Final answer: <number>

Do not end with prose after the final answer line."""

    def _build_synthesis_prompt(self, question: str, plan: List[str], steps: List[Step], answer_bearing_step_idx: Optional[int] = None) -> str:
        trace_lines = []
        for step in steps:
            trace_lines.append(f"Subgoal {step.idx+1}: {step.subgoal}")
            trace_lines.append(f"Step answer {step.idx+1}: {step.answer}")
            trace_lines.append(f"Reasoning {step.idx+1}: {step.reasoning}")
        trace = "\n".join(trace_lines)
        hint_block = ""
        if answer_bearing_step_idx is not None and 0 <= answer_bearing_step_idx < len(steps):
            hint_step = steps[answer_bearing_step_idx]
            hint_block = (
                f"\nLikely answer-bearing step:\n"
                f"- Step {hint_step.idx+1}\n"
                f"- Subgoal: {hint_step.subgoal}\n"
                f"- Extracted answer: {hint_step.answer}\n"
            )
        if self.constrained_final_answer_decoding:
            return f"""You are given a solved multi-step math trace. Return only the final numeric answer to the original question.

Original question: {question}

Plan:
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(plan)])}

Trace:
{trace}
{hint_block}

Output rules:
- Output digits only, with an optional leading minus sign or decimal point.
- No words.
- No explanation.
- No label such as 'Final answer'.
- If the trace contains intermediate values, return only the value that answers the original question."""

        return f"""You are given a solved multi-step math trace. Your job is to extract or compute the single final answer to the original question.

Original question: {question}

Plan:
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(plan)])}

Trace:
{trace}
{hint_block}

Return exactly one final line in the form:
Final answer: <number>

If an earlier intermediate is not the final answer to the original question, do not return it."""

    @staticmethod
    def _looks_candidate_line(line: str) -> bool:
        lowered = line.lower()
        if not lowered:
            return False
        return any(token in lowered for token in ["answer", "final", "therefore", "thus", "total", "altogether", "="])

    @staticmethod
    def _infer_target_type(question: str) -> str:
        lowered = question.lower()
        if any(token in lowered for token in ["altogether", "total", "in all", "sum", "combined"]):
            return "total"
        if any(token in lowered for token in ["left", "remain", "remaining", "still need", "more money", "difference", "how many more"]):
            return "difference_or_remaining"
        if any(token in lowered for token in ["each", "per", "rate", "hour", "hours will it take"]):
            return "rate_or_unit"
        return "direct_answer"

    @staticmethod
    def _target_score(question: str, subgoal: str, step_idx: int, total_steps: int, source: str) -> float:
        lowered_q = question.lower()
        lowered_s = subgoal.lower()
        score = 0.0
        if step_idx == total_steps - 1:
            score += 2.0
        if source == "synthesis":
            score += 1.5
        if any(token in lowered_q for token in ["altogether", "total", "combined", "in all"]):
            if any(token in lowered_s for token in ["add", "sum", "total", "altogether", "combine"]):
                score += 2.0
        if any(token in lowered_q for token in ["left", "remain", "remaining", "still need", "more"]):
            if any(token in lowered_s for token in ["subtract", "left", "remaining", "need", "difference", "more"]):
                score += 2.0
        if any(token in lowered_s for token in ["convert", "calculate how many", "find out how many"]):
            score -= 0.5
        return score

    def _extract_trace_candidates(self, question: str, rollout: Rollout) -> Tuple[List[str], List[Dict[str, object]]]:
        metadata: List[Dict[str, object]] = []

        def add_candidate(text: Optional[str], source: str, step_idx: int, subgoal: str) -> None:
            if text is None:
                return
            numeric = extract_numeric_value(text)
            if numeric is None:
                return
            normalized = str(int(numeric)) if float(numeric).is_integer() else str(numeric)
            for existing in metadata:
                existing_num = extract_numeric_value(existing["candidate"])
                if existing_num is not None and abs(existing_num - numeric) < 1e-6:
                    existing["mentions"] = int(existing.get("mentions", 1)) + 1
                    existing["score"] = max(float(existing.get("score", 0.0)), self._target_score(question, subgoal, step_idx, len(rollout.steps), source))
                    return
            metadata.append({
                "candidate": normalized,
                "source": source,
                "step_idx": step_idx,
                "subgoal": subgoal,
                "score": self._target_score(question, subgoal, step_idx, len(rollout.steps), source),
                "mentions": 1,
            })

        for step in rollout.steps:
            add_candidate(step.answer, source="step_answer", step_idx=step.idx, subgoal=step.subgoal)
            for line in step.reasoning.splitlines():
                stripped = line.strip()
                if len(stripped) > 160:
                    continue
                if self._looks_candidate_line(stripped):
                    add_candidate(stripped, source="reasoning_line", step_idx=step.idx, subgoal=step.subgoal)

        if rollout.final_answer:
            add_candidate(rollout.final_answer, source="synthesis", step_idx=max(len(rollout.steps) - 1, 0), subgoal=rollout.steps[-1].subgoal if rollout.steps else "")

        metadata.sort(key=lambda item: (-float(item["score"]), -int(item["mentions"]), int(item["step_idx"])))
        metadata = metadata[:8]
        candidates = [str(item["candidate"]) for item in metadata]
        return candidates, metadata

    @staticmethod
    def _build_candidate_selector_prompt(question: str, plan: List[str], steps: List[Step], candidates: List[str], metadata: List[Dict[str, object]], target_type: str) -> str:
        step_summary = []
        for step in steps:
            step_summary.append(f"Step {step.idx+1} subgoal: {step.subgoal}")
            step_summary.append(f"Step {step.idx+1} extracted answer: {step.answer}")

        candidate_summary = []
        for item in metadata:
            candidate_summary.append(
                f"{item['candidate']} | source={item['source']} | step={int(item['step_idx'])+1} | subgoal={item['subgoal']}"
            )

        return f"""Choose which candidate number answers the original question.

Original question: {question}
Target type: {target_type}

Plan:
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(plan)])}

Step summary:
{chr(10).join(step_summary)}

Candidate numbers with provenance:
{chr(10).join(candidate_summary)}

Return exactly one candidate number from the list above.
Do not generate a new number.
Do not include any words or explanation.
Prefer the candidate that answers the original question itself, not an intermediate value from an earlier sub-step.
If the question asks for a total, prefer totals or sums over partial quantities.
If the question asks how much remains or is still needed, prefer the remaining/difference quantity over subtotals."""

    def _match_candidate(self, selected_text: str, candidates: List[str]) -> Optional[str]:
        chosen_num = extract_numeric_value(selected_text)
        if chosen_num is None:
            return None
        for candidate in candidates:
            candidate_num = extract_numeric_value(candidate)
            if candidate_num is not None and abs(candidate_num - chosen_num) < 1e-6:
                return candidate
        return None

    def _fallback_candidate(self, candidates: List[str]) -> Optional[str]:
        return candidates[-1] if candidates else None

    def _heuristic_select_candidate(self, question: str, rollout: Rollout, candidates: List[str], metadata: List[Dict[str, object]]) -> Optional[str]:
        chosen, _score = self._heuristic_select_candidate_with_score(question, rollout, candidates, metadata)
        return chosen

    def _heuristic_select_candidate_with_score(
        self,
        question: str,
        rollout: Rollout,
        candidates: List[str],
        metadata: List[Dict[str, object]],
    ) -> Tuple[Optional[str], Optional[float]]:
        if not candidates or not metadata:
            return None, None
        target_type = self._infer_target_type(question)
        total_steps = max(len(rollout.steps), 1)
        best_candidate: Optional[str] = None
        best_key: Optional[Tuple[float, int, int]] = None
        step_answer_nums = [extract_numeric_value(step.answer) for step in rollout.steps]
        final_step_num = step_answer_nums[-1] if step_answer_nums else None

        for item in metadata:
            score = float(item.get("score", 0.0))
            step_idx = int(item.get("step_idx", 0))
            mentions = int(item.get("mentions", 1))
            source = str(item.get("source", ""))
            subgoal = str(item.get("subgoal", "")).lower()
            candidate_num = extract_numeric_value(item.get("candidate"))

            score += 0.15 * mentions
            if source == "synthesis":
                score += 0.5
            if step_idx == total_steps - 1:
                score += 1.0
            elif step_idx == total_steps - 2:
                score += 0.35
            else:
                score -= 0.2 * max(total_steps - step_idx - 2, 0)

            if target_type == "total" and any(token in subgoal for token in ["add", "sum", "total", "altogether", "combine"]):
                score += 0.75
            if target_type == "difference_or_remaining" and any(token in subgoal for token in ["subtract", "remaining", "left", "difference", "need", "more"]):
                score += 0.75
            if any(token in subgoal for token in ["convert", "rate", "per hour", "per week"]) and step_idx < total_steps - 1:
                score -= 0.5
            if source == "reasoning_line":
                score -= 0.2

            if candidate_num is not None:
                last_occurrence = -1
                early_only = False
                for idx, step_num in enumerate(step_answer_nums):
                    if step_num is not None and abs(step_num - candidate_num) < 1e-6:
                        last_occurrence = idx
                if last_occurrence != -1:
                    if last_occurrence == total_steps - 1:
                        score += 0.75
                    elif last_occurrence <= total_steps - 2:
                        score -= 0.6
                        early_only = True
                if (
                    final_step_num is not None
                    and candidate_num is not None
                    and abs(candidate_num - final_step_num) < 1e-6
                ):
                    score += 0.6
                if early_only and source != "synthesis":
                    score -= 0.4

            key = (score, mentions, step_idx)
            if best_key is None or key > best_key:
                best_key = key
                best_candidate = str(item["candidate"])

        return best_candidate, (best_key[0] if best_key is not None else None)

    def _refined_heuristic_override(self, question: str, rollout: Rollout, candidates: List[str], metadata: List[Dict[str, object]]) -> Optional[str]:
        best_candidate, best_score = self._heuristic_select_candidate_with_score(question, rollout, candidates, metadata)
        if best_candidate is None or best_score is None:
            return None

        synthesis_num = extract_numeric_value(rollout.final_answer)
        synth_score = None
        if synthesis_num is not None:
            for item in metadata:
                item_num = extract_numeric_value(item.get("candidate"))
                if item_num is not None and abs(item_num - synthesis_num) < 1e-6:
                    synth_score = float(item.get("score", 0.0))
                    break

        # If synthesis is already numeric, keep it unless the heuristic winner is meaningfully better.
        if synthesis_num is not None and synth_score is not None and best_score < synth_score + 0.9:
            return None

        # Avoid overriding with a clearly earlier intermediate if the winner does not come from synthesis.
        for item in metadata:
            if str(item.get("candidate")) != best_candidate:
                continue
            step_idx = int(item.get("step_idx", 0))
            source = str(item.get("source", ""))
            if source != "synthesis" and step_idx < max(len(rollout.steps) - 1, 0):
                if synthesis_num is not None:
                    return None
            break

        return best_candidate

    def _choose_answer_bearing_step_idx(self, question: str, rollout: Rollout) -> Optional[int]:
        if not rollout.steps:
            return None
        best_idx = None
        best_score = None
        total_steps = len(rollout.steps)
        for step in rollout.steps:
            score = self._target_score(question, step.subgoal, step.idx, total_steps, "step_answer")
            if best_score is None or score > best_score:
                best_score = score
                best_idx = step.idx
        return best_idx

    def _apply_trace_consistency_guard(self, question: str, rollout: Rollout) -> None:
        if not rollout.steps:
            return
        candidates, metadata = self._extract_trace_candidates(question, rollout)
        rollout.candidate_rerank_candidates = candidates
        rollout.candidate_rerank_metadata = metadata
        final_num = extract_numeric_value(rollout.final_answer)
        if final_num is None:
            rollout.synthesis_rejected_by_consistency = True
            fallback = self._fallback_candidate(candidates)
            if fallback:
                rollout.final_answer = fallback
                rollout.final_answer_source = "consistency_fallback"
            return

        for candidate in candidates:
            candidate_num = extract_numeric_value(candidate)
            if candidate_num is not None and abs(candidate_num - final_num) < 1e-6:
                return

        rollout.synthesis_rejected_by_consistency = True
        fallback = self._fallback_candidate(candidates)
        if fallback:
            rollout.final_answer = fallback
            rollout.final_answer_source = "consistency_fallback"

    def _apply_guarded_heuristic_fallback(self, question: str, rollout: Rollout) -> None:
        if not rollout.steps:
            return
        candidates, metadata = self._extract_trace_candidates(question, rollout)
        rollout.candidate_rerank_candidates = candidates
        rollout.candidate_rerank_metadata = metadata
        final_num = extract_numeric_value(rollout.final_answer)
        if final_num is not None:
            for candidate in candidates:
                candidate_num = extract_numeric_value(candidate)
                if candidate_num is not None and abs(candidate_num - final_num) < 1e-6:
                    return

        chosen = self._heuristic_select_candidate(question, rollout, candidates, metadata)
        if chosen:
            rollout.heuristic_selected_candidate = chosen
            rollout.final_answer = chosen
            rollout.final_answer_source = "heuristic_guard_fallback"

    def _normalize_numeric_answer(self, text: str) -> str:
        numeric = extract_numeric_value(text)
        if numeric is None:
            return ""
        return str(int(numeric)) if float(numeric).is_integer() else str(numeric)

    def _parse_plan(self, text: str) -> Optional[List[str]]:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1 or end == 0:
                return None
            parsed = json.loads(text[start:end])
            plan = parsed.get("plan", [])
            if not isinstance(plan, list):
                return self._repair_plan(text) if self.plan_parse_repair else None
            plan = [str(s).strip() for s in plan if len(str(s).strip()) > 5]
            if plan:
                return plan[:8]
            return self._repair_plan(text) if self.plan_parse_repair else None
        except:
            return self._repair_plan(text) if self.plan_parse_repair else None

    def _repair_plan(self, text: str) -> Optional[List[str]]:
        cleaned = str(text).strip()
        if not cleaned:
            return None

        numbered = re.findall(r'(?:^|\n)\s*(?:Step\s*\d+[:.)-]?|\d+[:.)-])\s*(.+)', cleaned, flags=re.IGNORECASE)
        if numbered:
            plan = [item.strip(' -"\'') for item in numbered if len(item.strip()) > 5]
            return plan[:8] if plan else None

        quoted = re.findall(r'"([^"\n]{6,})"', cleaned)
        if quoted:
            plan = [item.strip() for item in quoted if len(item.strip()) > 5]
            return plan[:8] if plan else None

        lines = []
        for line in cleaned.splitlines():
            line = line.strip(" -*\t")
            if len(line) > 8 and "plan" not in line.lower() and "json" not in line.lower():
                lines.append(line)
        return lines[:8] if len(lines) >= 2 else None

    def _extract_answer(self, text: str) -> str:
        return extract_final_answer(text)

    @staticmethod
    def extract_numeric_value(text: str) -> Optional[float]:
        return extract_numeric_value(text)

    def rollout(self, question: str, ground_truth: str) -> Rollout:
        return self.rollout_group(question, ground_truth, 1)[0]

    def rollout_group(self, question: str, ground_truth: str, num_rollouts: int) -> List[Rollout]:
        self._set_adapter(self.router_adapter)
        router_prompt = self._build_router_prompt(question)
        router_generations = self._generate_same_prompt(
            router_prompt,
            num_return_sequences=num_rollouts,
            max_tokens=self.router_max_tokens,
            temp=self.router_temperature,
        )
        rollouts: List[Rollout] = []
        previous_answers_by_rollout: List[List[str]] = []

        for router_text, router_prompt_ids, router_comp_ids in router_generations:
            plan = self._parse_plan(router_text)
            rollout = Rollout(
                question=question,
                ground_truth=ground_truth,
                router_prompt_ids=router_prompt_ids,
                router_completion_ids=router_comp_ids,
                plan=plan,
                router_raw_text=router_text,
            )
            if not plan:
                rollout.invalid_reason = "plan_parse_failed"
            rollouts.append(rollout)
            previous_answers_by_rollout.append([])

        active_indices = [idx for idx, rollout in enumerate(rollouts) if rollout.plan]
        if not active_indices:
            return rollouts

        self._set_adapter(self.solver_adapter)
        max_steps = max(len(rollouts[idx].plan) for idx in active_indices)

        for step_idx in range(max_steps):
            batch_indices: List[int] = []
            batch_prompts: List[str] = []

            for rollout_idx in active_indices:
                rollout = rollouts[rollout_idx]
                if step_idx >= len(rollout.plan):
                    continue
                batch_indices.append(rollout_idx)
                batch_prompts.append(
                    self._build_solver_prompt(
                        question,
                        rollout.plan,
                        previous_answers_by_rollout[rollout_idx],
                        step_idx,
                        rollout.plan[step_idx],
                    )
                )

            if not batch_prompts:
                continue

            solver_generations = self._generate_batch(
                batch_prompts,
                max_tokens=self.solver_max_tokens,
                temp=self.solver_temperature,
            )

            for rollout_idx, (solver_text, s_prompt_ids, s_comp_ids) in zip(batch_indices, solver_generations):
                rollout = rollouts[rollout_idx]
                answer = self._extract_answer(solver_text)
                previous_answers_by_rollout[rollout_idx].append(answer)
                rollout.steps.append(
                    Step(
                        idx=step_idx,
                        subgoal=rollout.plan[step_idx],
                        reasoning=solver_text,
                        answer=answer,
                        prompt_ids=s_prompt_ids,
                        completion_ids=s_comp_ids,
                    )
                )

        for rollout in rollouts:
            if rollout.steps:
                rollout.final_answer = rollout.steps[-1].answer
                rollout.final_answer_source = "last_step"
            elif rollout.plan:
                rollout.invalid_reason = "no_steps"

        synthesis_indices = [idx for idx, rollout in enumerate(rollouts) if rollout.steps and self.use_answer_synthesis]
        if synthesis_indices:
            for idx in synthesis_indices:
                if self.answer_bearing_step_hint:
                    rollouts[idx].answer_bearing_step_idx = self._choose_answer_bearing_step_idx(question, rollouts[idx])
            if self.synthesis_self_consistency:
                for rollout_idx in synthesis_indices:
                    rollout = rollouts[rollout_idx]
                    prompt = self._build_synthesis_prompt(
                        question,
                        rollout.plan,
                        rollout.steps,
                        rollout.answer_bearing_step_idx if self.answer_bearing_step_hint else None,
                    )
                    synth_generations = self._generate_same_prompt(
                        prompt,
                        num_return_sequences=self.synthesis_self_consistency_samples,
                        max_tokens=8 if self.constrained_final_answer_decoding else self.synthesis_max_tokens,
                        temp=0.7,
                    )
                    vote_counts: Dict[str, int] = {}
                    chosen_text = None
                    chosen_prompt_ids = None
                    chosen_comp_ids = None
                    rollout.synthesis_vote_answers = []
                    for synth_text, synth_prompt_ids, synth_comp_ids in synth_generations:
                        if self.constrained_final_answer_decoding:
                            answer = self._normalize_numeric_answer(synth_text)
                        else:
                            answer = self._extract_answer(synth_text)
                        rollout.synthesis_vote_answers.append(answer)
                        if answer:
                            vote_counts[answer] = vote_counts.get(answer, 0) + 1
                            current_votes = vote_counts[answer]
                            best_votes = vote_counts.get(chosen_text, 0) if chosen_text else -1
                            if chosen_text is None or current_votes > best_votes:
                                chosen_text = answer
                                chosen_prompt_ids = synth_prompt_ids
                                chosen_comp_ids = synth_comp_ids
                    if chosen_text:
                        rollout.final_answer = chosen_text
                        rollout.final_answer_source = "synthesis_self_consistency"
                    if synth_generations:
                        rollout.synthesis_reasoning = synth_generations[0][0]
                    rollout.synthesis_prompt_ids = chosen_prompt_ids
                    rollout.synthesis_completion_ids = chosen_comp_ids
            else:
                synthesis_prompts = [
                    self._build_synthesis_prompt(
                        question,
                        rollouts[idx].plan,
                        rollouts[idx].steps,
                        rollouts[idx].answer_bearing_step_idx if self.answer_bearing_step_hint else None,
                    )
                    for idx in synthesis_indices
                ]
                synthesis_generations = self._generate_batch(
                    synthesis_prompts,
                    max_tokens=8 if self.constrained_final_answer_decoding else self.synthesis_max_tokens,
                    temp=0.0,
                )
                for rollout_idx, (synth_text, synth_prompt_ids, synth_comp_ids) in zip(synthesis_indices, synthesis_generations):
                    rollout = rollouts[rollout_idx]
                    if self.constrained_final_answer_decoding:
                        synthesized_answer = self._normalize_numeric_answer(synth_text)
                    else:
                        synthesized_answer = self._extract_answer(synth_text)
                    if synthesized_answer:
                        rollout.final_answer = synthesized_answer
                        rollout.final_answer_source = "synthesis_constrained" if self.constrained_final_answer_decoding else "synthesis"
                    rollout.synthesis_reasoning = synth_text
                    rollout.synthesis_prompt_ids = synth_prompt_ids
                    rollout.synthesis_completion_ids = synth_comp_ids

        if self.guarded_heuristic_fallback:
            guard_indices = [idx for idx, rollout in enumerate(rollouts) if rollout.steps]
            for rollout_idx in guard_indices:
                self._apply_guarded_heuristic_fallback(question, rollouts[rollout_idx])

        if self.trace_consistency_guard:
            consistency_indices = [idx for idx, rollout in enumerate(rollouts) if rollout.steps]
            for rollout_idx in consistency_indices:
                self._apply_trace_consistency_guard(question, rollouts[rollout_idx])

        if self.heuristic_final_selector or self.heuristic_final_selector_refined:
            selector_indices = [idx for idx, rollout in enumerate(rollouts) if rollout.steps]
            for rollout_idx in selector_indices:
                rollout = rollouts[rollout_idx]
                candidates, metadata = self._extract_trace_candidates(question, rollout)
                rollout.candidate_rerank_candidates = candidates
                rollout.candidate_rerank_metadata = metadata
                if self.heuristic_final_selector_refined:
                    chosen = self._refined_heuristic_override(question, rollout, candidates, metadata)
                else:
                    chosen = self._heuristic_select_candidate(question, rollout, candidates, metadata)
                if chosen:
                    rollout.heuristic_selected_candidate = chosen
                    rollout.final_answer = chosen
                    rollout.final_answer_source = "heuristic_selector_refined" if self.heuristic_final_selector_refined else "heuristic_selector"

        rerank_indices = [idx for idx, rollout in enumerate(rollouts) if rollout.steps and self.candidate_rerank]
        if rerank_indices:
            rerank_prompts: List[str] = []
            rerank_rollout_indices: List[int] = []
            for rollout_idx in rerank_indices:
                rollout = rollouts[rollout_idx]
                candidates, metadata = self._extract_trace_candidates(question, rollout)
                rollout.candidate_rerank_candidates = candidates
                rollout.candidate_rerank_metadata = metadata
                if not candidates:
                    continue
                if len(candidates) == 1:
                    rollout.final_answer = candidates[0]
                    rollout.final_answer_source = "candidate_rerank_singleton"
                    continue
                target_type = self._infer_target_type(question)
                rerank_rollout_indices.append(rollout_idx)
                rerank_prompts.append(self._build_candidate_selector_prompt(question, rollout.plan, rollout.steps, candidates, metadata, target_type))

            if rerank_prompts:
                rerank_generations = self._generate_batch(
                    rerank_prompts,
                    max_tokens=8,
                    temp=0.0,
                )
                for rollout_idx, (selector_text, _prompt_ids, _comp_ids) in zip(rerank_rollout_indices, rerank_generations):
                    rollout = rollouts[rollout_idx]
                    rollout.candidate_selector_output = selector_text
                    matched = self._match_candidate(selector_text, rollout.candidate_rerank_candidates)
                    chosen = matched or self._fallback_candidate(rollout.candidate_rerank_candidates)
                    if chosen:
                        rollout.final_answer = chosen
                        rollout.final_answer_source = "candidate_rerank"

        return rollouts
