import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from src.memory.text_embedder import RetrievalTextEmbedder


def _hash_embed(text: str, dim: int = 384) -> torch.Tensor:
    text = str(text or "")
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    vec = torch.zeros(dim, dtype=torch.float32)
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        sign = 1.0 if (int(digest[8:16], 16) % 2 == 0) else -1.0
        vec[idx] += sign
    if torch.count_nonzero(vec) == 0:
        vec[0] = 1.0
    return F.normalize(vec, dim=0)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


@dataclass
class HeteroNode:
    node_id: str
    node_type: str
    text: str
    embedding: torch.Tensor
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class HeteroEdge:
    src: str
    dst: str
    edge_type: str
    weight: float = 1.0


class HeteroHopfieldMemory:
    def __init__(
        self,
        embed_dim: int = 384,
        beta: float = 8.0,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
    ):
        self.embed_dim = embed_dim
        self.beta = beta
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self.embedder = None if embedding_model_name == "hash" else RetrievalTextEmbedder(embedding_model_name, device=embedding_device)
        self.nodes: Dict[str, HeteroNode] = {}
        self.edges: List[HeteroEdge] = []
        self.case_ids: List[str] = []
        self.case_payloads: Dict[str, Dict[str, object]] = {}
        self.case_embeddings: List[torch.Tensor] = []

    @property
    def num_cases(self) -> int:
        return len(self.case_ids)

    def _add_node(self, node_id: str, node_type: str, text: str, metadata: Optional[Dict[str, object]] = None) -> None:
        embedding = self.embedder.encode_one(text, role="doc") if self.embedder is not None else _hash_embed(text, self.embed_dim)
        self.nodes[node_id] = HeteroNode(
            node_id=node_id,
            node_type=node_type,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )

    def _add_edge(self, src: str, dst: str, edge_type: str, weight: float = 1.0) -> None:
        self.edges.append(HeteroEdge(src=src, dst=dst, edge_type=edge_type, weight=weight))

    def add_case(
        self,
        question: str,
        plan: List[str],
        final_answer: str,
        step_answers: Optional[List[str]] = None,
        diagnostics: Optional[Dict[str, object]] = None,
    ) -> None:
        case_idx = len(self.case_ids)
        case_id = f"case_{case_idx}"
        q_id = f"{case_id}:Q"
        p_id = f"{case_id}:P"
        a_id = f"{case_id}:A"
        d_id = f"{case_id}:D"

        self._add_node(q_id, "Q", question)
        self._add_node(p_id, "P", "\n".join(plan), {"plan": plan})
        self._add_node(a_id, "A", str(final_answer), {"final_answer": final_answer})
        diagnostics = diagnostics or {}
        diagnostics_text = (
            f"source={diagnostics.get('final_answer_source', 'unknown')} "
            f"exact={diagnostics.get('exact_match', False)} "
            f"relaxed={diagnostics.get('relaxed_match', False)} "
            f"steps={diagnostics.get('num_steps', 0)} "
            f"answer_bearing_step={diagnostics.get('answer_bearing_step_idx', 'none')} "
            f"synthesis_rejected={diagnostics.get('synthesis_rejected_by_consistency', False)}"
        )
        self._add_node(d_id, "D", diagnostics_text, diagnostics)
        self._add_edge(q_id, p_id, "Q_TO_P")
        self._add_edge(p_id, a_id, "P_TO_A")
        self._add_edge(q_id, a_id, "Q_TO_A")
        self._add_edge(q_id, d_id, "Q_TO_D")
        self._add_edge(p_id, d_id, "P_TO_D")
        self._add_edge(d_id, a_id, "D_TO_A", weight=0.5)

        subgoals: List[str] = []
        for step_idx, subgoal in enumerate(plan):
            s_id = f"{case_id}:S:{step_idx}"
            self._add_node(s_id, "S", subgoal, {"step_idx": step_idx})
            self._add_edge(p_id, s_id, "P_TO_S")
            if step_idx > 0:
                prev_id = f"{case_id}:S:{step_idx-1}"
                self._add_edge(prev_id, s_id, "S_TO_S")
            subgoals.append(subgoal)

        if step_answers:
            for step_idx, answer in enumerate(step_answers[: len(plan)]):
                t_id = f"{case_id}:T:{step_idx}"
                self._add_node(t_id, "T", str(answer), {"step_idx": step_idx})
                s_id = f"{case_id}:S:{step_idx}"
                self._add_edge(s_id, t_id, "S_TO_T")
                self._add_edge(t_id, a_id, "T_TO_A", weight=0.5)

        question_node = self.nodes[q_id].embedding
        plan_node = self.nodes[p_id].embedding
        step_nodes = [self.nodes[f"{case_id}:S:{step_idx}"].embedding for step_idx in range(len(subgoals))]
        step_mean = torch.stack(step_nodes).mean(dim=0) if step_nodes else torch.zeros_like(question_node)
        case_embedding = F.normalize((0.7 * question_node) + (0.2 * plan_node) + (0.1 * step_mean), dim=0)

        self.case_ids.append(case_id)
        self.case_embeddings.append(case_embedding)
        self.case_payloads[case_id] = {
            "question": question,
            "normalized_question": _normalize_text(question),
            "question_tokens": _tokenize(question),
            "plan": plan,
            "final_answer": final_answer,
            "subgoals": subgoals,
            "diagnostics": diagnostics,
        }

    def retrieve(self, question: str, k: int = 3) -> List[Dict[str, object]]:
        if not self.case_embeddings:
            return []

        normalized_query = _normalize_text(question)
        query = self.embedder.encode_one(question, role="query") if self.embedder is not None else _hash_embed(question, self.embed_dim)
        memory_matrix = torch.stack(self.case_embeddings)
        sims = torch.mv(memory_matrix, query)
        candidate_rows: List[Tuple[float, int]] = []
        for idx, case_id in enumerate(self.case_ids):
            payload = self.case_payloads[case_id]
            if payload.get("normalized_question") == normalized_query:
                continue
            candidate_rows.append((float(sims[idx]), idx))

        if not candidate_rows:
            return []

        candidate_rows.sort(key=lambda item: item[0], reverse=True)
        top_rows = candidate_rows[: min(k, len(candidate_rows))]

        results: List[Dict[str, object]] = []
        for score, idx in top_rows:
            case_id = self.case_ids[idx]
            payload = dict(self.case_payloads[case_id])
            payload["score"] = score
            results.append(payload)
        return results

    @classmethod
    def from_rollout_traces(
        cls,
        trace_path: str,
        exclude_questions: Optional[Set[str]] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
    ) -> "HeteroHopfieldMemory":
        memory = cls(embedding_model_name=embedding_model_name, embedding_device=embedding_device)
        excluded = {_normalize_text(q) for q in (exclude_questions or set())}
        with open(trace_path, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                question = record.get("question", "")
                if _normalize_text(question) in excluded:
                    continue
                for rollout in record.get("rollouts", []):
                    if not rollout.get("valid", False):
                        continue
                    if not rollout.get("relaxed_match", False):
                        continue
                    plan = rollout.get("plan") or []
                    if not plan:
                        continue
                    step_answers = [step.get("answer", "") for step in rollout.get("steps", [])]
                    memory.add_case(
                        question=question,
                        plan=plan,
                        final_answer=rollout.get("final_answer", ""),
                        step_answers=step_answers,
                        diagnostics={
                            "final_answer_source": rollout.get("final_answer_source", ""),
                            "exact_match": rollout.get("exact_match", False),
                            "relaxed_match": rollout.get("relaxed_match", False),
                            "num_steps": len(rollout.get("steps", [])),
                            "answer_bearing_step_idx": rollout.get("answer_bearing_step_idx"),
                            "synthesis_rejected_by_consistency": rollout.get("synthesis_rejected_by_consistency", False),
                        },
                    )
        return memory

    @classmethod
    def from_corpus_json(
        cls,
        corpus_json: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
    ) -> "HeteroHopfieldMemory":
        memory = cls(embedding_model_name=embedding_model_name, embedding_device=embedding_device)
        with open(corpus_json, "r", encoding="ascii", errors="ignore") as f:
            corpus = json.load(f)
        for doc in corpus.get("docs", []):
            if not doc.get("use_as_positive", True):
                continue
            memory.add_case(
                question=doc.get("question", ""),
                plan=doc.get("plan") or [],
                final_answer=doc.get("final_answer", ""),
                diagnostics={
                    "difficulty_bucket": doc.get("difficulty_bucket", "unknown"),
                    "target_type": doc.get("target_type", "unknown"),
                    "operation_signature": doc.get("operation_signature", "unknown"),
                    "quality_reason": doc.get("quality_reason", "unknown"),
                    "source_q_idx": doc.get("source_q_idx"),
                },
            )
        return memory
