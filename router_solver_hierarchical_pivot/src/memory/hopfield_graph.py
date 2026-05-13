import hashlib
import json
import os
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


def _infer_target_type(question: str) -> str:
    lowered = str(question or "").lower()
    if any(token in lowered for token in ["altogether", "total", "in all", "sum", "combined"]):
        return "total"
    if any(token in lowered for token in ["left", "remain", "remaining", "still need", "more money", "difference", "how many more"]):
        return "difference_or_remaining"
    if any(token in lowered for token in ["each", "per", "rate", "hour", "hours", "every"]):
        return "rate_or_ratio"
    return "direct_answer"


def _infer_operation_signature(question: str, plan: Optional[List[str]] = None) -> str:
    text = " ".join([str(question or "")] + [str(step or "") for step in (plan or [])]).lower()
    ops: List[str] = []
    if any(token in text for token in ["times", "each", "per", "product", "rows of", "groups of"]):
        ops.append("mul")
    if any(token in text for token in ["left", "remain", "remaining", "difference", "more", "less", "spent", "after paying"]):
        ops.append("sub")
    if any(token in text for token in ["total", "altogether", "in all", "combined", "sum"]):
        ops.append("add")
    if any(token in text for token in ["share", "equally", "average", "half", "third", "quarter", "divide"]):
        ops.append("div")
    if not ops:
        ops.append("direct")
    seen: List[str] = []
    for op in ops:
        if op not in seen:
            seen.append(op)
    return "|".join(seen)


def _lexical_jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
    a = set(tokens_a)
    b = set(tokens_b)
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


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
        retrieval_pool_size: int = 8,
        hopfield_rerank_weight: float = 0.5,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        reranker_path: str = "",
        retrieval_gate_threshold: float = 0.0,
        retrieval_gate_coherence: float = 0.0,
        use_hopfield_readout: bool = True,
        use_learned_reranker: bool = True,
    ):
        self.embed_dim = embed_dim
        self.beta = beta
        self.retrieval_pool_size = retrieval_pool_size
        self.hopfield_rerank_weight = hopfield_rerank_weight
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self.embedder = None if embedding_model_name == "hash" else RetrievalTextEmbedder(embedding_model_name, device=embedding_device)
        self.reranker_path = reranker_path
        self.retrieval_gate_threshold = retrieval_gate_threshold
        self.retrieval_gate_coherence = retrieval_gate_coherence
        self.use_hopfield_readout = use_hopfield_readout
        self.use_learned_reranker = use_learned_reranker
        self.nodes: Dict[str, HeteroNode] = {}
        self.edges: List[HeteroEdge] = []
        self.case_ids: List[str] = []
        self.case_payloads: Dict[str, Dict[str, object]] = {}
        self.case_embeddings: List[torch.Tensor] = []
        self.reranker_spec = self._load_reranker(reranker_path)

    def fork(
        self,
        *,
        use_hopfield_readout: Optional[bool] = None,
        use_learned_reranker: Optional[bool] = None,
        retrieval_gate_threshold: Optional[float] = None,
        retrieval_gate_coherence: Optional[float] = None,
    ) -> "HeteroHopfieldMemory":
        other = HeteroHopfieldMemory(
            embed_dim=self.embed_dim,
            beta=self.beta,
            retrieval_pool_size=self.retrieval_pool_size,
            hopfield_rerank_weight=self.hopfield_rerank_weight,
            embedding_model_name=self.embedding_model_name,
            embedding_device=self.embedding_device,
            reranker_path=self.reranker_path,
            retrieval_gate_threshold=self.retrieval_gate_threshold if retrieval_gate_threshold is None else retrieval_gate_threshold,
            retrieval_gate_coherence=self.retrieval_gate_coherence if retrieval_gate_coherence is None else retrieval_gate_coherence,
            use_hopfield_readout=self.use_hopfield_readout if use_hopfield_readout is None else use_hopfield_readout,
            use_learned_reranker=self.use_learned_reranker if use_learned_reranker is None else use_learned_reranker,
        )
        other.embedder = self.embedder
        other.nodes = self.nodes
        other.edges = self.edges
        other.case_ids = self.case_ids
        other.case_payloads = self.case_payloads
        other.case_embeddings = self.case_embeddings
        other.reranker_spec = self.reranker_spec
        return other

    @property
    def num_cases(self) -> int:
        return len(self.case_ids)

    @staticmethod
    def _load_reranker(path: str) -> Optional[Dict[str, object]]:
        if not path:
            return None
        if not os.path.exists(path):
            raise FileNotFoundError(f"Reranker spec not found: {path}")
        with open(path, "r", encoding="ascii", errors="ignore") as f:
            spec = json.load(f)
        return spec

    def _candidate_feature_vector(
        self,
        query_tokens: List[str],
        query_target_type: str,
        base_score: float,
        hopfield_score: float,
        attention_weight: float,
        payload: Dict[str, object],
    ) -> List[float]:
        return [
            float(base_score),
            float(hopfield_score),
            float(attention_weight),
            float(_lexical_jaccard(query_tokens, payload.get("question_tokens", []))),
            1.0 if payload.get("target_type") == query_target_type else 0.0,
            float(payload.get("quality_score", 0.0)),
            1.0 if payload.get("has_answer_bearing_step") else 0.0,
        ]

    def _apply_reranker(self, features: List[float]) -> Optional[float]:
        if self.reranker_spec is None:
            return None
        weights = self.reranker_spec.get("weights", [])
        bias = float(self.reranker_spec.get("bias", 0.0))
        threshold = float(self.reranker_spec.get("threshold", 0.5))
        if len(weights) != len(features):
            return None
        logit = bias
        for weight, feature in zip(weights, features):
            logit += float(weight) * float(feature)
        score = 1.0 / (1.0 + torch.exp(torch.tensor(-logit, dtype=torch.float32)).item())
        return max(0.0, min(1.0, (score - threshold) / max(1e-6, 1.0 - threshold)))

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
        target_type = str(diagnostics.get("target_type") or _infer_target_type(question))
        operation_signature = str(diagnostics.get("operation_signature") or _infer_operation_signature(question, plan))
        answer_bearing_idx = diagnostics.get("answer_bearing_step_idx")
        has_answer_bearing_step = isinstance(answer_bearing_idx, int) and 0 <= answer_bearing_idx < len(subgoals)
        quality_score = 0.0
        if diagnostics.get("relaxed_match", False):
            quality_score += 1.0
        if has_answer_bearing_step:
            quality_score += 0.5

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
            "target_type": target_type,
            "operation_signature": operation_signature,
            "has_answer_bearing_step": has_answer_bearing_step,
            "quality_score": quality_score,
        }

    def retrieve(self, question: str, k: int = 3) -> List[Dict[str, object]]:
        if not self.case_embeddings:
            return []

        normalized_query = _normalize_text(question)
        query_tokens = _tokenize(question)
        query_target_type = _infer_target_type(question)
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
        top_pool = candidate_rows[: min(max(k, self.retrieval_pool_size), len(candidate_rows))]

        pool_indices = [idx for _score, idx in top_pool]
        pool_scores = torch.tensor([score for score, _idx in top_pool], dtype=torch.float32)
        pool_embeddings = torch.stack([self.case_embeddings[idx] for idx in pool_indices], dim=0)
        if self.use_hopfield_readout:
            attn = torch.softmax(self.beta * pool_scores, dim=0)
            hopfield_state = F.normalize((attn.unsqueeze(1) * pool_embeddings).sum(dim=0), dim=0)
        else:
            attn = torch.softmax(torch.zeros_like(pool_scores), dim=0)
            hopfield_state = F.normalize(query, dim=0)

        reranked_rows: List[Tuple[float, float, float, int, float]] = []
        for pool_rank, (base_score, idx) in enumerate(top_pool):
            hopfield_score = float(torch.dot(hopfield_state, self.case_embeddings[idx]))
            attn_weight = float(attn[pool_rank].item())
            payload = self.case_payloads[self.case_ids[idx]]
            features = self._candidate_feature_vector(
                query_tokens=query_tokens,
                query_target_type=query_target_type,
                base_score=float(base_score),
                hopfield_score=hopfield_score,
                attention_weight=attn_weight,
                payload=payload,
            )
            learned_score = self._apply_reranker(features) if self.use_learned_reranker else None
            rerank_score = learned_score if learned_score is not None else (
                ((self.hopfield_rerank_weight * float(base_score)) + ((1.0 - self.hopfield_rerank_weight) * hopfield_score))
                if self.use_hopfield_readout else float(base_score)
            )
            reranked_rows.append((rerank_score, float(base_score), hopfield_score, idx, attn_weight, features))
        reranked_rows.sort(key=lambda item: item[0], reverse=True)
        top_rows = reranked_rows[: min(k, len(reranked_rows))]
        if top_rows and self.retrieval_gate_threshold > 0.0 and top_rows[0][0] < self.retrieval_gate_threshold:
            return []
        if top_rows and self.retrieval_gate_coherence > 0.0:
            mean_coherence = sum(row[2] for row in top_rows[: min(3, len(top_rows))]) / float(min(3, len(top_rows)))
            if mean_coherence < self.retrieval_gate_coherence:
                return []

        results: List[Dict[str, object]] = []
        for rerank_score, base_score, hopfield_score, idx, attn_weight, features in top_rows:
            case_id = self.case_ids[idx]
            payload = dict(self.case_payloads[case_id])
            payload["score"] = rerank_score
            payload["base_score"] = base_score
            payload["hopfield_score"] = hopfield_score
            payload["hopfield_attention"] = attn_weight
            payload["reranker_features"] = features
            results.append(payload)
        return results

    @classmethod
    def from_rollout_traces(
        cls,
        trace_path: str,
        exclude_questions: Optional[Set[str]] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        reranker_path: str = "",
        retrieval_gate_threshold: float = 0.0,
        retrieval_gate_coherence: float = 0.0,
        use_hopfield_readout: bool = True,
        use_learned_reranker: bool = True,
    ) -> "HeteroHopfieldMemory":
        memory = cls(
            embedding_model_name=embedding_model_name,
            embedding_device=embedding_device,
            reranker_path=reranker_path,
            retrieval_gate_threshold=retrieval_gate_threshold,
            retrieval_gate_coherence=retrieval_gate_coherence,
            use_hopfield_readout=use_hopfield_readout,
            use_learned_reranker=use_learned_reranker,
        )
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
        reranker_path: str = "",
        retrieval_gate_threshold: float = 0.0,
        retrieval_gate_coherence: float = 0.0,
        use_hopfield_readout: bool = True,
        use_learned_reranker: bool = True,
    ) -> "HeteroHopfieldMemory":
        memory = cls(
            embedding_model_name=embedding_model_name,
            embedding_device=embedding_device,
            reranker_path=reranker_path,
            retrieval_gate_threshold=retrieval_gate_threshold,
            retrieval_gate_coherence=retrieval_gate_coherence,
            use_hopfield_readout=use_hopfield_readout,
            use_learned_reranker=use_learned_reranker,
        )
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
