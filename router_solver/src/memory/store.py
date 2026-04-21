"""Persistent plan store.

Spec: docs/07_plan_memory.md.
Start with a dict + numpy matrix for embeddings. Upgrade to FAISS if kNN gets slow.
Capacity 10k; FIFO eviction beyond that.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class MemoryEntry:
    question: str
    plan_json: str
    answer: int
    num_steps: int
    reward: float


class PlanStore:
    """In-memory store of (embedding, MemoryEntry) pairs."""

    def __init__(self, capacity: int = 10_000):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def add(self, embedding: np.ndarray, entry: MemoryEntry) -> None:
        """Append with FIFO eviction once capacity is hit."""
        raise NotImplementedError

    def topk(self, query_embedding: np.ndarray, k: int = 3, min_similarity: float = 0.5) -> list[MemoryEntry]:
        """Return up to k entries with cosine similarity >= min_similarity, sorted desc."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
