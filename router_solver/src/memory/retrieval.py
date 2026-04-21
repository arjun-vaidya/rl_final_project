"""End-to-end retrieval API exposed to the Router agent.

Spec: docs/07_plan_memory.md.
Encapsulates (embedder, store, write-gate, random-vs-similarity mode) behind one class.
"""
from enum import Enum
from .embedder import Embedder
from .store import MemoryEntry, PlanStore


class RetrievalMode(str, Enum):
    NONE = "none"                 # condition #4a: no retrieval
    RANDOM = "random"             # condition #4b: random sample from store
    SIMILARITY = "similarity"     # condition #4c: top-K cosine


class PlanMemory:
    """The object RouterSolverAgent holds as `self.memory`."""

    def __init__(
        self,
        mode: RetrievalMode = RetrievalMode.SIMILARITY,
        top_k: int = 3,
        min_similarity: float = 0.5,
        capacity: int = 10_000,
    ):
        raise NotImplementedError

    def retrieve(self, question: str) -> list[MemoryEntry]:
        """Return top-K relevant past plans for this question (empty list if cold/NONE)."""
        raise NotImplementedError

    def write_if_success(
        self,
        question: str,
        plan_json: str,
        answer: int,
        num_steps: int,
        reward: float,
        tool_errors: int,
    ) -> bool:
        """Write gate: reward==1.0 AND valid plan AND tool_errors==0. Returns True if written."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
