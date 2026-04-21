from typing import List, Dict, Optional
from enum import Enum
import random
from src.memory.store import MemoryStore
from src.memory.embedder import Embedder

class RetrievalMode(Enum):
    NONE = "none"
    RANDOM = "random"
    SIMILARITY = "similarity"

class PlanMemory:
    """High-level interface for plan memory retrieval and writing."""
    def __init__(self, mode: RetrievalMode = RetrievalMode.NONE, capacity: int = 1000):
        self.mode = mode
        self.store = MemoryStore(capacity=capacity)
        self.embedder = Embedder()

    def retrieve(self, question: str, k: int = 3) -> List[Dict]:
        """Retrieves past (question, plan) entries based on the current mode.

        Returns a list of dicts: {"question": str, "plan": Dict}. The Router
        prompt renders both per docs/04_design.md and docs/07_plan_memory.md.
        """
        if self.mode == RetrievalMode.NONE:
            return []

        if self.mode == RetrievalMode.RANDOM:
            if not self.store.values:
                return []
            indices = random.sample(range(len(self.store.values)), min(k, len(self.store.values)))
            return [
                {"question": self.store.keys[i], "plan": self.store.values[i]}
                for i in indices
            ]

        if self.mode == RetrievalMode.SIMILARITY:
            query_embedding = self.embedder.embed(question)
            results = self.store.topk(query_embedding, k=k)
            # Map back from value to stored question by identity.
            out = []
            for value, _sim in results:
                try:
                    idx = self.store.values.index(value)
                    past_q = self.store.keys[idx]
                except ValueError:
                    past_q = ""
                out.append({"question": past_q, "plan": value})
            return out

        return []

    def write_if_success(self, question: str, plan_dict: Dict, reward: float, tool_errors: int = 0) -> bool:
        """
        Writes a plan to memory if the episode was successful.
        Gated by reward=1.0 and no tool errors.
        """
        if reward < 1.0 or tool_errors > 0:
            return False
            
        embedding = self.embedder.embed(question)
        self.store.add(question, plan_dict, embedding)
        return True
