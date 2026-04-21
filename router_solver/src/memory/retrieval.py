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
        """Retrieves past plans based on the current mode."""
        if self.mode == RetrievalMode.NONE:
            return []
            
        if self.mode == RetrievalMode.RANDOM:
            if not self.store.values:
                return []
            return random.sample(self.store.values, min(k, len(self.store.values)))
            
        if self.mode == RetrievalMode.SIMILARITY:
            query_embedding = self.embedder.embed(question)
            results = self.store.topk(query_embedding, k=k)
            return [res[0] for res in results]
            
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
