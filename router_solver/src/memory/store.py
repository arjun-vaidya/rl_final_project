import numpy as np
from typing import List, Dict, Tuple

class MemoryStore:
    """Storage for successful plans and their embeddings."""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.keys: List[str] = [] # Original questions
        self.values: List[Dict] = [] # Plans
        self.embeddings: List[np.ndarray] = []

    def add(self, question: str, plan: Dict, embedding: np.ndarray):
        """Adds a new entry. Evicts oldest if capacity exceeded (FIFO)."""
        if len(self.keys) >= self.capacity:
            self.keys.pop(0)
            self.values.pop(0)
            self.embeddings.pop(0)
        
        self.keys.append(question)
        self.values.append(plan)
        self.embeddings.append(embedding)

    def topk(self, query_embedding: np.ndarray, k: int = 3, min_similarity: float = 0.0) -> List[Tuple[Dict, float]]:
        """Returns top-k most similar entries based on cosine similarity."""
        if not self.embeddings:
            return []
            
        # Cosine similarity for normalized vectors is just dot product
        sims = np.array([np.dot(query_embedding, e) for e in self.embeddings])
        
        # Filter by min_similarity
        mask = sims >= min_similarity
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            return []
            
        # Sort by similarity descending
        sorted_indices = indices[np.argsort(sims[indices])[::-1]]
        
        results = []
        for i in sorted_indices[:k]:
            results.append((self.values[i], sims[i]))
        return results
