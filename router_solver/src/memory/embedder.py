"""Question embedder for Plan Memory.

Spec: docs/07_plan_memory.md.
Model: sentence-transformers/all-MiniLM-L6-v2 (22M, 384-dim, CPU-fast).
"""
import numpy as np


class Embedder:
    """Thin wrapper around sentence-transformers for question embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        raise NotImplementedError

    def embed(self, text: str) -> np.ndarray:
        """Returns a 384-dim L2-normalized vector."""
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Returns an (N, 384) L2-normalized matrix."""
        raise NotImplementedError
