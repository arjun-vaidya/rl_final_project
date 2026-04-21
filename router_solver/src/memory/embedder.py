import numpy as np

class Embedder:
    """Simple embedder for similarity-based retrieval."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.dim = 384
        # We don't load the real model here to keep tests lightweight
        # In a real setup, we'd use sentence-transformers
        pass

    def embed(self, text: str) -> np.ndarray:
        """Returns a dummy L2-normalized vector for testing."""
        # Use a hash-based seed for reproducibility in tests
        import hashlib
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(h % (2**32))
        vec = np.random.randn(self.dim)
        return vec / np.linalg.norm(vec)
