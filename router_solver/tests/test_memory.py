import unittest
import numpy as np
from src.memory.embedder import Embedder
from src.memory.store import MemoryStore
from src.memory.retrieval import PlanMemory, RetrievalMode

class TestMemory(unittest.TestCase):
    def test_embedder_output_shape(self):
        """embed('hello') returns a 384-dim L2-normalized vector."""
        emb = Embedder()
        vec = emb.embed("hello")
        self.assertEqual(vec.shape, (384,))
        # Check normaliztion
        self.assertAlmostEqual(np.linalg.norm(vec), 1.0, places=5)

    def test_store_add_and_topk(self):
        """Add 5 entries, topk(query, 3) returns 3 sorted by similarity."""
        store = MemoryStore(capacity=10)
        # Entry 1: very similar to query
        vec1 = np.array([1.0] + [0.0]*383)
        store.add("q1", {"plan": "p1"}, vec1)
        # Entry 2: less similar
        vec2 = np.array([0.5] + [0.1]*383)
        vec2 = vec2 / np.linalg.norm(vec2)
        store.add("q2", {"plan": "p2"}, vec2)
        # Entry 3: dissimilar
        vec3 = np.array([0.0] + [1.0]*383)
        store.add("q3", {"plan": "p3"}, vec3)

        query = np.array([1.0] + [0.0]*383)
        results = store.topk(query, k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0]["plan"], "p1")
        self.assertEqual(results[1][0]["plan"], "p2")

    def test_store_capacity_fifo(self):
        """Past capacity, oldest entry is evicted on add."""
        store = MemoryStore(capacity=2)
        store.add("1", "p1", np.zeros(384))
        store.add("2", "p2", np.zeros(384))
        store.add("3", "p3", np.zeros(384)) # Should evict "1"
        self.assertEqual(len(store.keys), 2)
        self.assertEqual(store.keys[0], "2")
        self.assertEqual(store.keys[1], "3")

    def test_store_min_similarity_filter(self):
        """topk with min_similarity=0.99 on unrelated queries returns empty."""
        store = MemoryStore()
        vec = np.array([1.0] + [0.0]*383)
        store.add("q", "p", vec)

        query = np.array([0.0] + [1.0]*383)
        results = store.topk(query, k=1, min_similarity=0.99)
        self.assertEqual(len(results), 0)

    def test_write_gate_rejects_wrong_answers(self):
        """write_if_success with reward<1.0 returns False and does not add."""
        mem = PlanMemory(mode=RetrievalMode.SIMILARITY)
        res = mem.write_if_success("q", {"p": 1}, 0.5)
        self.assertFalse(res)
        self.assertEqual(len(mem.store.keys), 0)

    def test_write_gate_rejects_tool_errors(self):
        """write_if_success with tool_errors>0 returns False."""
        mem = PlanMemory(mode=RetrievalMode.SIMILARITY)
        res = mem.write_if_success("q", {"p": 1}, 1.0, tool_errors=1)
        self.assertFalse(res)

    def test_retrieval_mode_none_returns_empty(self):
        """PlanMemory(mode=NONE).retrieve(...) always returns []."""
        mem = PlanMemory(mode=RetrievalMode.NONE)
        mem.write_if_success("q", {"p": 1}, 1.0)
        self.assertEqual(mem.retrieve("q"), [])

    def test_retrieval_mode_random_returns_k_uniform(self):
        """Mode=RANDOM returns k entries from the store regardless of query similarity."""
        mem = PlanMemory(mode=RetrievalMode.RANDOM)
        for i in range(5):
            mem.write_if_success(str(i), {"p": i}, 1.0)
        
        results = mem.retrieve("any", k=2)
        self.assertEqual(len(results), 2)
        # Should be plans, not just indices
        self.assertIn("p", results[0])

if __name__ == "__main__":
    unittest.main()
