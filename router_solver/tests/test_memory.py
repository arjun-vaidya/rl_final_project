"""Tests for src/memory/*."""


def test_embedder_output_shape():
    """embed('hello') returns a 384-dim L2-normalized vector."""
    raise NotImplementedError


def test_store_add_and_topk():
    """Add 5 entries, topk(query, 3) returns 3 sorted by similarity."""
    raise NotImplementedError


def test_store_capacity_fifo():
    """Past capacity, oldest entry is evicted on add."""
    raise NotImplementedError


def test_store_min_similarity_filter():
    """topk with min_similarity=0.99 on unrelated queries returns empty."""
    raise NotImplementedError


def test_write_gate_rejects_wrong_answers():
    """write_if_success with reward<1.0 returns False and does not add."""
    raise NotImplementedError


def test_write_gate_rejects_tool_errors():
    """write_if_success with tool_errors>0 returns False."""
    raise NotImplementedError


def test_retrieval_mode_none_returns_empty():
    """PlanMemory(mode=NONE).retrieve(...) always returns []."""
    raise NotImplementedError


def test_retrieval_mode_random_returns_k_uniform():
    """Mode=RANDOM returns k entries from the store regardless of query similarity."""
    raise NotImplementedError
