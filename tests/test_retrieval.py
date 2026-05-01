# tests/test_retrieval.py
"""
Regression tests for BM25, VectorStore, and HybridRetriever.
These are golden tests — if they break, something fundamental regressed.
"""
import pytest
import numpy as np
from pathlib import Path

from retrieval.bm25 import BM25Retriever
from retrieval.vector_store import VectorStore
from retrieval.hybrid import HybridRetriever
from embeddings.vectorizer import EmbeddingPipeline

NPZ_PATH = Path("data/kb_vectors.npz")


@pytest.fixture(scope="module")
def kb_data():
    data = np.load(NPZ_PATH, allow_pickle=True)
    return {
        "vectors": data["vectors"],
        "texts": data["texts"].tolist(),
        "metadata": data["metadata"].tolist(),
    }


@pytest.fixture(scope="module")
def bm25(kb_data):
    r = BM25Retriever()
    r.index(kb_data["texts"], kb_data["metadata"])
    return r


@pytest.fixture(scope="module")
def vector_store():
    store = VectorStore()
    store.load_from_npz(NPZ_PATH, index_type="flat")
    return store


@pytest.fixture(scope="module")
def hybrid():
    h = HybridRetriever()
    h.load(NPZ_PATH)
    return h


@pytest.fixture(scope="module")
def pipeline():
    return EmbeddingPipeline()


# --- BM25 ---

def test_bm25_exact_keyword_hits(bm25):
    """'async' is an exact token — async_patterns.md must rank in top 2."""
    results = bm25.search("async exception handling", k=3)
    sources = [r.source for r in results]
    assert any("async" in s for s in sources), (
        f"async_patterns.md not in top 3 for exact keyword query. Got: {sources}"
    )


def test_bm25_returns_k_results(bm25):
    results = bm25.search("python error", k=3)
    assert len(results) == 3


def test_bm25_semantic_query_weak(bm25):
    """
    'code crashes unexpectedly' has no token overlap with KB.
    All BM25 scores should be 0 or near-0.
    This is the known BM25 weakness — dense must compensate.
    """
    results = bm25.search("my code crashes unexpectedly", k=3)
    assert all(r.score == 0.0 for r in results), (
        f"Expected zero scores for semantic query, got: {[(r.source, r.score) for r in results]}"
    )


# --- VectorStore ---

def test_vector_store_top_result_async(vector_store, pipeline):
    """Semantic query about async errors → async_patterns.md or error_handling.md in top 2."""
    qvec = pipeline.embed_texts(["async exception handling python"])
    results = vector_store.search(qvec, k=3)
    sources = [r.source for r in results]
    assert any("async" in s or "error" in s for s in sources), (
        f"Expected async or error doc in top 3. Got: {sources}"
    )


def test_vector_store_scores_bounded(vector_store, pipeline):
    """Cosine similarity on normalized vectors must be in [-1, 1]."""
    qvec = pipeline.embed_texts(["random query about nothing specific"])
    results = vector_store.search(qvec, k=5)
    for r in results:
        assert -1.0 <= r.score <= 1.0, f"Score out of bounds: {r.score}"


def test_vector_store_invalid_shape(vector_store):
    """Wrong shape should raise ValueError, not crash silently."""
    bad_vec = np.random.rand(5, 384).astype(np.float32)
    with pytest.raises(ValueError):
        vector_store.search(bad_vec, k=3)


# --- Hybrid ---

def test_hybrid_keyword_query_uses_bm25(hybrid):
    """Exact keyword query → top result should have a bm25_rank assigned."""
    results = hybrid.search("async exception KeyError", k=3)
    assert results[0].bm25_rank is not None, (
        "Top result for keyword query should appear in BM25 results"
    )


def test_hybrid_semantic_query_uses_dense(hybrid):
    """Semantic query → top result should have a dense_rank assigned."""
    results = hybrid.search("my code crashes when something unexpected happens", k=3)
    assert results[0].dense_rank is not None, (
        "Top result for semantic query should appear in dense results"
    )


def test_hybrid_rrf_scores_positive(hybrid):
    results = hybrid.search("python security credential handling", k=3)
    for r in results:
        assert r.rrf_score > 0.0