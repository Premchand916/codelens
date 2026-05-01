# scripts/test_vector_store.py
"""Compare brute force vs FAISS Flat vs FAISS IVF — same query, same results?"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)

from embeddings.vectorizer import EmbeddingPipeline
from retrieval.vector_store import VectorStore

def test(index_type: str):
    pipeline = EmbeddingPipeline()
    store = VectorStore()
    store.load_from_npz(Path("data/kb_vectors.npz"), index_type=index_type)

    query = "How should I handle exceptions in Python async code?"
    qvec = pipeline.embed_batch([query])

    results = store.search(qvec, k=3)
    print(f"\n--- {index_type.upper()} ---")
    for r in results:
        print(f"Rank {r.rank} | {r.score:.4f} | {r.source}")

test("flat")
test("ivf")
