# scripts/test_hybrid.py
"""
Show RRF fusion in action — compare all three retrievers side by side.
"""
import sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)

from retrieval.hybrid import HybridRetriever

retriever = HybridRetriever()
retriever.load(Path("data/kb_vectors.npz"))

queries = [
    "KeyError exception async",           # BM25 should win
    "my code crashes when something unexpected happens",  # Dense should win
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"Query: {q}")
    print(f"{'='*60}")
    results = retriever.search(q, k=3)
    for r in results:
        print(
            f"Rank {r.rank} | RRF={r.rrf_score:.4f} | "
            f"BM25_rank={r.bm25_rank} | Dense_rank={r.dense_rank} | "
            f"{r.source}"
        )