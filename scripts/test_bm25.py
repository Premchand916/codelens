# scripts/test_bm25.py
"""
Test BM25 with exact keyword query vs semantic query.
Watch where BM25 wins and where dense wins.
"""
import sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)

import numpy as np
from retrieval.bm25 import BM25Retriever

data = np.load(Path("data/kb_vectors.npz"), allow_pickle=True)
texts = data["texts"].tolist()
metadata = data["metadata"].tolist()

retriever = BM25Retriever()
retriever.index(texts, metadata)

# Query 1: exact keyword — BM25 should dominate
q1 = "KeyError exception async"
# Query 2: semantic — BM25 will struggle
q2 = "my code crashes when something unexpected happens"

for q in [q1, q2]:
    print(f"\nQuery: {q}")
    for r in retriever.search(q, k=3):
        print(f"  Rank {r.rank} | {r.score:.4f} | {r.source}")