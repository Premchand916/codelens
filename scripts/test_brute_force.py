# scripts/test_brute_force.py
"""
Brute force retrieval using only math_utils + .npz vectors.
Purpose: feel the O(n) scan before FAISS abstracts it away.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.math_utils import cosine_similarity_matrix, top_k_indices
from embeddings.vectorizer import EmbeddingPipeline


def metadata_source(metadata_item: dict | str) -> str:
    if isinstance(metadata_item, dict):
        return str(metadata_item.get("source", "unknown"))
    return str(metadata_item)

def brute_force_search(query: str, npz_path: Path, k: int = 3) -> list[dict]:
    """
    Query → embed → cosine similarity against ALL stored vectors → top-k.
    
    # [SURFACE] Scans every stored vector, computes similarity, returns top-k.
    # [INTERNAL] For n=6 docs and 384-dim vectors: 6 × 384 = 2,304 multiplications
    # per query. At n=1,000,000 that's 384 million multiplications — every query.
    # No indexing, no shortcuts. This is why we need FAISS.
    """
    pipeline = EmbeddingPipeline()
    
    # Load stored vectors + metadata
    data = np.load(npz_path, allow_pickle=True)
    stored_vectors = data["vectors"]      # shape: (n_docs, 384)
    texts = data["texts"].tolist()
    metadata = data["metadata"].tolist()
    
    # Embed the query — shape: (1, 384)
    query_vector = pipeline.embed_batch([query])
    
    # [INTERNAL] cosine_similarity_matrix: (384,) × (384, n) = (n,)
    # Each value = cosine similarity between query and one stored doc.
    # We built this from scratch in Session 3 — FAISS will replace this call.
    scores = cosine_similarity_matrix(query_vector, stored_vectors)
    
    indices = top_k_indices(scores, k=k)
    
    results = []
    for idx in indices:
        results.append({
            "rank": len(results) + 1,
            "score": float(scores[idx]),
            "file": metadata_source(metadata[idx]),
            "text_preview": texts[idx][:120] + "..."
        })
    return results


if __name__ == "__main__":
    npz_path = Path("data/kb_vectors.npz")
    query = "How should I handle exceptions in Python async code?"
    
    print(f"Query: {query}\n")
    results = brute_force_search(query, npz_path, k=3)
    
    for r in results:
        print(f"Rank {r['rank']} | Score: {r['score']:.4f} | File: {r['file']}")
        print(f"  Preview: {r['text_preview']}\n")
