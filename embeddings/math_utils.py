"""
math_utils.py — Raw vector math for embeddings.

Why build this before using sentence-transformers?
Because when an interviewer asks "how does cosine similarity work,"
you need to have implemented it, not just called it.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def normalize(vector: np.ndarray) -> np.ndarray:
    """
    L2-normalize a vector: divide by its magnitude.
    
    # [INTERNAL] Magnitude = sqrt(sum of squares) = ||v||
    # After normalization, ||v|| = 1.0 (unit vector).
    # WHY: Cosine similarity on raw vectors is affected by magnitude.
    # A long vector and short vector pointing the same direction
    # would score differently without normalization. We only care
    # about DIRECTION (semantic meaning), not magnitude (word count).
    """
    norm = np.linalg.norm(vector)  # sqrt(v[0]^2 + v[1]^2 + ... + v[n]^2)
    if norm == 0:
        logger.warning("Zero vector encountered during normalization")
        return vector  # can't normalize a zero vector — return as-is
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    
    # [INTERNAL] cos(θ) = (a · b) / (||a|| * ||b||)
    # dot product = sum(a[i] * b[i]) — how much a and b point together
    # If both are already normalized (||a|| = ||b|| = 1),
    # then denominator = 1, and cosine_similarity = just the dot product.
    # Range: -1 (opposite) → 0 (orthogonal) → 1 (identical direction)
    # For text embeddings, range is effectively 0 → 1.
    #
    # [INTERVIEW] "Why cosine over euclidean for text?"
    # → "Euclidean measures geometric distance — sensitive to vector length.
    #    'Python is great' and 'Python is great Python is great' would be
    #    far apart in euclidean space but identical in meaning.
    #    Cosine only cares about direction = semantic angle = meaning."
    """
    a_norm = normalize(a)
    b_norm = normalize(b)
    # clip to [-1, 1] — floating point arithmetic can produce 1.0000002
    return float(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))


def cosine_similarity_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query and N corpus vectors.
    
    Args:
        query:  shape (D,)      — single query vector
        corpus: shape (N, D)    — N stored vectors
    
    Returns: shape (N,) — similarity score for each corpus vector
    
    # [INTERNAL] Instead of calling cosine_similarity N times in a loop,
    # we normalize the entire corpus matrix at once, then do ONE matrix
    # multiply: query (1, D) @ corpus.T (D, N) = (1, N) scores.
    # This is 10-100x faster than a Python loop for large N.
    # NumPy dispatches this to BLAS — runs on SIMD CPU instructions.
    """
    query = np.asarray(query)
    if query.ndim == 2:
        if query.shape[0] != 1:
            raise ValueError(f"Expected one query vector, got shape {query.shape}")
        query = query[0]
    if query.ndim != 1:
        raise ValueError(f"Expected query shape (D,) or (1, D), got {query.shape}")

    query_norm = normalize(query)
    # Normalize each row of corpus matrix
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)  # shape (N, 1)
    norms = np.where(norms == 0, 1, norms)                 # avoid division by zero
    corpus_norm = corpus / norms                            # shape (N, D)
    
    scores = corpus_norm @ query_norm  # matrix multiply → shape (N,)
    return np.clip(scores, -1.0, 1.0)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of top-k scores (descending order).
    
    # [INTERNAL] np.argsort returns indices that would sort the array.
    # [-k:] grabs last k elements (highest scores).
    # [::-1] reverses to descending order.
    # This is O(n log n) — fine for our knowledge base size (<1000 docs).
    """
    if k >= len(scores):
        k = len(scores)
    return np.argsort(scores)[-k:][::-1]
