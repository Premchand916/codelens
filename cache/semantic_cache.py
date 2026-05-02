# cache/semantic_cache.py
"""
Semantic cache for CodeLens.

Instead of exact-match caching (cache["same string"] → hit),
we embed the query and check if any stored query is *close enough*
in vector space. This catches paraphrases and near-duplicates.

Why it matters: Two PRs with the same bug pattern generate near-identical
LLM prompts. Cache hit = skip LLM entirely = 0 tokens spent.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Embedding-based cache with cosine similarity threshold.

    [SURFACE] Stores (query_vector, result) pairs. On lookup,
    finds nearest stored query; if similarity >= threshold → cache hit.

    [INTERNAL] Every cache entry has a 384-dim embedding (same model as
    the retrieval pipeline). Lookup = cosine similarity against all stored
    embeddings (brute force — cache size stays small). Similarity of 1.0
    = identical direction, 0.0 = orthogonal, -1.0 = opposite.
    Threshold 0.85 = "similar enough to reuse" without false hits.

    [INTERVIEW] "Why 0.85 and not 0.99?"
    → 0.99 only catches exact rephrases — cache is useless.
      0.70 catches unrelated queries — wrong results served.
      0.85-0.90 is the empirical sweet spot for code review prompts.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        max_size: int = 500,
        persist_path: Path | None = None,
    ) -> None:
        self.threshold = threshold
        self.max_size = max_size
        self.persist_path = persist_path

        # Parallel arrays: index i in _vectors corresponds to index i in _entries
        self._vectors: list[np.ndarray] = []   # shape: (384,) each
        self._entries: list[dict[str, Any]] = []

        # Metrics — exposed on /metrics endpoint later
        self.hits = 0
        self.misses = 0

        if persist_path and persist_path.exists():
            self._load(persist_path)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, query_vector: np.ndarray) -> dict[str, Any] | None:
        """
        Lookup. Returns cached result if similarity >= threshold, else None.

        O(n) scan — acceptable because cache size is bounded (max_size=500).
        Above ~10K entries, switch to FAISS IndexFlatIP for this step.
        """
        if not self._vectors:
            self.misses += 1
            return None

        # Stack all stored vectors into matrix → one batched similarity call
        # Shape: (n_entries, 384)
        matrix = np.stack(self._vectors)

        # [INTERNAL] Cosine similarity = dot product of *normalized* vectors.
        # query_vector and stored vectors are already L2-normalized (unit vectors),
        # so dot product == cosine similarity directly. No sqrt needed.
        similarities = matrix @ query_vector  # shape: (n_entries,)

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.threshold:
            self.hits += 1
            logger.info(
                "Cache HIT | similarity=%.4f | threshold=%.2f",
                best_score,
                self.threshold,
            )
            entry = self._entries[best_idx]
            entry["_cache_similarity"] = best_score  # attach for /metrics
            return entry

        self.misses += 1
        logger.info(
            "Cache MISS | best_similarity=%.4f | threshold=%.2f",
            best_score,
            self.threshold,
        )
        return None

    def set(
        self,
        query_vector: np.ndarray,
        result: dict[str, Any],
        query_text: str = "",
    ) -> None:
        """Store a result. Evicts oldest entry if at capacity (FIFO)."""
        if len(self._vectors) >= self.max_size:
            # FIFO eviction — simplest policy, good enough for a 500-entry cache
            self._vectors.pop(0)
            self._entries.pop(0)
            logger.debug("Cache eviction (FIFO) — max_size=%d reached", self.max_size)

        # Normalize before storing so dot product == cosine similarity at lookup
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        self._vectors.append(query_vector)
        self._entries.append(
            {
                **result,
                "_cached_at": time.time(),
                "_query_text": query_text[:200],  # truncated for debugging
            }
        )

        if self.persist_path:
            self._save(self.persist_path)

    def invalidate(self) -> None:
        """Clear entire cache (e.g., after knowledge base update)."""
        self._vectors.clear()
        self._entries.clear()
        logger.info("Cache invalidated")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._vectors)

    def stats(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "size": self.size,
            "threshold": self.threshold,
            "max_size": self.max_size,
        }

    # ------------------------------------------------------------------
    # Persistence (optional)
    # ------------------------------------------------------------------

    def _save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path.with_suffix(".npy")), np.stack(self._vectors))
        path.with_suffix(".json").write_text(
            json.dumps(self._entries, default=str)
        )

    def _load(self, path: Path) -> None:
        vec_path = path.with_suffix(".npy")
        json_path = path.with_suffix(".json")
        if vec_path.exists() and json_path.exists():
            vecs = np.load(str(vec_path))
            self._vectors = [vecs[i] for i in range(len(vecs))]
            self._entries = json.loads(json_path.read_text())
            logger.info("Cache loaded: %d entries from %s", len(self._vectors), path)