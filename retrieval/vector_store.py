# retrieval/vector_store.py
"""
FAISS-backed vector store for CodeLens knowledge base retrieval.
Progression: IndexFlatL2 (exact, brute) → IndexIVFFlat (approximate, clustered).
"""
import logging
import numpy as np
import faiss
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_IVF_TRAINING_POINTS_PER_CLUSTER = 39


@dataclass
class RetrievalResult:
    rank: int
    score: float          # cosine similarity (0-1), higher = better
    source: str           # file path in knowledge base
    text: str             # full chunk text


class VectorStore:
    """
    Wraps FAISS index with load/search/metadata management.
    
    Why FAISS instead of our math_utils?
    - FAISS is written in C++ with SIMD (vectorized CPU instructions).
    - Same O(n) for Flat, but ~10-50x faster per operation than pure NumPy.
    - IVF changes the algorithm: O(n) → O(sqrt(n)) approximately.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index: faiss.Index | None = None
        self.texts: list[str] = []
        self.metadata: list[dict | str] = []

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_flat(self, vectors: np.ndarray) -> None:
        """
        IndexFlatIP: exact inner product search on L2-normalized vectors.
        
        # [SURFACE] Stores all vectors, checks every one at query time.
        # [INTERNAL] We use IndexFlatIP (inner product) NOT IndexFlatL2.
        # On L2-normalized vectors: inner_product == cosine_similarity.
        # Proof: cos(q,k) = (q·k) / (|q||k|). If |q|=|k|=1, = q·k.
        # Our EmbeddingPipeline already normalizes — so IP = cosine. Exact.
        # [INTERVIEW] "Why not IndexFlatL2?"
        # → "L2 distance and cosine similarity give the same ranking on
        #    normalized vectors, but IP scores are directly interpretable
        #    as cosine similarity (0-1). L2 scores are distances (lower=better),
        #    which is confusing to work with."
        """
        vectors = vectors.astype(np.float32)  # FAISS requires float32
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        logger.info(f"Built FlatIP index: {self.index.ntotal} vectors, {self.dim} dims")

    def build_ivf(self, vectors: np.ndarray, n_clusters: int | None = None) -> None:
        """
        IndexIVFFlat: cluster vectors → search only top clusters at query time.
        
        # [SURFACE] Splits vectors into clusters; searches only the nearest ones.
        # [INTERNAL] Training phase: k-means clusters all vectors into n_clusters
        # centroids. Each vector is assigned to its nearest centroid.
        # Query phase: embed query → find nearest nprobe centroids → search
        # only vectors in those clusters. Skips (n_clusters - nprobe)/n_clusters
        # fraction of the index entirely.
        # At 1M vectors, 1000 clusters, nprobe=10: searches ~10K vectors, not 1M.
        # The risk: if your answer lives in cluster 11 and nprobe=10, you miss it.
        # That's the "approximate" tradeoff. More nprobe = slower but more accurate.
        # [INTERVIEW] "FAISS IVF searched 3 clusters but the answer was in cluster 4?"
        # → "That's the ANN tradeoff. Increase nprobe. Or use HNSW which is graph-
        #    based and handles this better at the cost of higher memory."
        """
        n = len(vectors)
        if n_clusters is None:
            # Rule of thumb: sqrt(n) clusters, min 2 (IVF needs ≥ n_clusters vectors)
            n_clusters = max(2, int(np.sqrt(n)))
        
        # IVF requires at least n_clusters training vectors
        if n < n_clusters:
            logger.warning(
                f"Too few vectors ({n}) for {n_clusters} clusters. "
                "Falling back to FlatIP."
            )
            self.build_flat(vectors)
            return
        min_training_points = MIN_IVF_TRAINING_POINTS_PER_CLUSTER * n_clusters
        if n < min_training_points:
            logger.warning(
                f"Too few vectors ({n}) to train IVF reliably with {n_clusters} "
                f"clusters. FAISS recommends at least {min_training_points}; "
                "falling back to FlatIP."
            )
            self.build_flat(vectors)
            return

        vectors = vectors.astype(np.float32)
        
        # Quantizer decides HOW to measure distance between query and centroids
        quantizer = faiss.IndexFlatIP(self.dim)
        
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
        
        # Training: runs k-means to find cluster centroids
        # Must happen before add()
        self.index.train(vectors)
        self.index.add(vectors)
        
        # nprobe: how many clusters to search at query time
        # nprobe=1: fastest, most approximate
        # nprobe=n_clusters: same as Flat (exact), slowest
        self.index.nprobe = max(1, n_clusters // 4)
        
        logger.info(
            f"Built IVFFlat index: {self.index.ntotal} vectors, "
            f"{n_clusters} clusters, nprobe={self.index.nprobe}"
        )

    # ------------------------------------------------------------------
    # Load from .npz
    # ------------------------------------------------------------------

    def load_from_npz(self, npz_path: Path, index_type: str = "flat") -> None:
        """Load KB vectors from Session 3's .npz and build FAISS index."""
        data = np.load(npz_path, allow_pickle=True)
        vectors = data["vectors"]
        self.texts = data["texts"].tolist()
        self.metadata = data["metadata"].tolist()

        if index_type == "ivf":
            self.build_ivf(vectors)
        else:
            self.build_flat(vectors)

        logger.info(f"Loaded {len(self.texts)} docs from {npz_path}")

    def _source_for_index(self, idx: int) -> str:
        """Return source path for both legacy dict metadata and string metadata."""
        metadata = self.metadata[idx]
        if isinstance(metadata, dict):
            return str(metadata.get("source", "unknown"))
        return str(metadata)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[RetrievalResult]:
        """
        Search FAISS index, return top-k results as RetrievalResult objects.
        
        Args:
            query_vector: shape (1, 384), already normalized
            k: number of results to return
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call load_from_npz() first.")

        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.ndim != 2 or query_vector.shape[1] != self.dim:
            raise ValueError(
                f"query_vector must have shape ({self.dim},) or (n, {self.dim}); "
                f"got {query_vector.shape}"
            )
        
        # FAISS returns: scores shape (1, k), indices shape (1, k)
        # scores = inner product = cosine similarity (on normalized vectors)
        # indices = position in the original vectors array
        scores, indices = self.index.search(query_vector, k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 when fewer than k results exist
                continue
            idx = int(idx)
            results.append(RetrievalResult(
                rank=rank + 1,
                score=float(score),
                source=self._source_for_index(idx),
                text=self.texts[idx],
            ))
        return results
