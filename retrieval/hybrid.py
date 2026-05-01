"""
Hybrid retrieval: BM25 sparse search + FAISS dense search, fused with RRF.

This is the retrieval layer CodeLens should use for PR review queries because
code review questions often contain both exact identifiers and semantic intent.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np

from embeddings.vectorizer import EmbeddingPipeline
from retrieval.bm25 import BM25Result, BM25Retriever
from retrieval.vector_store import RetrievalResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    rank: int
    rrf_score: float
    source: str
    text: str
    bm25_rank: int | None
    dense_rank: int | None


class _FusionEntry(TypedDict):
    rrf_score: float
    source: str
    text: str
    bm25_rank: int | None
    dense_rank: int | None


class HybridRetriever:
    """
    Combines BM25 and dense retrieval with Reciprocal Rank Fusion.

    RRF uses rank positions rather than raw scores, so BM25 scores and cosine
    similarities do not need to be normalized onto the same scale.
    """

    def __init__(
        self,
        rrf_k: int = 60,
        pipeline: EmbeddingPipeline | None = None,
        bm25: BM25Retriever | None = None,
        dense: VectorStore | None = None,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        self.rrf_k = rrf_k
        self.pipeline = pipeline or EmbeddingPipeline()
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or VectorStore(dim=self.pipeline.dimension)
        self._indexed = False
        self._doc_count = 0

    def load(self, npz_path: str | Path, index_type: str = "flat") -> None:
        """Load one .npz knowledge-base bundle into both retrievers."""
        npz_path = Path(npz_path)
        data = np.load(npz_path, allow_pickle=True)
        texts = data["texts"].tolist()
        metadata = data["metadata"].tolist()

        if len(texts) != len(metadata):
            raise ValueError(
                f"texts and metadata must have the same length: "
                f"{len(texts)} != {len(metadata)}"
            )

        self.bm25.index(texts, metadata)
        self.dense.load_from_npz(npz_path, index_type=index_type)
        self._doc_count = len(texts)
        self._indexed = True
        logger.info("HybridRetriever loaded %s docs", self._doc_count)

    def search(self, query: str, k: int = 5, fetch_k: int = 10) -> list[HybridResult]:
        """Return top-k fused results for a natural-language/code-review query."""
        if not self._indexed:
            raise RuntimeError("Call load() before search()")
        if not query.strip():
            raise ValueError("Cannot search with an empty query")
        if k <= 0:
            return []

        fetch_k = max(k, fetch_k)
        fetch_k = min(fetch_k, self._doc_count)

        bm25_results = self.bm25.search(query, k=fetch_k)
        query_vector = self.pipeline.embed_batch([query])
        dense_results = self.dense.search(query_vector, k=fetch_k)

        fused = self._fuse(bm25_results, dense_results)
        sorted_entries = sorted(
            fused.values(),
            key=lambda entry: entry["rrf_score"],
            reverse=True,
        )

        return [
            HybridResult(
                rank=rank,
                rrf_score=entry["rrf_score"],
                source=entry["source"],
                text=entry["text"],
                bm25_rank=entry["bm25_rank"],
                dense_rank=entry["dense_rank"],
            )
            for rank, entry in enumerate(sorted_entries[:k], start=1)
        ]

    def _fuse(
        self,
        bm25_results: list[BM25Result],
        dense_results: list[RetrievalResult],
    ) -> dict[str, _FusionEntry]:
        fused: dict[str, _FusionEntry] = {}

        for result in bm25_results:
            entry = self._entry_for_result(fused, result.source, result.text)
            entry["rrf_score"] += self._rrf_score(result.rank)
            entry["bm25_rank"] = result.rank

        for result in dense_results:
            entry = self._entry_for_result(fused, result.source, result.text)
            entry["rrf_score"] += self._rrf_score(result.rank)
            entry["dense_rank"] = result.rank

        return fused

    def _entry_for_result(
        self,
        fused: dict[str, _FusionEntry],
        source: str,
        text: str,
    ) -> _FusionEntry:
        if source not in fused:
            fused[source] = {
                "rrf_score": 0.0,
                "source": source,
                "text": text,
                "bm25_rank": None,
                "dense_rank": None,
            }
        return fused[source]

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)
