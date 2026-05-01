"""
eval/metrics.py
Retrieval and review evaluation metrics.
Built from scratch — no libraries. Understand the math before using RAGAS.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """One retrieval result with a relevance label."""
    doc_id: str
    content: str
    is_relevant: bool   # ground truth label


def precision_at_k(results: list[RetrievalResult], k: int) -> float:
    """
    Precision@K = relevant docs in top-K / K

    # [INTERNAL] Only looks at the top-K results.
    # Penalizes returning irrelevant docs — noise in context = wasted tokens.
    # Range: 0.0 (all noise) to 1.0 (all relevant).
    """
    if k <= 0:
        return 0.0
    top_k = results[:k]
    relevant = sum(1 for r in top_k if r.is_relevant)
    return relevant / k


def recall_at_k(results: list[RetrievalResult], k: int, total_relevant: int) -> float:
    """
    Recall@K = relevant docs in top-K / total relevant docs that exist

    # [INTERNAL] Measures coverage — did we find everything useful?
    # total_relevant comes from your ground truth dataset.
    # Range: 0.0 (found nothing) to 1.0 (found everything).
    """
    if total_relevant <= 0:
        return 0.0
    top_k = results[:k]
    relevant_found = sum(1 for r in top_k if r.is_relevant)
    return relevant_found / total_relevant


def reciprocal_rank(results: list[RetrievalResult]) -> float:
    """
    Reciprocal Rank = 1 / rank_of_first_relevant_doc

    # [INTERNAL] Rewards finding a relevant doc early.
    # First result relevant → RR = 1/1 = 1.0
    # Second result relevant → RR = 1/2 = 0.5
    # Tenth result relevant → RR = 1/10 = 0.1
    # No relevant result → RR = 0.0
    # MRR (Mean Reciprocal Rank) = average RR across many queries.
    # Used when you care about "did we get ONE good result near the top?"
    """
    for i, result in enumerate(results):
        if result.is_relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(all_results: list[list[RetrievalResult]]) -> float:
    """MRR across multiple queries."""
    if not all_results:
        return 0.0
    return sum(reciprocal_rank(r) for r in all_results) / len(all_results)