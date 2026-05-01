# tests/test_eval.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import (
    RetrievalResult,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


def _make_results(relevant_flags: list[bool]) -> list[RetrievalResult]:
    return [
        RetrievalResult(doc_id=str(i), content="doc", is_relevant=flag)
        for i, flag in enumerate(relevant_flags)
    ]


def test_precision_at_5_three_relevant():
    results = _make_results([True, False, True, False, True])
    assert precision_at_k(results, 5) == 0.6


def test_precision_at_3():
    results = _make_results([True, False, True, False, True])
    assert precision_at_k(results, 3) == pytest.approx(0.667, abs=0.01)


def test_recall_at_5():
    results = _make_results([True, False, True, False, True])
    # 3 relevant returned, 5 total relevant exist
    assert recall_at_k(results, 5, total_relevant=5) == 0.6


def test_recall_perfect():
    results = _make_results([True, True, True])
    assert recall_at_k(results, 3, total_relevant=3) == 1.0


def test_reciprocal_rank_first():
    results = _make_results([True, False, False])
    assert reciprocal_rank(results) == 1.0


def test_reciprocal_rank_second():
    results = _make_results([False, True, False])
    assert reciprocal_rank(results) == 0.5


def test_reciprocal_rank_none():
    results = _make_results([False, False, False])
    assert reciprocal_rank(results) == 0.0


def test_mrr():
    all_results = [
        _make_results([True, False, False]),   # RR = 1.0
        _make_results([False, True, False]),   # RR = 0.5
        _make_results([False, False, True]),   # RR = 0.33
    ]
    mrr = mean_reciprocal_rank(all_results)
    assert abs(mrr - 0.611) < 0.01


import pytest