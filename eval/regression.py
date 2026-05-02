"""
eval/regression.py
Runs golden test cases against the full pipeline.
CI gate: if pass rate < 0.8, block the merge.
"""

import asyncio
import json
import logging
from pathlib import Path

from agent.reviewer import CodeReviewer, FileReviewInput
from llm.client import OllamaClient

logger = logging.getLogger(__name__)

PASS_RATE_THRESHOLD = 0.80   # must pass 80% of golden tests to merge


async def run_regression(test_cases_path: str = "eval/test_cases.json") -> dict:
    """
    Runs all golden test cases. Returns pass rate and per-test results.

    # [INTERNAL] Regression tests = golden cases that must ALWAYS pass.
    # They encode your minimum correctness bar. If a code change breaks tc001
    # (plaintext password → critical/security), you know immediately.
    # This is the CI gate: GitHub Actions runs this before merge.
    # Pass rate < 80% → CI fails → PR cannot merge.
    """
    cases = json.loads(Path(test_cases_path).read_text())
    client = OllamaClient(model="deepseek-coder:6.7b")
    reviewer = CodeReviewer(client=client)

    results = []

    for case in cases:
        file_input = FileReviewInput(
            file_path=case["file_path"],
            diff_chunk=case["diff_chunk"],
            language=case["language"],
        )

        # Override retriever with golden docs from test case
        reviewer._retrieve = lambda diff, lang, docs=case["retrieved_docs"]: _return(docs)

        summary = await reviewer.review_pr([file_input])
        comment = summary.comments[0] if summary.comments else None

        passed, reason = _check_case(case, comment)
        results.append({
            "id": case["id"],
            "description": case["description"],
            "passed": passed,
            "reason": reason,
            "got_severity": comment.severity if comment else None,
            "got_category": comment.category if comment else None,
        })

        status = "PASS" if passed else "FAIL"
        logger.info("[%s] %s — %s", status, case["id"], reason)

    await client.close()

    passed_count = sum(1 for r in results if r["passed"])
    pass_rate = passed_count / len(results) if results else 0.0

    return {
        "pass_rate": pass_rate,
        "passed": passed_count,
        "total": len(results),
        "threshold": PASS_RATE_THRESHOLD,
        "gate_passed": pass_rate >= PASS_RATE_THRESHOLD,
        "results": results,
    }


def _check_case(case: dict, comment) -> tuple[bool, str]:
    """Checks if output matches golden expectations."""
    if not case["should_comment"]:
        if comment is None:
            return True, "correctly produced no comment"
        return False, f"expected no comment, got severity={comment.severity}"

    if comment is None:
        return False, "expected a comment, got none"

    if case["expected_severity"] and comment.severity != case["expected_severity"]:
        return False, f"severity: expected {case['expected_severity']}, got {comment.severity}"

    if case["expected_category"] and comment.category != case["expected_category"]:
        return False, f"category: expected {case['expected_category']}, got {comment.category}"

    return True, "severity and category match"


async def _return(docs: list[str]) -> list[str]:
    return docs


if __name__ == "__main__":
    result = asyncio.run(run_regression())
    print(json.dumps(result, indent=2))
    exit(0 if result["gate_passed"] else 1)