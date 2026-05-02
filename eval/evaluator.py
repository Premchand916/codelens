
"""
eval/evaluator.py
LLM-as-judge pattern: use one LLM call to evaluate another's output.
Measures relevance, faithfulness, specificity of generated review comments.
"""

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from llm.client import OllamaClient
from llm.output_parser import extract_json

logger = logging.getLogger(__name__)


@dataclass
class ReviewEvalResult:
    """Evaluation scores for one generated review comment."""
    comment_id: str
    relevance: float
    faithfulness: float
    specificity: float
    overall: float
    judge_reasoning: str


_JUDGE_SYSTEM_PROMPT = """You are an impartial code review evaluator.
Score the given review comment on three dimensions.

Return ONLY valid JSON:
{
  "relevance": <0.0-1.0>,
  "faithfulness": <0.0-1.0>,
  "specificity": <0.0-1.0>,
  "reasoning": "<one sentence explaining your scores>"
}

Scoring rubric:
- relevance: Does the comment address an actual issue in the diff? (0=off-topic, 1=directly relevant)
- faithfulness: Is every claim grounded in the diff? (0=hallucinated, 1=fully grounded)
- specificity: Does it reference actual variable/function names? (0=generic, 1=highly specific)"""


async def evaluate_comment(
    diff_chunk: str,
    retrieved_docs: list[str],
    generated_comment: str,
    comment_id: str,
    judge_client: OllamaClient,
) -> ReviewEvalResult:
    """Evaluate one review comment using LLM-as-judge."""
    context = "\n".join(f"[Doc {i+1}] {d}" for i, d in enumerate(retrieved_docs[:2]))

    user_message = f"""## Diff
{diff_chunk}

## Retrieved Best Practices
{context}

## Generated Review Comment
{generated_comment}

Score this comment."""

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        raw = await judge_client.chat(messages, temperature=0.0)
        data = extract_json(raw)

        relevance = float(data.get("relevance", 0.0))
        faithfulness = float(data.get("faithfulness", 0.0))
        specificity = float(data.get("specificity", 0.0))
        overall = (relevance * 0.3) + (faithfulness * 0.5) + (specificity * 0.2)

        return ReviewEvalResult(
            comment_id=comment_id,
            relevance=relevance,
            faithfulness=faithfulness,
            specificity=specificity,
            overall=overall,
            judge_reasoning=data.get("reasoning", ""),
        )

    except Exception as e:
        logger.error("Judge evaluation failed for %s: %s", comment_id, e)
        return ReviewEvalResult(
            comment_id=comment_id,
            relevance=0.0,
            faithfulness=0.0,
            specificity=0.0,
            overall=0.0,
            judge_reasoning=f"Evaluation error: {e}",
        )


# ─────────────────────────────────────────────
# Suite runner
# ─────────────────────────────────────────────

@dataclass
class TestCaseResult:
    test_id: str
    passed: bool
    reason: str
    scores: ReviewEvalResult | None = None


@dataclass
class EvalSuiteResult:
    total: int
    passed: int
    failed: int
    failures: list[TestCaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total


PASS_THRESHOLD = 0.70


async def _evaluate_one(tc: dict, judge: OllamaClient) -> TestCaseResult:
    test_id = tc["id"]

    if not tc["should_comment"]:
        if tc.get("sample_comment") is None:
            return TestCaseResult(test_id=test_id, passed=True,
                                  reason="Correctly produced no comment for clean code")
        return TestCaseResult(test_id=test_id, passed=False,
                              reason="False positive: comment generated for clean code")

    sample_comment = tc.get("sample_comment")
    if not sample_comment:
        return TestCaseResult(test_id=test_id, passed=False,
                              reason="Misconfigured: should_comment=True but sample_comment is null")

    scores = await evaluate_comment(
        diff_chunk=tc["diff_chunk"],
        retrieved_docs=tc["retrieved_docs"],
        generated_comment=sample_comment,
        comment_id=test_id,
        judge_client=judge,
    )

    passed = scores.overall >= PASS_THRESHOLD
    reason = (
        f"overall={scores.overall:.2f} >= {PASS_THRESHOLD} → PASS"
        if passed else
        f"overall={scores.overall:.2f} < {PASS_THRESHOLD} → FAIL "
        f"(rel={scores.relevance:.2f} faith={scores.faithfulness:.2f} spec={scores.specificity:.2f})"
    )

    return TestCaseResult(test_id=test_id, passed=passed, reason=reason, scores=scores)


async def _run_full_eval_async(
    test_cases_path: str = "eval/test_cases.json",
    verbose: bool = False,
) -> EvalSuiteResult:
    cases = json.loads(Path(test_cases_path).read_text())
    judge = OllamaClient(model="llama3.1:8b")
    results: list[TestCaseResult] = []

    for tc in cases:
        result = await _evaluate_one(tc, judge)
        results.append(result)
        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {result.test_id}: {result.reason}")

    passed = sum(1 for r in results if r.passed)
    failures = [r for r in results if not r.passed]

    suite = EvalSuiteResult(total=len(results), passed=passed,
                            failed=len(failures), failures=failures)

    if verbose:
        print(f"\nResult: {passed}/{len(results)} passed ({suite.pass_rate:.1%})")

    return suite


def run_full_eval(
    test_cases_path: str = "eval/test_cases.json",
    verbose: bool = False,
) -> EvalSuiteResult:
    """Sync wrapper — CI and __main__ both call this."""
    return asyncio.run(_run_full_eval_async(test_cases_path, verbose))


def run_ci_eval(output_path: str) -> dict:
    """Run eval suite in CI mode — writes JSON, returns dict."""
    results = run_full_eval()

    output = {
        "pass_rate": results.pass_rate,
        "total": results.total,
        "passed": results.passed,
        "failed": results.failed,
        "failures": [f.test_id for f in results.failures],
    }

    Path(output_path).write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev"], default="dev")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    if args.mode == "ci":
        result = run_ci_eval(args.output)
        print(json.dumps(result, indent=2))
    else:
        run_full_eval(verbose=True)
