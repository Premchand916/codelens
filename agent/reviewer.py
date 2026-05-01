"""
agent/reviewer.py
Orchestrates the full review pipeline for a single pull request.
"""

import logging
import time
from dataclasses import dataclass, field

from agent.decision import classify_file_priority, compute_severity_floor, should_skip_file
from agent.guardrails import validate_comment
from llm.client import OllamaClient
from llm.output_parser import ReviewComment, generate_review_comment
from llm.prompts import ReviewContext

logger = logging.getLogger(__name__)


@dataclass
class FileReviewInput:
    file_path: str
    diff_chunk: str
    language: str


@dataclass
class ReviewSummary:
    files_reviewed: int = 0
    files_skipped: int = 0
    comments: list[ReviewComment] = field(default_factory=list)
    skip_reasons: dict[str, str] = field(default_factory=dict)
    processing_time_ms: float = 0.0


class CodeReviewer:
    """Main review agent. Stateless across PRs."""

    def __init__(self, client: OllamaClient, retriever=None) -> None:
        self.client = client
        self.retriever = retriever

    async def review_pr(self, files: list[FileReviewInput]) -> ReviewSummary:
        summary = ReviewSummary()
        start = time.monotonic()

        for file in files:
            diff_lines = len(file.diff_chunk.splitlines())
            skip, reason = should_skip_file(file.file_path, diff_lines)
            if skip:
                summary.files_skipped += 1
                summary.skip_reasons[file.file_path] = reason
                logger.info("Skipping %s: %s", file.file_path, reason)
                continue

            priority = classify_file_priority(file.file_path)
            severity_floor = compute_severity_floor(priority)
            retrieved_docs = await self._retrieve(file.diff_chunk, file.language)

            ctx = ReviewContext(
                file_path=file.file_path,
                diff_chunk=file.diff_chunk,
                retrieved_docs=retrieved_docs,
                language=file.language,
            )
            comment = await generate_review_comment(ctx, self.client)

            if comment is None:
                summary.files_reviewed += 1
                continue

            if severity_floor and _severity_rank(comment.severity) < _severity_rank(severity_floor):
                comment.severity = severity_floor

            valid, rejection_reason = validate_comment(comment, file.diff_chunk)
            if not valid:
                logger.warning("Comment rejected for %s: %s", file.file_path, rejection_reason)
                summary.files_reviewed += 1
                continue

            summary.comments.append(comment)
            summary.files_reviewed += 1

        summary.processing_time_ms = (time.monotonic() - start) * 1000
        return summary

    async def _retrieve(self, diff_chunk: str, language: str) -> list[str]:
        if self.retriever is None:
            return [
                "Follow language style conventions.",
                "Handle exceptions explicitly.",
                "Never store secrets in source code.",
            ]

        try:
            if hasattr(self.retriever, "search"):
                results = self.retriever.search(f"{language}\n{diff_chunk}", k=3)
                return [getattr(result, "text", "") for result in results]
            results = self.retriever.retrieve(diff_chunk, top_k=3)
            return [getattr(result, "content", "") for result in results]
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return []


def _severity_rank(severity: str | None) -> int:
    """Map severity to an integer for floor comparison. Higher is worse."""
    return {"nitpick": 1, "suggestion": 2, "warning": 3, "critical": 4}.get(
        severity or "",
        0,
    )
