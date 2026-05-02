# agent/reviewer.py
"""
Orchestrates the full review pipeline for a single pull request.
"""
import logging
import time
from dataclasses import dataclass, field

from agent.decision import classify_file_priority, compute_severity_floor, should_skip_file
from agent.guardrails import validate_comment
from cache.semantic_cache import SemanticCache
from guardrails.input_guard import InputGuard
from guardrails.output_guard import OutputGuard
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
    cache_hits: int = 0                          # NEW
    comments: list[ReviewComment] = field(default_factory=list)
    skip_reasons: dict[str, str] = field(default_factory=dict)
    processing_time_ms: float = 0.0


class CodeReviewer:
    """Main review agent. Stateless across PRs."""

    def __init__(
        self,
        client: OllamaClient,
        retriever=None,
        cache: SemanticCache | None = None,      # NEW — injected, not created here
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.cache = cache
        self.input_guard = InputGuard(block_on_credentials=True)
        self.output_guard = OutputGuard(min_confidence=0.4)

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

            # --- Input guard ---
            # Runs before anything else — before retrieval, before cache, before LLM.
            # If credentials found, we redact and continue (never block the whole PR).
            guard_result = self.input_guard.check(file.diff_chunk, file.file_path)
            if not guard_result.passed:
                logger.warning(
                    "InputGuard violations in %s: %s",
                    file.file_path,
                    guard_result.violations,
                )
            # Always use redacted content downstream — even if guard passed,
            # redacted_content == original content, so this is safe unconditionally.
            safe_diff = guard_result.redacted_content

            priority = classify_file_priority(file.file_path)
            severity_floor = compute_severity_floor(priority)

            # --- Cache check ---
            # We need an embedding of the diff to do similarity lookup.
            # Retriever already has an embed() method — reuse it.
            # [INTERNAL] Cache key = embedding of (file_path + safe_diff).
            # Including file_path means same diff in different files = different cache entries.
            # This is intentional: context.py and auth.py have different review standards.
            comment = None
            cache_key_text = f"{file.file_path}\n{safe_diff}"

            if self.cache is not None and self.retriever is not None:
                try:
                    query_vec = self._embed(cache_key_text)
                    cached = self.cache.get(query_vec)
                    if cached is not None:
                        # Rebuild ReviewComment from cached dict
                        comment = ReviewComment(**{
                            k: v for k, v in cached.items()
                            if not k.startswith("_")   # strip _cached_at, _cache_similarity etc.
                        })
                        summary.cache_hits += 1
                        logger.info("Cache hit for %s", file.file_path)
                except Exception as exc:
                    logger.error("Cache lookup failed: %s — proceeding without cache", exc)

            # --- LLM call (only if cache missed) ---
            if comment is None:
                retrieved_docs = await self._retrieve(safe_diff, file.language)
                ctx = ReviewContext(
                    file_path=file.file_path,
                    diff_chunk=safe_diff,
                    retrieved_docs=retrieved_docs,
                    language=file.language,
                )
                comment = await generate_review_comment(ctx, self.client)

                # Store in cache after successful LLM call
                if comment is not None and self.cache is not None and self.retriever is not None:
                    try:
                        query_vec = self._embed(cache_key_text)
                        self.cache.set(query_vec, comment.__dict__, query_text=cache_key_text)
                    except Exception as exc:
                        logger.error("Cache store failed: %s", exc)

            if comment is None:
                summary.files_reviewed += 1
                continue

            # --- Severity floor ---
            if severity_floor and _severity_rank(comment.severity) < _severity_rank(severity_floor):
                comment.severity = severity_floor

            # --- Existing line-level guardrail (agent/guardrails.py) ---
            valid, rejection_reason = validate_comment(comment, safe_diff)
            if not valid:
                logger.warning("Comment rejected for %s: %s", file.file_path, rejection_reason)
                summary.files_reviewed += 1
                continue

            # --- Output guard ---
            # Extracts valid line numbers from diff for hallucination check.
            valid_lines = _extract_changed_lines(safe_diff)
            out_result = self.output_guard.check([comment.__dict__], valid_lines)
            if not out_result.filtered_comments:
                logger.warning(
                    "OutputGuard dropped all comments for %s: %s",
                    file.file_path,
                    out_result.violations,
                )
                summary.files_reviewed += 1
                continue

            summary.comments.append(comment)
            summary.files_reviewed += 1

        summary.processing_time_ms = (time.monotonic() - start) * 1000
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
                return [getattr(r, "text", "") for r in results]
            results = self.retriever.retrieve(diff_chunk, top_k=3)
            return [getattr(r, "content", "") for r in results]
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return []

    def _embed(self, text: str):
        """
        Reuse retriever's embedding method for cache key generation.
        Avoids loading a second model just for the cache.
        """
        if hasattr(self.retriever, "embed"):
            return self.retriever.embed(text)
        if hasattr(self.retriever, "embedder"):
            return self.retriever.embedder.embed(text)
        raise AttributeError("Retriever has no embed() method — cannot generate cache key")


def _extract_changed_lines(diff_chunk: str) -> set[int]:
    """
    Parse unified diff to extract line numbers actually changed (+lines).

    [INTERNAL] Unified diff hunk header: @@ -old_start,old_count +new_start,new_count @@
    We track the new-file line counter. Lines starting with '+' are additions.
    Lines starting with '-' don't advance the new-file counter.
    Lines starting with ' ' (context) advance the counter but aren't "changed".
    """
    changed: set[int] = set()
    current_line = 0

    for line in diff_chunk.splitlines():
        if line.startswith("@@"):
            # Extract new file start line from hunk header
            # e.g. "@@ -10,6 +15,8 @@" → new start = 15
            import re
            match = re.search(r'\+(\d+)', line)
            if match:
                current_line = int(match.group(1)) - 1  # -1 because we pre-increment
        elif line.startswith("+") and not line.startswith("+++"):
            current_line += 1
            changed.add(current_line)
        elif line.startswith("-") and not line.startswith("---"):
            pass  # removed lines don't exist in new file
        else:
            current_line += 1  # context line — advances counter

    return changed


def _severity_rank(severity: str | None) -> int:
    return {"nitpick": 1, "suggestion": 2, "warning": 3, "critical": 4}.get(
        severity or "", 0,
    )