# guardrails/output_guard.py
"""
Output guardrails for CodeLens.

Runs AFTER the LLM generates a review comment, BEFORE it gets posted to GitHub.
Three jobs:
1. Hallucinated line number detection — did LLM cite a line not in the diff?
2. Confidence filter — drop low-confidence comments (noise reduction)
3. Comment sanity checks — length, empty content, duplicate detection

[INTERVIEW] "What's the difference between input and output guardrails?"
→ Input guards protect the LLM from bad data (secrets, oversized context).
  Output guards protect the user from bad LLM output (hallucinations,
  low-quality comments). They're complementary layers — neither alone is enough.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result type
# ------------------------------------------------------------------

@dataclass
class OutputGuardResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    filtered_comments: list[dict] = field(default_factory=list)  # comments that survived
    dropped_count: int = 0

    def drop(self, comment: dict, reason: str) -> None:
        self.violations.append(f"Dropped comment (line {comment.get('line_number', '?')}): {reason}")
        self.dropped_count += 1
        logger.warning("OutputGuard dropped comment: %s | reason: %s", comment, reason)


# ------------------------------------------------------------------
# Main guard
# ------------------------------------------------------------------

class OutputGuard:
    """
    Validates LLM-generated review comments before GitHub posting.

    [INTERNAL] The LLM sees a diff chunk and generates line numbers.
    But it hallucinates — it might cite line 87 when the diff only
    touched lines 10-30. We catch this by passing in the set of
    valid changed lines and rejecting any comment outside that set.
    This is a rule-based guardrail — deterministic, zero LLM cost.
    """

    def __init__(
        self,
        min_confidence: float = 0.4,
        max_comment_length: int = 1000,
        min_comment_length: int = 20,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_comment_length = max_comment_length
        self.min_comment_length = min_comment_length

    def check(
        self,
        comments: list[dict],
        valid_line_numbers: set[int] | None = None,
    ) -> OutputGuardResult:
        """
        Filter a list of LLM-generated comments.

        Args:
            comments: Raw comments from LLM output parser
            valid_line_numbers: Set of line numbers actually changed in diff.
                                 If None, line number check is skipped.

        Returns:
            OutputGuardResult with filtered_comments = comments that passed.
        """
        result = OutputGuardResult(passed=True, filtered_comments=[])
        seen_comments: set[str] = set()  # for duplicate detection

        for comment in comments:
            dropped = False

            # --- Check 1: Hallucinated line number ---
            # [INTERNAL] We compare the LLM's cited line_number against the
            # set of lines actually present in the unified diff. If the line
            # wasn't touched, the comment is fabricated context — dangerous
            # because it looks authoritative but refers to unchanged code.
            if valid_line_numbers is not None:
                line_num = comment.get("line_number")
                if line_num is not None and int(line_num) not in valid_line_numbers:
                    result.drop(comment, f"line {line_num} not in diff (hallucinated)")
                    dropped = True

            if dropped:
                continue

            # --- Check 2: Confidence threshold ---
            # Low-confidence comments are noise. Better to post 3 high-quality
            # comments than 10 where half are wrong.
            confidence = comment.get("confidence", 1.0)
            if confidence < self.min_confidence:
                result.drop(comment, f"confidence {confidence:.2f} < {self.min_confidence}")
                dropped = True

            if dropped:
                continue

            # --- Check 3: Comment length ---
            comment_text = comment.get("comment", "")
            if len(comment_text) < self.min_comment_length:
                result.drop(comment, f"comment too short ({len(comment_text)} chars)")
                dropped = True
            elif len(comment_text) > self.max_comment_length:
                # Truncate rather than drop — content is valid, just verbose
                comment["comment"] = comment_text[: self.max_comment_length] + "... [truncated]"
                logger.info("Comment truncated to %d chars", self.max_comment_length)

            if dropped:
                continue

            # --- Check 4: Duplicate detection ---
            # LLMs sometimes repeat the same observation for adjacent lines.
            # Fingerprint = first 80 chars of comment text (ignores minor rephrasing)
            fingerprint = re.sub(r'\s+', ' ', comment_text[:80].lower().strip())
            if fingerprint in seen_comments:
                result.drop(comment, "duplicate comment (same observation repeated)")
                dropped = True
            else:
                seen_comments.add(fingerprint)

            if not dropped:
                result.filtered_comments.append(comment)

        if result.dropped_count > 0:
            result.passed = result.dropped_count < len(comments)  # False only if ALL dropped
            logger.info(
                "OutputGuard: %d/%d comments passed",
                len(result.filtered_comments),
                len(comments),
            )

        return result