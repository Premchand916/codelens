"""
agent/guardrails.py
Validates LLM output before it gets posted to GitHub.
"""

import logging
import re

from llm.output_parser import ReviewComment

logger = logging.getLogger(__name__)


def extract_changed_line_numbers(diff_chunk: str) -> set[int]:
    """Parse a unified diff chunk and return line numbers for added lines."""
    changed_lines: set[int] = set()
    current_line: int | None = None

    for line in diff_chunk.splitlines():
        hunk_match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if hunk_match:
            current_line = int(hunk_match.group(1))
            continue

        if current_line is None:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            changed_lines.add(current_line)
            current_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        else:
            current_line += 1

    return changed_lines


def validate_comment(
    comment: ReviewComment,
    diff_chunk: str,
    min_confidence: float = 0.5,
) -> tuple[bool, str]:
    """Return whether an LLM review comment is safe and useful to post."""
    if comment.line_number is not None:
        valid_lines = extract_changed_line_numbers(diff_chunk)
        if valid_lines and comment.line_number not in valid_lines:
            logger.warning(
                "Hallucinated line number %d. Valid lines: %s",
                comment.line_number,
                sorted(valid_lines),
            )
            return False, (
                f"line {comment.line_number} not in diff "
                f"(valid: {sorted(valid_lines)})"
            )

    if not comment.comment or len(comment.comment.strip()) < 20:
        return False, "comment too short or empty"

    confidence = getattr(comment, "confidence", None)
    if confidence is not None and confidence < min_confidence:
        return False, f"confidence {confidence:.2f} below threshold {min_confidence}"

    return True, ""
