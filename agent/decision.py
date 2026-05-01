"""
agent/decision.py
Decides which files to review and assigns severity routing.
Rule-based filtering runs before LLM calls.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SKIP_PATTERNS: list[str] = [
    "*.pyc",
    "*.pyo",
    "__pycache__/*",
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "migrations/*",
    "alembic/versions/*",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
    "*.ico",
    "*.pdf",
    "*.zip",
    "*.tar.gz",
    "*.md",
    "*.rst",
    ".env*",
    "*.txt",
    "dist/*",
    "build/*",
    "*.egg-info/*",
    "node_modules/*",
]

HIGH_PRIORITY_SIGNALS = ("auth", "security", "payment", "credential", "token", "password")
LOW_PRIORITY_SIGNALS = ("test_", "_test", "spec_", "_spec", "fixture")
MAX_REVIEWABLE_DIFF_LINES = 200


def should_skip_file(file_path: str, diff_lines: int) -> tuple[bool, str]:
    """Return whether a changed file should be skipped before LLM review."""
    path = Path(file_path)

    for pattern in SKIP_PATTERNS:
        if path.match(pattern):
            return True, f"matches skip pattern: {pattern}"

    if diff_lines > MAX_REVIEWABLE_DIFF_LINES:
        return True, (
            f"diff too large: {diff_lines} lines exceeds "
            f"{MAX_REVIEWABLE_DIFF_LINES}-line limit"
        )

    if diff_lines == 0:
        return True, "empty diff - no changes to review"

    return False, ""


def classify_file_priority(file_path: str) -> str:
    """Return review priority: high, normal, or low."""
    path_str = file_path.lower()

    if any(signal in path_str for signal in LOW_PRIORITY_SIGNALS):
        return "low"
    if any(signal in path_str for signal in HIGH_PRIORITY_SIGNALS):
        return "high"
    return "normal"


def compute_severity_floor(file_priority: str) -> str | None:
    """Return the minimum severity for a file priority, if any."""
    return "warning" if file_priority == "high" else None
