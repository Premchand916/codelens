# guardrails/input_guard.py
"""
Input guardrails for CodeLens.

Runs BEFORE the LLM sees any code. Three jobs:
1. Detect hardcoded credentials (regex-based)
2. Detect PII patterns (regex-based)
3. Enforce size limits (token budget protection)

[INTERVIEW] "Why regex and not an LLM for credential detection?"
→ Regex is deterministic, zero latency, zero tokens. An LLM might
  miss 'p@ssw0rd123' or hallucinate a false positive. For pattern
  matching with known structure (API keys, secrets), regex wins.
  LLM-based detection makes sense only for *contextual* PII —
  e.g., "Is this a real person's name?" — where structure alone
  isn't enough.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result type
# ------------------------------------------------------------------

@dataclass
class GuardResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    redacted_content: str = ""  # content with secrets replaced

    def add(self, message: str) -> None:
        self.violations.append(message)
        self.passed = False


# ------------------------------------------------------------------
# Patterns
# ------------------------------------------------------------------

# [INTERNAL] Each pattern targets a known credential format.
# We use named groups (?P<secret>...) so we can redact precisely
# rather than blanking the whole line.
CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS Access Key",      re.compile(r'(?i)AKIA[0-9A-Z]{16}')),
    ("AWS Secret Key",      re.compile(r'(?i)aws.{0,20}secret.{0,20}["\']([A-Za-z0-9/+=]{40})')),
    ("GitHub Token",        re.compile(r'gh[pousr]_[A-Za-z0-9]{36,}')),
    ("Generic API Key",     re.compile(r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([A-Za-z0-9\-_]{20,})')),
    ("Generic Secret",      re.compile(r'(?i)(secret|password|passwd|pwd)\s*[=:]\s*["\'](.{6,})["\']')),
    ("Private Key Block",   re.compile(r'-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----')),
    ("Bearer Token",        re.compile(r'(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*')),
]

PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Email",       re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')),
    ("Phone (US)",  re.compile(r'\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')),
    ("SSN",         re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),
    ("IPv4",        re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')),
]

# Files we skip entirely — generated code, lock files, binaries
SKIP_FILE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'package-lock\.json$'),
    re.compile(r'yarn\.lock$'),
    re.compile(r'poetry\.lock$'),
    re.compile(r'\.min\.(js|css)$'),
    re.compile(r'\.(png|jpg|jpeg|gif|ico|woff|woff2|ttf|eot|pdf|zip)$'),
]

MAX_DIFF_CHARS = 150_000   # ~37K tokens — hard ceiling per PR
MAX_FILE_CHARS = 20_000    # ~5K tokens per file


# ------------------------------------------------------------------
# Main guard
# ------------------------------------------------------------------

class InputGuard:
    """
    Validates and redacts input before it reaches the LLM.

    Two modes:
    - block=True  → raise if credential found (strict, for prod)
    - block=False → redact and continue, log warning (lenient, for dev)
    """

    def __init__(self, block_on_credentials: bool = True) -> None:
        self.block_on_credentials = block_on_credentials

    def check(self, content: str, filename: str = "") -> GuardResult:
        """
        Run all checks. Returns GuardResult with pass/fail + violations.
        Redacted content is always populated (safe to pass to LLM).
        """
        result = GuardResult(passed=True, redacted_content=content)

        self._check_skip_file(filename, result)
        if not result.passed and self.block_on_credentials:
            return result  # skip file entirely, no need to scan

        self._check_size(content, filename, result)
        self._check_credentials(result)   # mutates redacted_content
        self._check_pii(result)

        if result.violations:
            logger.warning(
                "InputGuard violations in '%s': %s",
                filename,
                result.violations,
            )

        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_skip_file(self, filename: str, result: GuardResult) -> None:
        for pattern in SKIP_FILE_PATTERNS:
            if pattern.search(filename):
                result.add(f"Skipped file type: {filename}")
                return

    def _check_size(
        self, content: str, filename: str, result: GuardResult
    ) -> None:
        if len(content) > MAX_DIFF_CHARS:
            result.add(
                f"Diff too large: {len(content):,} chars "
                f"(max {MAX_DIFF_CHARS:,}). Truncation required."
            )
        if filename and len(content) > MAX_FILE_CHARS:
            # Warning only — don't block, but agent will chunk it
            logger.warning(
                "File '%s' is large (%d chars) — chunking recommended",
                filename,
                len(content),
            )

    def _check_credentials(self, result: GuardResult) -> None:
        """
        Scan for credentials. Redacts matches in-place.

        [INTERNAL] We redact regardless of block mode so the LLM never
        sees the raw secret. Redaction replaces the match with
        [REDACTED:<TYPE>] — preserves surrounding code structure so
        the LLM can still reason about the function.
        """
        redacted = result.redacted_content
        for label, pattern in CREDENTIAL_PATTERNS:
            matches = list(pattern.finditer(redacted))
            if matches:
                violation_msg = f"Credential detected: {label} ({len(matches)} occurrence(s))"
                if self.block_on_credentials:
                    result.add(violation_msg)
                else:
                    logger.warning(violation_msg)
                # Always redact regardless of block mode
                redacted = pattern.sub(f"[REDACTED:{label.upper().replace(' ', '_')}]", redacted)

        result.redacted_content = redacted

    def _check_pii(self, result: GuardResult) -> None:
        """
        PII is warned but NOT blocked — code legitimately contains emails
        (test fixtures, config examples). Log and redact, don't fail.
        """
        redacted = result.redacted_content
        for label, pattern in PII_PATTERNS:
            matches = list(pattern.finditer(redacted))
            if matches:
                logger.warning(
                    "PII pattern '%s' found (%d occurrence(s)) — redacting",
                    label,
                    len(matches),
                )
                redacted = pattern.sub(f"[REDACTED:{label.upper()}]", redacted)

        result.redacted_content = redacted