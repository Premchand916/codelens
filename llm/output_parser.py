"""
llm/output_parser.py
Parses and validates structured JSON output from the LLM.
Handles malformed JSON with retry logic — small models fail ~20% of the time.
"""

import json
import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from llm.client import OllamaClient
from llm.prompts import ReviewContext, build_review_prompt, fits_in_context

logger = logging.getLogger(__name__)


# [SURFACE] Pydantic model enforces the exact schema we told the LLM to return.
# [INTERNAL] This is your output contract. Even if the LLM returns the right
# JSON keys, wrong types fail here (e.g. line_number as "11" string vs 11 int).
# Pydantic coerces where safe, raises ValidationError where it can't.
class ReviewComment(BaseModel):
    should_comment: bool
    severity: str | None = Field(None, pattern="^(critical|warning|suggestion|nitpick)$")
    category: str | None = Field(None, pattern="^(bug|security|performance|style|best_practice)$")
    line_number: int | None = None
    comment: str | None = None
    suggested_fix: str | None = None


def extract_json(raw: str) -> dict[str, Any] | None:
    """
    Extract JSON from LLM output — handles common failure modes.
    
    # [INTERNAL] Small models (8B) fail to return clean JSON in ~20% of calls.
    # Common failures:
    #   1. Wrapped in markdown: ```json { ... } ```
    #   2. Extra explanation before/after the JSON
    #   3. Trailing commas (invalid JSON)
    #   4. Single quotes instead of double quotes
    # Strategy: try direct parse first, then regex extraction, then give up.
    # Never silently swallow errors — log what broke so you can improve prompts.
    """
    # Attempt 1: direct parse (works ~80% of the time)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown code fences
    # LLMs often return: ```json\n{...}\n```
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: find the first {...} block in the response
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # All attempts failed
    logger.error("Failed to extract JSON from LLM output: %r", raw[:200])
    raise ValueError(f"Could not parse JSON from model output: {raw[:200]}")

class ParseResult(Enum):
    NO_COMMENT_NEEDED = "no_comment"   # model decided correctly → don't retry
    PARSE_FAILED = "parse_failed"      # malformed JSON → retry


def parse_review_output(raw: str) -> ReviewComment | ParseResult:
    """
    Parse raw LLM string → validated ReviewComment.
    Returns None if model correctly decided not to comment.
    
    # [INTERNAL] Two-stage validation:
    # Stage 1: JSON extraction (structural) — is it valid JSON?
    # Stage 2: Pydantic validation (semantic) — do the values make sense?
    # Separating these stages tells you WHERE parsing broke, not just THAT it broke.
    # Critical for debugging prompt issues at scale.
    """
    try:
        data = extract_json(raw)
    except ValueError:
        return ParseResult.PARSE_FAILED          # retry this

    if data is None:
        return ParseResult.NO_COMMENT_NEEDED
    if not isinstance(data, dict):
        return ParseResult.PARSE_FAILED
    if not data.get("should_comment", True):
        return ParseResult.NO_COMMENT_NEEDED     # don't retry this

    try:
        comment = ReviewComment(**data)
        if not comment.comment:
            return ParseResult.PARSE_FAILED      # retry — model half-answered
        return comment
    except ValidationError as e:
        logger.error("Validation failed: %s", e)
        return ParseResult.PARSE_FAILED          # retry this


async def generate_review_comment(
    ctx: ReviewContext,
    client: OllamaClient,
    max_retries: int = 2,
) -> ReviewComment | None:
    messages = build_review_prompt(ctx)

    if not fits_in_context(messages):
        ctx.retrieved_docs = ctx.retrieved_docs[:1]
        messages = build_review_prompt(ctx)

    for attempt in range(max_retries + 1):
        try:
            raw = await client.chat(messages, temperature=0.0)
            result = parse_review_output(raw)

            if isinstance(result, ReviewComment):
                return result                              # success
            if result == ParseResult.NO_COMMENT_NEEDED:
                return None                               # done, don't retry
            # ParseResult.PARSE_FAILED → fall through to retry
            logger.warning("Parse failed attempt %d/%d", attempt + 1, max_retries + 1)

        except Exception as e:
            logger.error("LLM call failed attempt %d: %s", attempt + 1, e)
            if attempt == max_retries:
                return None

    return None
