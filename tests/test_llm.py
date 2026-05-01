import pytest

from llm.client import OllamaClient
from llm.output_parser import (
    ParseResult,
    ReviewComment,
    generate_review_comment,
    parse_review_output,
)
from llm.prompts import ReviewContext, build_review_prompt, fits_in_context


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        self.calls += 1
        return self.responses.pop(0)


def review_context(diff_chunk: str) -> ReviewContext:
    return ReviewContext(
        file_path="app/auth.py",
        language="python",
        diff_chunk=diff_chunk,
        retrieved_docs=[
            "Never hardcode credentials. Secrets must come from environment variables.",
            "Validate user input and avoid logging sensitive values.",
        ],
    )


def test_build_review_prompt_contains_diff_and_context():
    ctx = review_context("+ password = 'secret123'")
    messages = build_review_prompt(ctx)

    assert [m["role"] for m in messages] == ["system", "user"]
    assert "JSON" in messages[0]["content"]
    assert "app/auth.py" in messages[1]["content"]
    assert "+ password = 'secret123'" in messages[1]["content"]
    assert "Never hardcode credentials" in messages[1]["content"]


def test_fits_in_context_rejects_oversized_prompt():
    messages = [{"role": "user", "content": "x" * 20}]
    assert fits_in_context(messages, max_chars=20)
    assert not fits_in_context(messages, max_chars=19)


def test_parse_review_output_valid_comment():
    raw = """
    ```json
    {
      "should_comment": true,
      "severity": "critical",
      "category": "security",
      "line_number": 12,
      "comment": "Do not hardcode passwords in source code.",
      "suggested_fix": "Load the password from an environment variable."
    }
    ```
    """

    result = parse_review_output(raw)

    assert isinstance(result, ReviewComment)
    assert result.severity == "critical"
    assert result.category == "security"
    assert result.line_number == 12


def test_parse_review_output_no_comment():
    assert parse_review_output('{"should_comment": false}') == ParseResult.NO_COMMENT_NEEDED
    assert parse_review_output("null") == ParseResult.NO_COMMENT_NEEDED


def test_parse_review_output_malformed_json():
    assert parse_review_output("not json") == ParseResult.PARSE_FAILED


@pytest.mark.asyncio
async def test_generate_review_comment_retries_then_succeeds():
    client = FakeClient(
        [
            "not json",
            """
            {
              "should_comment": true,
              "severity": "warning",
              "category": "security",
              "line_number": 3,
              "comment": "Avoid hardcoded secrets.",
              "suggested_fix": "Read the value from os.environ."
            }
            """,
        ]
    )

    result = await generate_review_comment(
        review_context("+ API_KEY = 'abc123'"),
        client,  # type: ignore[arg-type]
        max_retries=1,
    )

    assert isinstance(result, ReviewComment)
    assert result.comment == "Avoid hardcoded secrets."
    assert client.calls == 2


@pytest.mark.asyncio
async def test_generate_review_comment_returns_none_for_clean_code():
    client = FakeClient(['{"should_comment": false}'])

    result = await generate_review_comment(
        review_context("+ total = sum(item.price for item in items)"),
        client,  # type: ignore[arg-type]
    )

    assert result is None
    assert client.calls == 1


@pytest.mark.asyncio
async def test_ollama_available():
    client = OllamaClient()
    try:
        available = await client.is_available()
    finally:
        await client.close()

    if not available:
        pytest.skip("Ollama is not running or llama3.1:8b is not installed")

    assert available
