"""
llm/prompts.py
Prompt templates for the code review agent.
Separates prompt logic from LLM client logic — easier to iterate on prompts
without touching infrastructure code.
"""

from dataclasses import dataclass


@dataclass
class ReviewContext:
    """Everything the LLM needs to generate one review comment."""
    file_path: str
    diff_chunk: str           # The actual changed code
    retrieved_docs: list[str] # Best practices from RAG retrieval
    language: str             # "python", "javascript", etc.


# [SURFACE] System prompt defines the LLM's role and output contract.
# [INTERNAL] The system prompt is injected as role="system" — it sits ABOVE
# the conversation and sets persistent behavior. Most instruction-following
# models (llama3, GPT-4) are trained to prioritize system prompts over user
# turns. By defining the output schema here, we're using the model's
# instruction-following capability to enforce structure.
SYSTEM_PROMPT = """You are a senior software engineer performing a code review.
Your job: analyze a code diff and return a structured JSON review comment.

Rules:
- Only comment on the CHANGED lines (lines starting with +)
- Never hallucinate line numbers — use only line numbers present in the diff
- If the change is correct, set should_comment to false
- Be specific: reference the actual variable names and logic from the diff

Output format (JSON only, no markdown, no explanation):
{
  "should_comment": true | false,
  "severity": "critical" | "warning" | "suggestion" | "nitpick",
  "category": "bug" | "security" | "performance" | "style" | "best_practice",
  "line_number": <integer from the diff>,
  "comment": "<specific, actionable feedback>",
  "suggested_fix": "<code snippet or null>"
}"""


def build_review_prompt(ctx: ReviewContext) -> list[dict]:
    """
    Builds the full messages array for the chat API.
    
    Returns OpenAI-compatible message format — Ollama uses the same schema:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    
    # [INTERNAL] Few-shot prompting: giving the model 1-2 examples of
    # input→output pairs dramatically improves structured output quality.
    # The model pattern-matches against examples rather than reasoning from scratch.
    # This is especially important for small models (8B) that struggle with
    # complex JSON schemas without examples.
    """
    # Format retrieved docs as numbered context
    context_block = "\n\n".join(
        f"[Best Practice {i+1}]\n{doc}"
        for i, doc in enumerate(ctx.retrieved_docs[:3])  # cap at 3 to save tokens
    )

    user_message = f"""## Code Review Task

**File:** {ctx.file_path}
**Language:** {ctx.language}

## Relevant Best Practices (from knowledge base)
{context_block}

## Diff to Review
{ctx.diff_chunk}

Return JSON only."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def fits_in_context(messages: list[dict], max_chars: int = 16_000) -> bool:
    """
    Lightweight context check before sending to a local model.

    Ollama does tokenization internally; this keeps obviously oversized prompts
    out of the request path without pulling tokenizer dependencies into tests.
    """
    total_chars = sum(len(str(message.get("content", ""))) for message in messages)
    return total_chars <= max_chars
