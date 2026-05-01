"""
eval/evaluator.py
LLM-as-judge pattern: use one LLM call to evaluate another's output.
Measures relevance, faithfulness, specificity of generated review comments.
"""

import json
import logging
from dataclasses import dataclass

from llm.client import OllamaClient
from llm.output_parser import extract_json

logger = logging.getLogger(__name__)


@dataclass
class ReviewEvalResult:
    """Evaluation scores for one generated review comment."""
    comment_id: str
    relevance: float        # 0-1: is the comment relevant to the diff?
    faithfulness: float     # 0-1: is the comment grounded in the diff (not hallucinated)?
    specificity: float      # 0-1: does the comment reference actual code, not generic advice?
    overall: float          # weighted average
    judge_reasoning: str    # why the judge scored it this way


# [INTERNAL] LLM-as-judge: use a second LLM call to score the first LLM's output.
# The judge model sees: the diff, the retrieved context, the generated comment.
# It returns scores 0-1 for each dimension.
# Why not human eval? Too slow for CI. Why not regex? Can't measure semantics.
# LLM-as-judge is ~85% correlated with human judgment at a fraction of the cost.
# [INTERVIEW] "What's the risk of LLM-as-judge?"
# → "Self-serving bias — same model tends to rate its own outputs highly.
#    Mitigation: use a DIFFERENT model as judge, or use a stronger model
#    (e.g. judge with llama3.1:8b, generate with deepseek-coder:6.7b)."
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
    """
    Evaluate one review comment using LLM-as-judge.
    Uses a separate client/model from the generator to reduce self-serving bias.
    """
    context = "\n".join(f"[Doc {i+1}] {d}" for i, d in enumerate(retrieved_docs[:2]))

    user_message = f"""## Diff

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

        # Weighted average — faithfulness weighted highest (hallucination is worst failure)
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
            judge_reasoning=f"Evaluation failed: {e}",
        )