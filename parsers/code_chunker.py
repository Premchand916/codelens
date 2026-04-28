"""
parsers/code_chunker.py — Breaks parsed diffs into LLM-sized review chunks.

Strategy:
  1. Each hunk becomes one chunk (natural logical boundary)
  2. File path + language prepended to every chunk (LLM needs this context)
  3. Chunks exceeding token limit are split by line count
  4. Tiny hunks from the same file are merged to reduce LLM calls
"""
import logging
from dataclasses import dataclass

from models.review import PRDiff, FileDiff, DiffHunk

logger = logging.getLogger(__name__)

# Approximate token limits
# [INTERNAL] Tokens ≠ words. On average, 1 token ≈ 4 characters in English code.
# We use character count as a cheap proxy — no tokenizer needed.
# 4000 token budget → ~16000 characters per chunk (conservative, leaves room for prompt)
MAX_CHUNK_CHARS = 16_000
MIN_CHUNK_CHARS = 200   # hunks smaller than this get merged with neighbors


@dataclass
class ReviewChunk:
    """
    One unit of work for the LLM reviewer.
    Contains everything the LLM needs to review this piece of code.
    """
    chunk_id: str           # "pr1_calculator.py_hunk0"
    file_path: str
    language: str
    hunk_index: int
    content: str            # formatted text sent to LLM
    added_lines: list[str]  # just the + lines (for guardrail checks later)
    start_line: int         # line number in new file (for posting comments)
    char_count: int


def format_chunk_content(file_path: str, language: str, hunk: DiffHunk) -> str:
    """
    Format a hunk into the text we send to the LLM.

    [INTERNAL] We give the LLM:
    1. File path — so it knows what kind of file it's reviewing
    2. Language — so it applies the right best practices
    3. The raw hunk — +/- lines with context
    This is the "prompt context" that feeds into RAG + LLM in Session 5.
    """
    return (
        f"File: {file_path}\n"
        f"Language: {language}\n"
        f"Changes:\n"
        f"{hunk.raw_hunk}\n"
    )


def chunk_file_diff(file_diff: FileDiff, pr_number: int) -> list[ReviewChunk]:
    """
    Convert one FileDiff into a list of ReviewChunks.
    One hunk = one chunk (unless too large or too small).
    """
    # Skip files we don't have knowledge base docs for
    # [SURFACE] No point reviewing YAML or JSON with a Python best-practices agent
    if file_diff.language == "unknown":
        logger.info(f"Skipping unknown language file: {file_diff.file_path}")
        return []

    chunks: list[ReviewChunk] = []
    pending_small_hunks: list[DiffHunk] = []   # buffer for merging tiny hunks

    def flush_pending(hunk_index: int) -> None:
        """Merge accumulated small hunks into one chunk."""
        if not pending_small_hunks:
            return
        # Merge by concatenating raw hunks
        merged_raw = "\n".join(h.raw_hunk for h in pending_small_hunks)
        merged_hunk = DiffHunk(
            old_start=pending_small_hunks[0].old_start,
            new_start=pending_small_hunks[0].new_start,
            old_lines=sum(h.old_lines for h in pending_small_hunks),
            new_lines=sum(h.new_lines for h in pending_small_hunks),
            added_lines=[l for h in pending_small_hunks for l in h.added_lines],
            removed_lines=[l for h in pending_small_hunks for l in h.removed_lines],
            context_lines=[l for h in pending_small_hunks for l in h.context_lines],
            raw_hunk=merged_raw,
        )
        content = format_chunk_content(
            file_diff.file_path, file_diff.language, merged_hunk
        )
        chunks.append(ReviewChunk(
            chunk_id=f"pr{pr_number}_{file_diff.file_path}_merged{hunk_index}",
            file_path=file_diff.file_path,
            language=file_diff.language,
            hunk_index=hunk_index,
            content=content,
            added_lines=merged_hunk.added_lines,
            start_line=merged_hunk.new_start,
            char_count=len(content),
        ))
        pending_small_hunks.clear()

    for i, hunk in enumerate(file_diff.hunks):
        content = format_chunk_content(file_diff.file_path, file_diff.language, hunk)
        char_count = len(content)

        if char_count < MIN_CHUNK_CHARS:
            # Too small — buffer it, merge with next hunk
            # [INTERNAL] Sending 3-line hunks individually wastes LLM calls.
            # A tiny import change + a tiny docstring change = one review call.
            pending_small_hunks.append(hunk)
            continue

        # Flush any pending small hunks before this normal-sized one
        flush_pending(i)

        if char_count > MAX_CHUNK_CHARS:
            # Too large — split by lines
            # [INTERNAL] Rare but happens in generated files or large refactors.
            # We split the raw_hunk lines into sub-chunks.
            logger.warning(
                f"Hunk {i} in {file_diff.file_path} is {char_count} chars — splitting"
            )
            lines = hunk.raw_hunk.split('\n')
            sub_chunks = [
                lines[j:j+200] for j in range(0, len(lines), 200)
            ]
            for k, sub in enumerate(sub_chunks):
                sub_content = (
                    f"File: {file_diff.file_path}\n"
                    f"Language: {file_diff.language}\n"
                    f"Changes (part {k+1}/{len(sub_chunks)}):\n"
                    + "\n".join(sub)
                )
                chunks.append(ReviewChunk(
                    chunk_id=f"pr{pr_number}_{file_diff.file_path}_hunk{i}_part{k}",
                    file_path=file_diff.file_path,
                    language=file_diff.language,
                    hunk_index=i,
                    content=sub_content,
                    added_lines=hunk.added_lines,
                    start_line=hunk.new_start,
                    char_count=len(sub_content),
                ))
        else:
            # Normal size — one hunk, one chunk
            chunks.append(ReviewChunk(
                chunk_id=f"pr{pr_number}_{file_diff.file_path}_hunk{i}",
                file_path=file_diff.file_path,
                language=file_diff.language,
                hunk_index=i,
                content=content,
                added_lines=hunk.added_lines,
                start_line=hunk.new_start,
                char_count=char_count,
            ))

    flush_pending(len(file_diff.hunks))  # flush any remaining small hunks
    return chunks


def chunk_pr_diff(pr_diff: PRDiff) -> list[ReviewChunk]:
    """
    Convert entire PRDiff into ordered list of ReviewChunks.
    This is what the LLM agent processes — one chunk at a time.
    """
    all_chunks: list[ReviewChunk] = []

    for file_diff in pr_diff.files:
        if file_diff.is_deleted_file:
            # [SURFACE] No point reviewing deleted files — code is gone
            logger.info(f"Skipping deleted file: {file_diff.file_path}")
            continue

        file_chunks = chunk_file_diff(file_diff, pr_diff.pr_number)
        all_chunks.extend(file_chunks)
        logger.info(
            f"{file_diff.file_path} → {len(file_chunks)} chunks "
            f"({file_diff.total_additions}+ {file_diff.total_deletions}-)"
        )

    logger.info(f"PR #{pr_diff.pr_number} → {len(all_chunks)} total chunks")
    return all_chunks