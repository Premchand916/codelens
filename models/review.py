"""
models/review.py — Data models for PR review pipeline.
Defines the shape of parsed diffs and review requests/results.
"""
from pydantic import BaseModel, Field
from typing import Literal


class DiffHunk(BaseModel):
    """
    One contiguous block of changes in a file.
    A file can have multiple hunks if changes are spread across it.
    """
    old_start: int          # line number in original file
    new_start: int          # line number in new file
    old_lines: int          # how many lines from original
    new_lines: int          # how many lines in new version
    added_lines: list[str]  # lines starting with +
    removed_lines: list[str]# lines starting with -
    context_lines: list[str]# unchanged lines (for context)
    raw_hunk: str           # original raw text of this hunk


class FileDiff(BaseModel):
    """
    All changes in a single file within the PR.
    """
    file_path: str
    language: str                    # detected from extension
    hunks: list[DiffHunk]
    total_additions: int
    total_deletions: int
    is_new_file: bool = False
    is_deleted_file: bool = False


class PRDiff(BaseModel):
    """
    Complete parsed diff for an entire PR.
    """
    pr_number: int
    repo_full_name: str
    files: list[FileDiff]
    total_files_changed: int
    total_additions: int
    total_deletions: int