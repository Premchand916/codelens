"""
parsers/diff_parser.py — Unified diff format parser.
Converts raw GitHub diff text into structured FileDiff objects.

Unified diff format primer:
  diff --git a/foo.py b/foo.py     ← file header
  --- a/foo.py                     ← old file marker
  +++ b/foo.py                     ← new file marker
  @@ -10,7 +10,9 @@               ← hunk header
  -removed line                    ← deletion
  +added line                      ← addition
   context line                    ← unchanged (space prefix)
"""
import re
import logging
from pathlib import Path

from models.review import DiffHunk, FileDiff, PRDiff

logger = logging.getLogger(__name__)

# Map file extensions → language names
# Used later to select the right knowledge base docs for retrieval
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".jsx":  "javascript",
    ".tsx":  "typescript",
    ".java": "java",
    ".go":   "go",
    ".rs":   "rust",
    ".md":   "markdown",
    ".yml":  "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".sh":   "bash",
}


def detect_language(file_path: str) -> str:
    """Detect programming language from file extension."""
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, "unknown")

def parse_hunk_header(header_line: str) -> tuple[int, int, int, int]:
    """
    Parse @@ -old_start,old_lines +new_start,new_lines @@ format.
    
    [INTERNAL] The regex captures 4 numbers from the hunk header.
    Example: '@@ -10,7 +10,9 @@'
    → old_start=10, old_lines=7, new_start=10, new_lines=9
    Edge case: '@@ -1 +1,4 @@' means old_lines=1 (comma+count omitted when =1)
    """
    # Pattern: @@ -NUM,NUM +NUM,NUM @@ (commas+second num optional)
    pattern = r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@'
    match = re.search(pattern, header_line)
    
    if not match:
        raise ValueError(f"Cannot parse hunk header: {header_line}")
    
    old_start  = int(match.group(1))
    old_lines  = int(match.group(2)) if match.group(2) else 1
    new_start  = int(match.group(3))
    new_lines  = int(match.group(4)) if match.group(4) else 1
    
    return old_start, old_lines, new_start, new_lines


def parse_hunk(hunk_lines: list[str]) -> DiffHunk:
    """
    Convert raw hunk lines into a structured DiffHunk.
    First line is always the @@ header.
    """
    header = hunk_lines[0]
    old_start, old_lines, new_start, new_lines = parse_hunk_header(header)
    
    added_lines   = []
    removed_lines = []
    context_lines = []
    
    for line in hunk_lines[1:]:  # skip the @@ header
        if line.startswith('+'):
            added_lines.append(line[1:])    # strip the + prefix
        elif line.startswith('-'):
            removed_lines.append(line[1:])  # strip the - prefix
        elif line.startswith(' '):
            context_lines.append(line[1:])  # strip the space prefix
        # Lines starting with \ are "no newline at end of file" — skip
    
    return DiffHunk(
        old_start=old_start,
        new_start=new_start,
        old_lines=old_lines,
        new_lines=new_lines,
        added_lines=added_lines,
        removed_lines=removed_lines,
        context_lines=context_lines,
        raw_hunk="\n".join(hunk_lines),
    )

def parse_file_diff(file_block: str) -> FileDiff | None:
    """
    Parse one file's complete diff block into a FileDiff.
    A file block starts with 'diff --git' and contains all its hunks.
    """
    lines = file_block.strip().split('\n')
    
    # Extract file path from '+++ b/path/to/file' line
    file_path = None
    is_new_file = False
    is_deleted_file = False
    
    for line in lines:
        if line.startswith('+++ b/'):
            file_path = line[6:]   # strip '+++ b/'
        elif line.startswith('+++ /dev/null'):
            is_deleted_file = True
        elif line.startswith('--- /dev/null'):
            is_new_file = True
    
    if not file_path or file_path == '/dev/null':
        logger.warning("Could not extract file path from diff block")
        return None
    
    # Split into individual hunks — each starts with @@
    # [INTERNAL] We split on @@ but keep the delimiter by using a lookahead
    hunk_blocks: list[list[str]] = []
    current_hunk: list[str] = []
    
    for line in lines:
        if line.startswith('@@'):
            if current_hunk:
                hunk_blocks.append(current_hunk)
            current_hunk = [line]
        elif current_hunk:
            current_hunk.append(line)
    
    if current_hunk:
        hunk_blocks.append(current_hunk)
    
    if not hunk_blocks:
        logger.info(f"No hunks found in {file_path} — skipping")
        return None
    
    hunks = [parse_hunk(block) for block in hunk_blocks]
    total_additions = sum(len(h.added_lines) for h in hunks)
    total_deletions = sum(len(h.removed_lines) for h in hunks)
    
    return FileDiff(
        file_path=file_path,
        language=detect_language(file_path),
        hunks=hunks,
        total_additions=total_additions,
        total_deletions=total_deletions,
        is_new_file=is_new_file,
        is_deleted_file=is_deleted_file,
    )


def parse_pr_diff(raw_diff: str, pr_number: int, repo_full_name: str) -> PRDiff:
    """
    Parse complete PR diff (multiple files) into structured PRDiff.
    
    [INTERNAL] GitHub's diff format separates files with 'diff --git a/... b/...'
    We split on this marker to get individual file blocks, then parse each.
    """
    if not raw_diff.strip():
        logger.warning(f"Empty diff for PR #{pr_number}")
        return PRDiff(
            pr_number=pr_number,
            repo_full_name=repo_full_name,
            files=[],
            total_files_changed=0,
            total_additions=0,
            total_deletions=0,
        )
    
    # Split on file boundaries — 'diff --git' starts each file block
    # [INTERNAL] re.split with a capture group keeps the delimiter in results
    file_blocks = re.split(r'(?=diff --git )', raw_diff)
    file_blocks = [b for b in file_blocks if b.strip()]
    
    files: list[FileDiff] = []
    for block in file_blocks:
        parsed = parse_file_diff(block)
        if parsed:
            files.append(parsed)
    
    return PRDiff(
        pr_number=pr_number,
        repo_full_name=repo_full_name,
        files=files,
        total_files_changed=len(files),
        total_additions=sum(f.total_additions for f in files),
        total_deletions=sum(f.total_deletions for f in files),
    )

