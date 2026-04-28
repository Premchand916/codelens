"""
Unified diff parser for pull request patches.
"""
import re
from pathlib import PurePosixPath

from models.review import DiffHunk, FileDiff, PRDiff


HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_lines>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_lines>\d+))? @@"
)

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
}


def detect_language(file_path: str) -> str:
    """Infer a readable language name from a file extension."""
    suffix = PurePosixPath(file_path).suffix.lower()
    return LANGUAGE_BY_EXTENSION.get(suffix, "unknown")


def normalize_diff_path(path: str) -> str:
    """Convert Git diff paths like b/app/main.py into app/main.py."""
    path = path.strip()
    if path == "/dev/null":
        return path
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def parse_hunk(hunk_lines: list[str]) -> DiffHunk:
    if not hunk_lines:
        raise ValueError("Cannot parse an empty diff hunk")

    header = hunk_lines[0]
    match = HUNK_HEADER_RE.match(header)
    if not match:
        raise ValueError(f"Invalid hunk header: {header}")

    added_lines: list[str] = []
    removed_lines: list[str] = []
    context_lines: list[str] = []

    for line in hunk_lines[1:]:
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed_lines.append(line[1:])
        elif line.startswith(" "):
            context_lines.append(line[1:])
        elif line.startswith("\\"):
            continue
        else:
            context_lines.append(line)

    return DiffHunk(
        old_start=int(match.group("old_start")),
        new_start=int(match.group("new_start")),
        old_lines=int(match.group("old_lines") or "1"),
        new_lines=int(match.group("new_lines") or "1"),
        added_lines=added_lines,
        removed_lines=removed_lines,
        context_lines=context_lines,
        raw_hunk="\n".join(hunk_lines),
    )


def parse_pr_diff(
    diff_text: str,
    pr_number: int,
    repo_full_name: str,
) -> PRDiff:
    """
    Parse a GitHub unified PR diff into structured review models.

    The parser focuses on the data the review pipeline needs: changed files,
    hunks, added lines, removed lines, and aggregate change counts.
    """
    files: list[FileDiff] = []
    current_path: str | None = None
    current_hunks: list[DiffHunk] = []
    current_hunk_lines: list[str] = []
    is_new_file = False
    is_deleted_file = False

    def finish_hunk() -> None:
        nonlocal current_hunk_lines
        if current_hunk_lines:
            current_hunks.append(parse_hunk(current_hunk_lines))
            current_hunk_lines = []

    def finish_file() -> None:
        nonlocal current_path, current_hunks, is_new_file, is_deleted_file
        finish_hunk()
        if current_path is None:
            return

        total_additions = sum(len(hunk.added_lines) for hunk in current_hunks)
        total_deletions = sum(len(hunk.removed_lines) for hunk in current_hunks)
        files.append(
            FileDiff(
                file_path=current_path,
                language=detect_language(current_path),
                hunks=current_hunks,
                total_additions=total_additions,
                total_deletions=total_deletions,
                is_new_file=is_new_file,
                is_deleted_file=is_deleted_file,
            )
        )

        current_path = None
        current_hunks = []
        is_new_file = False
        is_deleted_file = False

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            finish_file()
            parts = line.split()
            current_path = normalize_diff_path(parts[3]) if len(parts) >= 4 else None
            continue

        if current_path is None:
            continue

        if line.startswith("new file mode"):
            is_new_file = True
            continue

        if line.startswith("deleted file mode"):
            is_deleted_file = True
            continue

        if line.startswith("+++ "):
            new_path = normalize_diff_path(line[4:])
            if new_path != "/dev/null":
                current_path = new_path
            continue

        if line.startswith("@@ "):
            finish_hunk()
            current_hunk_lines = [line]
            continue

        if current_hunk_lines:
            current_hunk_lines.append(line)

    finish_file()

    total_additions = sum(file_diff.total_additions for file_diff in files)
    total_deletions = sum(file_diff.total_deletions for file_diff in files)

    return PRDiff(
        pr_number=pr_number,
        repo_full_name=repo_full_name,
        files=files,
        total_files_changed=len(files),
        total_additions=total_additions,
        total_deletions=total_deletions,
    )
