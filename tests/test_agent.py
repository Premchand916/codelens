# tests/test_agent.py

from agent.decision import should_skip_file, classify_file_priority
from agent.guardrails import extract_changed_line_numbers, validate_comment
from llm.output_parser import ReviewComment


def test_skip_generated_files():
    assert should_skip_file("__pycache__/auth.cpython-311.pyc", 10)[0] is True
    assert should_skip_file("package-lock.json", 50)[0] is True
    assert should_skip_file("migrations/0042_add_user.py", 30)[0] is True


def test_skip_large_diff():
    assert should_skip_file("src/auth.py", 201)[0] is True
    assert should_skip_file("src/auth.py", 200)[0] is False


def test_no_skip_source_files():
    assert should_skip_file("src/auth.py", 10)[0] is False
    assert should_skip_file("api/routes.py", 50)[0] is False


def test_priority_classification():
    assert classify_file_priority("src/auth.py") == "high"
    assert classify_file_priority("tests/test_auth.py") == "low"
    assert classify_file_priority("api/routes.py") == "normal"


def test_extract_changed_lines():
    diff = (
        "@@ -10,4 +10,5 @@\n"
        " def foo():\n"
        "+    x = 1\n"
        "+    return x\n"
        "-    pass\n"
        " \n"
    )
    lines = extract_changed_line_numbers(diff)
    assert 11 in lines   # +    x = 1
    assert 12 in lines   # +    return x
    assert 10 not in lines  # context line


def test_guardrail_rejects_hallucinated_line():
    diff = "@@ -10,3 +10,4 @@\n+    x = 1\n context\n"
    comment = ReviewComment(
        should_comment=True,
        severity="warning",
        category="bug",
        line_number=99,   # not in diff
        comment="This line has an issue with the logic here",
        suggested_fix=None,
    )
    valid, reason = validate_comment(comment, diff)
    assert valid is False
    assert "99" in reason


def test_guardrail_accepts_valid_comment():
    diff = "@@ -10,3 +10,4 @@\n+    x = 1\n context\n"
    comment = ReviewComment(
        should_comment=True,
        severity="warning",
        category="bug",
        line_number=10,
        comment="Variable x is assigned but never used in this scope",
        suggested_fix=None,
    )
    valid, _ = validate_comment(comment, diff)
    assert valid is True