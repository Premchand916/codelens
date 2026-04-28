# tests/test_diff_parser.py
"""Tests for the unified diff parser."""
from parsers.diff_parser import parse_pr_diff

SAMPLE_DIFF = """diff --git a/calculator.py b/calculator.py
--- a/calculator.py
+++ b/calculator.py
@@ -10,7 +10,9 @@
 def calculate_total(items):
-    total = 0
-    for item in items:
-        total += item.price
+    if not items:
+        return 0
+    total = sum(item.price for item in items)
     return total
"""

def test_basic_parse():
    result = parse_pr_diff(SAMPLE_DIFF, pr_number=1, repo_full_name="test/repo")
    
    assert result.total_files_changed == 1
    assert result.files[0].file_path == "calculator.py"
    assert result.files[0].language == "python"
    assert result.files[0].total_additions == 3
    assert result.files[0].total_deletions == 3
    print("✅ Parse test passed")
    print(f"   File: {result.files[0].file_path}")
    print(f"   +{result.files[0].total_additions} / -{result.files[0].total_deletions}")
    hunk = result.files[0].hunks[0]
    print(f"   Added: {hunk.added_lines}")
    print(f"   Removed: {hunk.removed_lines}")

if __name__ == "__main__":
    test_basic_parse()