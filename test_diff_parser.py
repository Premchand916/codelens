from parsers.diff_parser import parse_pr_diff
from parsers.code_chunker import chunk_pr_diff

SAMPLE_DIFF = '''diff --git a/calculator.py b/calculator.py
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
'''

if __name__ == "__main__":
    pr_diff = parse_pr_diff(SAMPLE_DIFF, pr_number=1, repo_full_name='test/repo')
    chunks = chunk_pr_diff(pr_diff)
    
    print(f'Total chunks: {len(chunks)}')
    for c in chunks:
        print(f'  {c.chunk_id} | {c.char_count} chars | line {c.start_line}')
        print(f'  Content preview: {c.content[:100]}')