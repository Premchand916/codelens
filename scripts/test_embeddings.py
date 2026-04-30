"""Quick smoke test for the embedding pipeline."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.vectorizer import EmbeddingPipeline
from embeddings.math_utils import cosine_similarity

pipeline = EmbeddingPipeline()

# Test 1: basic embedding shape
v = pipeline.embed_one("Python error handling")
print(f"Shape: {v.shape}")           # (384,)
print(f"Magnitude: {np.linalg.norm(v):.6f}")  # should be ~1.000000

# Test 2: similar sentences score high
a = pipeline.embed_one("Python error handling")
b = pipeline.embed_one("handling Python errors")
c = pipeline.embed_one("JavaScript promises and callbacks")

print(f"\nSimilar pair:    {cosine_similarity(a, b):.4f}")  # expect ~0.90+
print(f"Dissimilar pair: {cosine_similarity(a, c):.4f}")   # expect ~0.20-0.40

# Test 3: batch shape
texts = ["error handling", "async patterns", "security practices"]
batch = pipeline.embed_batch(texts)
print(f"\nBatch shape: {batch.shape}")  # (3, 384)