"""
vectorizer.py — Embedding pipeline for CodeLens knowledge base.

Converts text documents into 384-dim vectors using sentence-transformers.
These vectors are stored and searched during PR review to retrieve
relevant best practices.
"""

import logging
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# [SURFACE] Model name — downloads ~90MB on first run, cached after
# [INTERNAL] all-MiniLM-L6-v2: 6 transformer layers (vs 12 in full BERT),
# projected to 384 dims. "Distilled" — trained to mimic a larger model.
# Fast enough for real-time retrieval, good enough for code review context.
MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingPipeline:
    """Loads model once, embeds documents and queries."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        logger.info(f"Loading embedding model: {model_name}")
        # [INTERNAL] SentenceTransformer wraps the tokenizer + transformer + pooling.
        # Loading takes ~1-2s. We load ONCE at startup, reuse for every embed call.
        # This is why it's a class — shared state (the loaded model).
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension() # 384
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed_one(self, text: str) -> np.ndarray:
        """
        Embed a single string → normalized 384-dim vector.
        Used for query embedding at retrieval time.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text")

        # [INTERNAL] encode() runs the full pipeline:
        # tokenize → positional encoding → 6 attention layers → mean pool → 384 dims
        # normalize=True divides by magnitude → unit vector (magnitude = 1.0)
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.astype(np.float32)  # FAISS expects float32

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts efficiently.
        Returns shape (N, 384) — one row per text.

        # [INTERNAL] batch_size=32 means 32 texts go through the transformer
        # simultaneously. GPU/MPS parallelism (your M5 has MPS — Apple's GPU).
        # Batch of 32 ≈ same time as batch of 1. Don't embed one-by-one in a loop.
        """
        if not texts:
            raise ValueError("Cannot embed empty list")

        texts = [t.strip() for t in texts]
        empty = [i for i, t in enumerate(texts) if not t]
        if empty:
            raise ValueError(f"Empty strings at indices: {empty}")

        logger.info(f"Embedding {len(texts)} documents (batch_size={batch_size})")

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # unit vectors — ready for dot product search
            show_progress_bar=len(texts) > 10,  # progress bar only for large batches
        )
        return vectors.astype(np.float32)
    
def load_knowledge_base(kb_dir: str | Path) -> tuple[list[str], list[str]]:
    """
    Walk the knowledge_base/ directory, load all .md files.
    
    Returns:
        texts: list of document contents
        metadata: list of file paths (used to tell the LLM where a best practice came from)
    
    # [INTERNAL] We return metadata separately because FAISS only stores vectors.
    # The index position links them: vector[i] corresponds to metadata[i].
    # When retrieval returns index 42, we look up metadata[42] to get the source.
    """
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_path}")

    texts: list[str] = []
    metadata: list[str] = []

    # Walk all subdirectories — python/, javascript/, general/, security/
    for md_file in sorted(kb_path.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            logger.warning(f"Skipping empty file: {md_file}")
            continue
        texts.append(content)
        # Store relative path — "python/error_handling.md" is more useful than absolute
        metadata.append(str(md_file.relative_to(kb_path)))
        logger.debug(f"Loaded: {md_file.name} ({len(content)} chars)")

    logger.info(f"Loaded {len(texts)} documents from knowledge base")
    return texts, metadata


def embed_knowledge_base(
    kb_dir: str | Path,
    output_path: str | Path,
    pipeline: EmbeddingPipeline,
) -> None:
    """
    Embed entire knowledge base and save to disk as .npz

    # [INTERNAL] .npz = NumPy's compressed archive format.
    # Stores vectors + metadata together. Loaded once at startup.
    # Alternative: store in FAISS index file (.index) — we'll do that in Session 4.
    # For now, raw .npz teaches you what FAISS actually stores internally.
    """
    texts, metadata = load_knowledge_base(kb_dir)
    vectors = pipeline.embed_batch(texts)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        vectors=vectors,       # shape (N, 384)
        metadata=metadata,     # list of file paths
        texts=texts,           # original text (needed for reranking later)
    )
    logger.info(f"Saved {len(texts)} embeddings → {output_path}")