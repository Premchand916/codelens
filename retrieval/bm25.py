# retrieval/bm25.py
"""
BM25 sparse retrieval — keyword-based, no embeddings.
Learns: when BM25 beats dense retrieval and why.
"""
import logging
import math
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)

DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "code",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "to",
    "when",
    "with",
}


@dataclass
class BM25Result:
    rank: int
    score: float
    source: str
    text: str


class BM25Retriever:
    """
    BM25 from scratch — no rank-bm25 library yet.
    Build the formula by hand first so you own it.

    # [INTERNAL] BM25 formula per document d for query q:
    # score(d,q) = Σ IDF(t) × [tf(t,d) × (k1+1)] / [tf(t,d) + k1×(1 - b + b×|d|/avgdl)]
    #
    # tf(t,d)  = how many times term t appears in doc d
    # IDF(t)   = log((N - df + 0.5) / (df + 0.5) + 1)
    #            N=total docs, df=docs containing term t
    # |d|      = doc length in tokens
    # avgdl    = average doc length across corpus
    # k1 (1.5) = term frequency saturation. After k1 hits, more tf adds little.
    # b  (0.75) = length normalization. b=1: full normalization. b=0: none.
    #
    # [INTERVIEW] "When does BM25 beat dense retrieval?"
    # → "Exact keyword matches: function names, error codes, variable names.
    #    'KeyError' in a query will score high in BM25 for docs containing
    #    'KeyError', but dense embeddings might match semantically similar
    #    docs that never use that word. For code review, both matter."
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: set[str] | None = None,
    ):
        self.k1 = k1
        self.b = b
        self.stopwords = DEFAULT_STOPWORDS if stopwords is None else stopwords
        self.corpus: list[list[str]] = []       # tokenized docs
        self.texts: list[str] = []              # raw texts
        self.metadata: list[dict | str] = []
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0

    # ------------------------------------------------------------------
    # Tokenizer — dead simple, no NLTK needed
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """
        Lowercase + split on non-alphanumeric.
        Good enough for KB docs. Real systems use stemming/lemmatization.
        """
        import re
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return [token for token in tokens if token not in self.stopwords]

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index(self, texts: list[str], metadata: list[dict | str]) -> None:
        """Tokenize corpus, compute IDF scores, compute avgdl."""
        if not texts:
            raise ValueError("Cannot index an empty corpus")
        if len(texts) != len(metadata):
            raise ValueError(
                f"texts and metadata must have the same length: "
                f"{len(texts)} != {len(metadata)}"
            )

        self.texts = texts
        self.metadata = metadata
        self.corpus = [self._tokenize(t) for t in texts]

        N = len(self.corpus)
        self.avgdl = sum(len(doc) for doc in self.corpus) / N or 1.0

        # IDF: penalizes terms that appear in many docs (like "the", "is")
        # rewards rare terms (like "KeyError", "async", "credential")
        all_terms = set(term for doc in self.corpus for term in doc)
        for term in all_terms:
            df = sum(1 for doc in self.corpus if term in doc)
            # Smoothed IDF — avoids log(0) when df=N
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

        logger.info(f"BM25 indexed {N} docs, vocab size {len(self.idf)}")

    def _source_for_index(self, idx: int) -> str:
        """Return source path for both legacy dict metadata and string metadata."""
        metadata = self.metadata[idx]
        if isinstance(metadata, dict):
            return str(metadata.get("source", "unknown"))
        return str(metadata)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[BM25Result]:
        """Score all docs against query, return top-k."""
        if not self.corpus:
            raise RuntimeError("BM25 index is empty. Call index() first.")
        if k <= 0:
            return []

        query_terms = self._tokenize(query)
        scores = []

        for doc_idx, doc_tokens in enumerate(self.corpus):
            tf_map = Counter(doc_tokens)
            doc_len = len(doc_tokens)
            score = 0.0

            for term in query_terms:
                if term not in self.idf:
                    continue  # unseen term — skip

                tf = tf_map.get(term, 0)

                # BM25 numerator: tf with saturation
                numerator = tf * (self.k1 + 1)

                # BM25 denominator: tf + length normalization term
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avgdl
                )

                score += self.idf[term] * (numerator / denominator)

            scores.append((doc_idx, score))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(scores[:min(k, len(scores))]):
            results.append(BM25Result(
                rank=rank + 1,
                score=score,
                source=self._source_for_index(idx),
                text=self.texts[idx],
            ))
        return results
