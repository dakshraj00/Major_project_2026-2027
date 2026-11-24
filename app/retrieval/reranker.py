"""Cross-encoder reranker: scores (query, passage) pairs jointly, which is
slower but more accurate than comparing separate embeddings."""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import RERANKER_MODEL


@lru_cache
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL, device="cpu")


def rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Adds 'rerank_score' to each chunk and sorts best first."""
    if not chunks:
        return []
    scores = get_reranker().predict([(query, c["text"]) for c in chunks])
    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
