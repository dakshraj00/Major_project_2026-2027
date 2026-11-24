"""Hybrid retrieval: dense (Chroma) + keyword (BM25) merged with Reciprocal
Rank Fusion, then reranked with a cross-encoder.

    RRF score(d) = sum over retrievers of 1 / (RRF_K + rank of d)
"""
import json
from functools import lru_cache

from app.config import CHUNKING_STRATEGY, PROCESSED_DIR, RRF_K, TOP_K_EACH, TOP_K_FINAL
from app.retrieval import bm25_index, vector_store
from app.retrieval.reranker import rerank


@lru_cache
def load_chunks(strategy: str) -> dict[str, dict]:
    path = PROCESSED_DIR / f"chunks_{strategy}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: python scripts/build_index.py")
    return {c["id"]: c for c in json.loads(path.read_text())}


def rrf(result_lists: list[list[tuple[str, float]]], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(query: str, subject: str = "", source_type: str = "study_material",
             mode: str = "hybrid", top_k: int = TOP_K_FINAL, strategy: str = CHUNKING_STRATEGY,
             use_reranker: bool = True) -> list[dict]:
    """mode: 'hybrid' | 'vector' | 'bm25'. Returns chunk dicts (copies) best first."""
    filters = {"subject": subject, "source_type": source_type}
    lists = []
    if mode in ("hybrid", "vector"):
        lists.append(vector_store.search(strategy, query, TOP_K_EACH, filters))
    if mode in ("hybrid", "bm25"):
        lists.append(bm25_index.search(strategy, query, TOP_K_EACH, filters))
    fused = rrf(lists)
    chunks = load_chunks(strategy)
    candidates = []
    for doc_id, score in fused:
        c = dict(chunks[doc_id])
        c["fusion_score"] = score
        candidates.append(c)
    if use_reranker:
        candidates = rerank(query, candidates)
    return candidates[:top_k]


def context_text(chunk: dict) -> str:
    """Hierarchical chunks hand the LLM their full parent section."""
    return chunk["metadata"].get("parent_text") or chunk["text"]


def citation(chunk: dict) -> dict:
    md = chunk["metadata"]
    return {"subject": md["subject"], "unit": md.get("unit", ""), "topic": md.get("topic", ""),
            "source_type": md["source_type"], "source_file": md["source_file"],
            "page": md["page"], "year": md.get("year", 0), "marks": md.get("marks", 0),
            "score": round(chunk.get("rerank_score", chunk.get("fusion_score", 0.0)), 3),
            "text": chunk["text"]}
