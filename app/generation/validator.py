"""Answerability check: re-query retrieval with each generated question. If no
study-material passage scores above the threshold, the material cannot answer
it, so the question is rejected."""
from app.config import ANSWERABLE_THRESHOLD
from app.retrieval.hybrid import citation, retrieve


def validate(question: str, subject: str, threshold: float = ANSWERABLE_THRESHOLD) -> dict:
    hits = retrieve(question, subject=subject, source_type="study_material", top_k=3)
    best = hits[0]["rerank_score"] if hits else float("-inf")
    ok = best >= threshold
    return {
        "answerable": ok,
        "score": round(best, 3),
        "reason": (f"supported by {hits[0]['metadata']['topic']} (score {best:.2f})" if ok
                   else f"no supporting passage (best score {best:.2f} < {threshold})"),
        "evidence": citation(hits[0]) if hits else None,
    }
