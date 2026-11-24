"""Grounded Q&A: retrieve -> prompt with numbered passages -> answer with citations."""
from app.config import QA_MIN_SCORE
from app.generation.llm import chat, load_prompt
from app.retrieval.hybrid import citation, context_text, retrieve

NOT_FOUND = "Not found in the material."


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        md = c["metadata"]
        parts.append(f"[{i}] ({md['subject']}, {md['unit']}, {md['topic']}, "
                     f"{md['source_file']} p.{md['page']})\n{context_text(c)}")
    return "\n\n".join(parts)


def dedupe_parents(chunks: list[dict]) -> list[dict]:
    """Hierarchical children of the same parent would repeat the same context."""
    seen, out = set(), []
    for c in chunks:
        key = c["metadata"].get("parent_id", c["id"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def answer(question: str, subject: str = "", mode: str = "hybrid") -> dict:
    chunks = dedupe_parents(retrieve(question, subject=subject, mode=mode, use_reranker=(mode != "vector")))
    best = max((c.get("rerank_score", 0.0) for c in chunks), default=-99)
    # keep only passages that are plausibly relevant (always keep the best one)
    chunks = [c for i, c in enumerate(chunks) if i == 0 or c.get("rerank_score", 0.0) >= QA_MIN_SCORE]
    if not chunks or best < QA_MIN_SCORE:
        return {"answer": NOT_FOUND, "found": False, "sources": [], "contexts": [],
                "reason": f"best retrieval score {best:.2f} < {QA_MIN_SCORE}"}
    prompt = load_prompt("qa_template.txt").format(context=format_context(chunks), question=question)
    text = chat(prompt)
    found = NOT_FOUND.lower().rstrip(".") not in text.lower()
    return {"answer": text, "found": found,
            "sources": [citation(c) for c in chunks] if found else [],
            "contexts": [context_text(c) for c in chunks]}
