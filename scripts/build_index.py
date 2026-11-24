"""Ingest PDFs, chunk with every strategy, and build Chroma + BM25 indexes.

Run: python scripts/build_index.py            (all strategies)
     python scripts/build_index.py hierarchical
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import STRATEGIES, chunk_all
from app.config import PROCESSED_DIR
from app.ingestion import ingest
from app.retrieval import bm25_index, vector_store


def tag_pyq_topics(study: list[dict], pyqs: list[dict]) -> None:
    """PYQs rarely state their topic, so assign the closest study-material
    topic from the same subject (and unit, when the paper gives it)."""
    model = vector_store.get_embedder()
    s_emb = model.encode([f"{s['metadata']['topic']}. {s['text']}" for s in study], normalize_embeddings=True)
    q_emb = model.encode([q["text"] for q in pyqs], normalize_embeddings=True)
    for qi, q in enumerate(pyqs):
        qm = q["metadata"]
        best, best_sim = None, -1.0
        for si, s in enumerate(study):
            sm = s["metadata"]
            if sm["subject"] != qm["subject"] or (qm["unit"] and sm["unit"] != qm["unit"]):
                continue
            sim = float(q_emb[qi] @ s_emb[si])
            if sim > best_sim:
                best, best_sim = sm, sim
        if best:
            qm["topic"] = best["topic"]
            qm["unit"] = qm["unit"] or best["unit"]
            qm["unit_title"] = best["unit_title"]


def main():
    strategies = sys.argv[1:] or STRATEGIES
    study, pyqs = ingest()
    print(f"ingested {len(study)} study sections, {len(pyqs)} PYQ questions")
    tag_pyq_topics(study, pyqs)
    (PROCESSED_DIR / "pyqs.json").write_text(json.dumps(pyqs, indent=1))
    for strategy in strategies:
        chunks = chunk_all(strategy, study, pyqs)
        (PROCESSED_DIR / f"chunks_{strategy}.json").write_text(json.dumps(chunks, indent=1))
        vector_store.build(strategy, chunks)
        bm25_index.build(strategy, chunks)
        n_study = sum(c["metadata"]["source_type"] == "study_material" for c in chunks)
        avg = sum(len(c["text"]) for c in chunks if c["metadata"]["source_type"] == "study_material") / max(n_study, 1)
        print(f"  {strategy:<13} {n_study:3d} study chunks (avg {avg:.0f} chars) + {len(chunks) - n_study} PYQ chunks -> indexed")


if __name__ == "__main__":
    main()
