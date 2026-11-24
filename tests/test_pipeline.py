"""Offline tests (no Groq calls). Requires: python scripts/build_index.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.generation.validator import validate
from app.graph.workflow import router
from app.ingestion import ingest
from app.ingestion.cleaner import clean_text
from app.retrieval.bm25_index import tokenize
from app.retrieval.hybrid import retrieve, rrf


def test_cleaner_keeps_formulas():
    assert "S = G e^(-2G)" in clean_text("S = G e^(−2G)\n\n12\n")


def test_ingest_metadata():
    study, pyqs = ingest()
    assert study and pyqs
    assert all(s["metadata"]["unit"].startswith("Unit") for s in study)
    assert all(q["metadata"]["year"] and q["metadata"]["marks"] for q in pyqs)


def test_tokenizer_keeps_technical_terms():
    assert "csma/cd" in tokenize("CSMA/CD protocol")
    assert "belady" in tokenize("Belady's anomaly")


def test_rrf_rewards_agreement():
    fused = rrf([[("a", 1), ("b", 1)], [("b", 1), ("c", 1)]])
    assert fused[0][0] == "b"


def test_hybrid_exact_term():
    top = retrieve("Belady's anomaly", top_k=1)[0]
    assert top["metadata"]["topic"] == "Page Replacement"


def test_subject_filter():
    hits = retrieve("deadlock", subject="DBMS", top_k=5)
    assert all(h["metadata"]["subject"] == "DBMS" for h in hits)


def test_validator():
    assert validate("Explain Belady's anomaly", "Operating Systems")["answerable"]
    assert not validate("Compare RAID 5 and RAID 6", "Operating Systems")["answerable"]


def test_router():
    assert router({"question": "generate practice questions on TCP"})["mode"] == "generate"
    assert router({"question": "what is TCP?"})["mode"] == "qa"
