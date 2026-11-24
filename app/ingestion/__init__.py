from app.ingestion.cleaner import clean_text
from app.ingestion.loader import load_all
from app.ingestion.metadata import pyq_questions, study_sections


def ingest() -> tuple[list[dict], list[dict]]:
    """Returns (study_sections, pyq_records)."""
    pages = load_all()
    for p in pages:
        p["text"] = clean_text(p["text"])
    study = study_sections([p for p in pages if p["source_type"] == "study_material"])
    pyqs = pyq_questions([p for p in pages if p["source_type"] == "pyq"])
    return study, pyqs
