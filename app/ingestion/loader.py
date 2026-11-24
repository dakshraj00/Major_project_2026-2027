"""Read PDFs page by page with PyMuPDF.

Folder layout decides the metadata we know for free:
    data/raw/<pyqs|study_material>/<subject_folder>/<file>.pdf
"""
from pathlib import Path

import fitz  # PyMuPDF

from app.config import PYQ_DIR, STUDY_DIR


def subject_from_folder(folder: str) -> str:
    """'operating_systems' -> 'Operating Systems' (keeps short names like DBMS upper)."""
    if len(folder) <= 5:
        return folder.upper()
    return folder.replace("_", " ").title()


def load_pdf(path: Path, source_type: str) -> list[dict]:
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append({
                "text": page.get_text("text"),
                "source_file": path.name,
                "page": i,
                "subject": subject_from_folder(path.parent.name),
                "source_type": source_type,
            })
    return pages


def load_all() -> list[dict]:
    pages = []
    for base, kind in ((STUDY_DIR, "study_material"), (PYQ_DIR, "pyq")):
        for pdf in sorted(base.rglob("*.pdf")):
            pages.extend(load_pdf(pdf, kind))
    return pages
