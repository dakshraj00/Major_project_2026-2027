"""Turn cleaned pages into structured records with metadata.

Study material -> sections: one per (unit, topic), with the page it starts on.
PYQ papers     -> one record per question with marks, year, unit, question type.
"""
import re

UNIT_RE = re.compile(r"^Unit\s+(\d+)\s*[:\-]\s*(.+)$", re.I)
TOPIC_RE = re.compile(r"^Topic\s*[:\-]\s*(.+)$", re.I)
YEAR_RE = re.compile(r"\b(20\d{2})\b")
QUESTION_RE = re.compile(r"Q\s*(\d+)[.)]\s*(.+?)(?=\nQ\s*\d+[.)]|\Z)", re.S)
MARKS_RE = re.compile(r"\[(\d+)\s*marks?\]", re.I)
QUNIT_RE = re.compile(r"\(Unit\s*(\d+)\)", re.I)

QUESTION_TYPES = [
    ("numerical", r"\b(calculate|compute|find|how many|apply|check whether|given)\b"),
    ("derivation", r"\b(derive|prove|show that)\b"),
    ("comparison", r"\b(compare|differentiate|difference|vs)\b"),
    ("definition", r"\b(define|what is|state)\b"),
    ("explanation", r"\b(explain|describe|discuss|draw|how)\b"),
]


def question_type(q: str) -> str:
    for name, pattern in QUESTION_TYPES:
        if re.search(pattern, q, re.I):
            return name
    return "other"


def study_sections(pages: list[dict]) -> list[dict]:
    """Walk pages in order, tracking current unit/topic headings."""
    sections, current = [], None
    unit_no, unit_title = "", ""
    for pg in pages:
        for line in pg["text"].splitlines():
            if m := UNIT_RE.match(line):
                unit_no, unit_title = m.group(1), m.group(2).strip()
                continue
            if m := TOPIC_RE.match(line):
                current = {"text": "", "metadata": {
                    "subject": pg["subject"], "unit": f"Unit {unit_no}",
                    "unit_title": unit_title, "topic": m.group(1).strip(),
                    "source_type": "study_material", "question_type": "",
                    "marks": 0, "year": 0,
                    "source_file": pg["source_file"], "page": pg["page"]}}
                sections.append(current)
                continue
            if current is not None:
                current["text"] += line + " "
            # text before the first Topic heading (title page) is skipped
    for s in sections:
        s["text"] = s["text"].strip()
    return [s for s in sections if s["text"]]


def pyq_questions(pages: list[dict]) -> list[dict]:
    records = []
    by_file: dict[str, list[dict]] = {}
    for pg in pages:
        by_file.setdefault(pg["source_file"], []).append(pg)
    for fname, fpages in by_file.items():
        full = "\n".join(p["text"] for p in fpages)
        year_m = YEAR_RE.search(full)
        year = int(year_m.group(1)) if year_m else 0
        for m in QUESTION_RE.finditer(full):
            raw = " ".join(m.group(2).split())
            marks = MARKS_RE.search(raw)
            unit = QUNIT_RE.search(raw)
            text = QUNIT_RE.sub("", MARKS_RE.sub("", raw)).strip()
            page = next((p["page"] for p in fpages if f"Q{m.group(1)}." in p["text"]), 1)
            records.append({"text": text, "metadata": {
                "subject": fpages[0]["subject"],
                "unit": f"Unit {unit.group(1)}" if unit else "",
                "unit_title": "", "topic": "",
                "source_type": "pyq", "question_type": question_type(text),
                "marks": int(marks.group(1)) if marks else 0, "year": year,
                "source_file": fname, "page": page}})
    return records
