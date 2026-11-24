"""FastAPI app. Run from vidyarag/:  uvicorn app.main:app --reload
(also works from vidyarag/app/:     uvicorn main:app --reload)"""
import json
import sys
from pathlib import Path

# Make the `app` package importable even when started from inside app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import groq
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import routes_generate, routes_qa
from app.config import GROQ_API_KEY, PROCESSED_DIR
from app.generation.llm import MissingAPIKey

app = FastAPI(title="VidyaRAG", description="Exam-prep RAG chatbot for JUIT students")
app.include_router(routes_qa.router)
app.include_router(routes_generate.router)


@app.exception_handler(MissingAPIKey)
def missing_key(_: Request, exc: MissingAPIKey):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(groq.APIError)
def groq_error(_: Request, exc: groq.APIError):
    return JSONResponse(status_code=503, content={"detail": f"LLM error (Groq): {exc}"})


@app.exception_handler(FileNotFoundError)
def missing_index(_: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "groq_key_set": bool(GROQ_API_KEY),
            "index_built": (PROCESSED_DIR / "pyqs.json").exists()}


@app.get("/subjects")
def subjects():
    """Subjects and their topics, for the UI selectors."""
    path = PROCESSED_DIR / "chunks_hierarchical.json"
    if not path.exists():
        return {}
    out: dict[str, set] = {}
    for c in json.loads(path.read_text()):
        md = c["metadata"]
        if md["source_type"] == "study_material":
            out.setdefault(md["subject"], set()).add(md["topic"])
    return {s: sorted(t) for s, t in sorted(out.items())}
