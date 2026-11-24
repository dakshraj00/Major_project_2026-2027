from fastapi import APIRouter
from pydantic import BaseModel

from app.graph.workflow import run

router = APIRouter()


class QARequest(BaseModel):
    question: str
    subject: str = ""                 # "" = search all subjects
    retrieval_mode: str = "hybrid"    # hybrid | vector | bm25


@router.post("/qa")
def qa(req: QARequest):
    return run(mode="qa", question=req.question, subject=req.subject,
               retrieval_mode=req.retrieval_mode)
