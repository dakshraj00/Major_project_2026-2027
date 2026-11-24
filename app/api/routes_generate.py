from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.graph.workflow import run

router = APIRouter()


class GenerateRequest(BaseModel):
    subject: str
    topic: str
    n: int = Field(5, ge=1, le=10)


@router.post("/generate")
def generate(req: GenerateRequest):
    return run(mode="generate", subject=req.subject, topic=req.topic, n=req.n)
