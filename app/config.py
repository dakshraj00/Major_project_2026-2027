"""Central settings. All secrets come from .env, never from code."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Paths
DATA_DIR = ROOT / "data"
PYQ_DIR = DATA_DIR / "raw" / "pyqs"
STUDY_DIR = DATA_DIR / "raw" / "study_material"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
CHROMA_DIR = PROCESSED_DIR / "chroma"
BM25_PATH = PROCESSED_DIR / "bm25.pkl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
PROMPTS_DIR = ROOT / "app" / "prompts"
RESULTS_DIR = ROOT / "evaluation" / "results"

# Models
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Chunking / retrieval
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "hierarchical")
CHUNK_SIZE = 800          # characters, for fixed chunking
CHUNK_OVERLAP = 120
TOP_K_EACH = 10           # candidates from each retriever before fusion
TOP_K_FINAL = 5           # chunks passed to the LLM after reranking
RRF_K = 60                # reciprocal rank fusion constant

# Validator: a generated question is "answerable" if the best reranker score
# of the study material for it is above this threshold.
ANSWERABLE_THRESHOLD = float(os.getenv("ANSWERABLE_THRESHOLD", "0.0"))
# Q&A: if even the best chunk scores below this, answer "not found" without calling the LLM.
QA_MIN_SCORE = float(os.getenv("QA_MIN_SCORE", "-5.0"))

API_URL = os.getenv("API_URL", "http://localhost:8000")

for d in (PYQ_DIR, STUDY_DIR, PROCESSED_DIR, EVAL_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
