"""ChromaDB dense index. One collection per chunking strategy."""
from functools import lru_cache

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DIR, EMBEDDING_MODEL

# bge models work better when queries (not passages) get this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: " if "bge" in EMBEDDING_MODEL else ""


@lru_cache
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@lru_cache
def _client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def _clean_md(md: dict) -> dict:
    # Chroma accepts only str/int/float/bool; parent_text is kept in chunks.json
    return {k: v for k, v in md.items() if isinstance(v, (str, int, float, bool)) and k != "parent_text"}


def build(strategy: str, chunks: list[dict]) -> None:
    client = _client()
    name = f"chunks_{strategy}"
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    emb = get_embedder().encode([c["text"] for c in chunks], normalize_embeddings=True,
                                show_progress_bar=False)
    col.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
            embeddings=emb.tolist(), metadatas=[_clean_md(c["metadata"]) for c in chunks])


def make_where(filters: dict) -> dict | None:
    conds = [{k: v} for k, v in filters.items() if v]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def search(strategy: str, query: str, k: int, filters: dict) -> list[tuple[str, float]]:
    """Returns [(chunk_id, similarity)] best first."""
    col = _client().get_collection(f"chunks_{strategy}")
    q = get_embedder().encode([QUERY_PREFIX + query], normalize_embeddings=True)
    res = col.query(query_embeddings=q.tolist(), n_results=k, where=make_where(filters))
    return [(i, 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]
