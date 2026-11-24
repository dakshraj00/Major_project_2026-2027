"""Semantic chunking: embed consecutive sentences and start a new chunk where
the topic shifts (cosine similarity between neighbours drops sharply)."""
import numpy as np

from app.chunking.common import group_by_file, make_chunk, split_sentences
from app.retrieval.vector_store import get_embedder

MAX_CHARS = 1000


def chunk(sections: list[dict], breakpoint_percentile: float = 25) -> list[dict]:
    model = get_embedder()
    chunks = []
    for fname, secs in group_by_file(sections).items():
        sents, metas = [], []
        for s in secs:
            for sent in split_sentences(s["text"]):
                sents.append(sent)
                metas.append(s["metadata"])
        if not sents:
            continue
        emb = model.encode(sents, normalize_embeddings=True)
        sims = (emb[:-1] * emb[1:]).sum(axis=1)       # similarity to next sentence
        cut = np.percentile(sims, breakpoint_percentile) if len(sims) else 0
        n, current, current_md = 0, [sents[0]], metas[0]
        for i in range(1, len(sents)):
            too_long = sum(len(x) for x in current) + len(sents[i]) > MAX_CHARS
            if sims[i - 1] < cut or too_long:
                chunks.append(make_chunk("semantic", n, " ".join(current), current_md))
                n, current, current_md = n + 1, [], metas[i]
            current.append(sents[i])
        chunks.append(make_chunk("semantic", n, " ".join(current), current_md))
    return chunks
