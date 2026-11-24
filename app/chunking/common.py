"""Helpers shared by the chunkers."""
import re

SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def group_by_file(sections: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for s in sections:
        out.setdefault(s["metadata"]["source_file"], []).append(s)
    return out


def make_chunk(strategy: str, n: int, text: str, metadata: dict, **extra) -> dict:
    md = dict(metadata)
    md.update(extra)
    md["chunk_strategy"] = strategy
    return {"id": f"{strategy}-{md['source_file']}-{n}", "text": text, "metadata": md}


def pyq_chunks(strategy: str, pyqs: list[dict]) -> list[dict]:
    """Each PYQ question is already a natural chunk, whatever the strategy."""
    return [make_chunk(strategy, i, q["text"], q["metadata"])
            for i, q in enumerate(pyqs)]
