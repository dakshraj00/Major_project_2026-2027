"""Hierarchical chunking: the parent is a whole (unit, topic) section, the
children are small sentence groups. We search over children (precise matches)
but give the LLM the parent text (full context)."""
from app.chunking.common import make_chunk, split_sentences

CHILD_CHARS = 300


def chunk(sections: list[dict]) -> list[dict]:
    chunks, n = [], 0
    for parent_id, s in enumerate(sections):
        group = []
        for sent in split_sentences(s["text"]):
            group.append(sent)
            if sum(len(x) for x in group) >= CHILD_CHARS:
                chunks.append(make_chunk("hierarchical", n, " ".join(group), s["metadata"],
                                         parent_id=parent_id, parent_text=s["text"]))
                n, group = n + 1, []
        if group:
            chunks.append(make_chunk("hierarchical", n, " ".join(group), s["metadata"],
                                     parent_id=parent_id, parent_text=s["text"]))
            n += 1
    return chunks
