"""Fixed-size chunking: ignore structure, cut each document into N-character
windows with overlap. Metadata comes from the section where the window starts."""
from app.chunking.common import group_by_file, make_chunk
from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk(sections: list[dict], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    chunks = []
    for fname, secs in group_by_file(sections).items():
        # Join the file into one string, remembering where each section starts.
        text, starts = "", []
        for s in secs:
            starts.append((len(text), s["metadata"]))
            text += s["text"] + " "
        pos, n = 0, 0
        while pos < len(text):
            end = min(pos + size, len(text))
            if end < len(text):                      # don't cut mid-word
                space = text.rfind(" ", pos, end)
                end = space if space > pos else end
            md = [m for start, m in starts if start <= pos][-1]
            chunks.append(make_chunk("fixed", n, text[pos:end].strip(), md))
            n += 1
            if end >= len(text):
                break
            pos = max(end - overlap, pos + 1)
            while pos < len(text) and text[pos - 1] != " ":   # start on a word
                pos += 1
    return chunks
