"""BM25 keyword index. The tokenizer keeps technical tokens such as
'csma/cd', '3nf', 'x->y', 'e^(-g)' pieces and "banker's" intact."""
import pickle
import re

from rank_bm25 import BM25Okapi

from app.config import PROCESSED_DIR

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[/'\-][a-z0-9]+)*")
STOPWORDS = set("a an the is are was were of in on for to and or with by as at be it this that "
                "what which how why when explain define give between from its their".split())


def tokenize(text: str) -> list[str]:
    toks = TOKEN_RE.findall(text.lower())
    out = []
    for t in toks:
        if t in STOPWORDS:
            continue
        out.append(t)
        if t.endswith("'s"):            # "belady's" also matches "belady"
            out.append(t[:-2])
    return out


def _path(strategy: str):
    return PROCESSED_DIR / f"bm25_{strategy}.pkl"


def build(strategy: str, chunks: list[dict]) -> None:
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    with open(_path(strategy), "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [c["id"] for c in chunks],
                     "metadata": [c["metadata"] for c in chunks]}, f)


_cache: dict[str, dict] = {}


def _load(strategy: str) -> dict:
    if strategy not in _cache:
        with open(_path(strategy), "rb") as f:
            _cache[strategy] = pickle.load(f)
    return _cache[strategy]


def search(strategy: str, query: str, k: int, filters: dict) -> list[tuple[str, float]]:
    idx = _load(strategy)
    scores = idx["bm25"].get_scores(tokenize(query))
    hits = []
    for i in scores.argsort()[::-1]:
        if scores[i] <= 0:
            break
        md = idx["metadata"][i]
        if all(md.get(key) == val for key, val in filters.items() if val):
            hits.append((idx["ids"][i], float(scores[i])))
            if len(hits) == k:
                break
    return hits
