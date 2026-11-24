from app.chunking.common import pyq_chunks

STRATEGIES = ["fixed", "semantic", "hierarchical"]


def chunk_all(strategy: str, study: list[dict], pyqs: list[dict]) -> list[dict]:
    if strategy == "fixed":
        from app.chunking.fixed import chunk
    elif strategy == "semantic":
        from app.chunking.semantic import chunk
    elif strategy == "hierarchical":
        from app.chunking.hierarchical import chunk
    else:
        raise ValueError(f"unknown chunking strategy: {strategy}")
    return chunk(study) + pyq_chunks(strategy, pyqs)
