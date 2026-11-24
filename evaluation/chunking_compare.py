"""Compare chunking strategies on the eval set without any LLM calls.

Metrics per strategy (hybrid retrieval, all subjects searched):
  hit@1 / hit@3 - expected topic appears in the top 1 / top 3 chunks
  MRR           - mean reciprocal rank of the first correct chunk
  avg_ctx_chars - how much text the LLM would receive (cost proxy)

Run: python evaluation/chunking_compare.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.chunking import STRATEGIES
from app.config import EVAL_DIR, RESULTS_DIR
from app.retrieval.hybrid import context_text, retrieve


def main():
    tests = json.loads((EVAL_DIR / "testset.json").read_text())
    rows = []
    for strategy in STRATEGIES:
        for mode in ("vector", "hybrid"):
            h1 = h3 = rr = chars = 0
            for t in tests:
                hits = retrieve(t["question"], mode=mode, strategy=strategy, top_k=3,
                                use_reranker=(mode == "hybrid"))
                topics = [h["metadata"]["topic"] for h in hits]
                # fixed chunks can span topics, so also accept the reference text appearing in the chunk
                ok = [tp == t["topic"] or t["topic"].lower() in context_text(h).lower()
                      for tp, h in zip(topics, hits)]
                h1 += ok[0] if ok else 0
                h3 += any(ok)
                rr += next((1 / (i + 1) for i, o in enumerate(ok) if o), 0)
                chars += sum(len(context_text(h)) for h in hits)
            n = len(tests)
            rows.append({"strategy": strategy, "retrieval": mode, "hit@1": round(h1 / n, 3),
                         "hit@3": round(h3 / n, 3), "MRR": round(rr / n, 3),
                         "avg_ctx_chars": round(chars / n)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = RESULTS_DIR / "chunking_comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
