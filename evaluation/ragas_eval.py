"""RAGAS evaluation: hybrid (dense + BM25 + reranker) vs vector-only retrieval.

For each test question we run the real Q&A chain, then RAGAS scores:
  faithfulness       - is the answer supported by the retrieved context?
  answer_relevancy   - does the answer address the question?
  context_precision  - are the retrieved chunks relevant (ranked well)?
  context_recall     - does the context contain what the reference answer needs?

The judge LLM is the same Groq model (via langchain-groq); embeddings are the
local sentence-transformers model. Free-tier rate limits make this slow, so
max_workers is kept low.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                           LLMContextRecall, ResponseRelevancy)

from app.config import EMBEDDING_MODEL, EVAL_DIR, GROQ_API_KEY, GROQ_MODEL, RESULTS_DIR
from app.generation.qa_chain import answer

METRIC_COLS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]


def collect(tests: list[dict], mode: str) -> list[dict]:
    samples = []
    for t in tests:
        r = answer(t["question"], subject=t["subject"], mode=mode)
        samples.append({"user_input": t["question"], "response": r["answer"],
                        "retrieved_contexts": r["contexts"] or [""], "reference": t["reference"]})
        time.sleep(1)  # be gentle with the free tier
    return samples


def main(limit: int | None = None):
    tests = json.loads((EVAL_DIR / "testset.json").read_text())[:limit]
    llm = LangchainLLMWrapper(ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(),
               LLMContextRecall()]
    run_config = RunConfig(max_workers=2, timeout=240, max_retries=10, max_wait=60)

    summary, per_question = [], []
    for mode in ("vector", "hybrid"):
        print(f"running Q&A with retrieval={mode} on {len(tests)} questions...")
        samples = collect(tests, mode)
        result = evaluate(EvaluationDataset.from_list(samples), metrics=metrics, llm=llm,
                          embeddings=emb, run_config=run_config, show_progress=False)
        df = result.to_pandas()
        df.insert(0, "retrieval", mode)
        per_question.append(df)
        row = {"retrieval": mode}
        row.update({c: round(df[c].mean(skipna=True), 3) for c in METRIC_COLS if c in df})
        summary.append(row)
        print("  ", row)

    table = pd.DataFrame(summary).rename(columns={"llm_context_precision_with_reference": "context_precision"})
    pd.concat(per_question).to_csv(RESULTS_DIR / "ragas_per_question.csv", index=False)
    table.to_csv(RESULTS_DIR / "ragas_summary.csv", index=False)
    md = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    (RESULTS_DIR / "ragas_summary.md").write_text("\n".join(md) + "\n")
    print("\n" + table.to_string(index=False))
    print(f"\nsaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
