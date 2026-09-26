# 📚 VidyaRAG

**Your syllabus, your PYQs, one intelligent study companion.**

VidyaRAG is a Retrieval-Augmented Generation (RAG) system built for exam preparation. It combines previous-year question papers (PYQs) and study materials into a single interactive chatbot that can **answer student questions with grounded, cited responses** — and **generate brand-new practice questions** modeled on real exam patterns.

Generic AI tools hallucinate facts and can't cite exact syllabus sources. Basic search misses technical terms and formulas, and truncates long definitions. VidyaRAG fixes this with hybrid search, adaptive chunking, and rigorous testing on real past exam papers.

---

## ✨ Features

- 🤖 **Q&A Mode** — Ask a doubt, get an answer grounded in your actual syllabus and study material, with source traceability.
- 📝 **Question Generation Mode** — Generate new, exam-style practice questions based on PYQ patterns for any topic.
- 🔍 **Hybrid Retrieval** — Combines semantic (vector) search with keyword (BM25) search, so exact formulas and technical terms are never missed.
- ✂️ **Adaptive Chunking** — Preserves complete definitions instead of cutting them off mid-thought.
- ✅ **Answerability Validation** — Every generated question is re-verified against retrieval to confirm the source material can actually answer it.
- 📊 **RAGAS-Benchmarked** — Evaluated on real PYQ test sets for faithfulness, relevancy, and retrieval precision — not synthetic data.

---

## 🏗️ Architecture

```
Data Sources (PYQs + Notes)
        │
        ▼
   Chunking (Adaptive Split)
        │
        ▼
Hybrid Retrieval (Vector + BM25)
        │
        ▼
  Generation Layer (Q&A + New Questions)
```

Each subject's PYQs and materials are tagged with metadata (subject, unit, topic, question type, marks) to support accurate retrieval across multiple subjects in one system.

---

## 🛠️ Tech Stack

| Category | Tool |
|---|---|
| Language | Python |
| Orchestration | LangGraph |
| Vector Database | ChromaDB |
| Keyword Retrieval | rank-bm25 |
| Backend | FastAPI |
| Frontend | React / Streamlit |
| Evaluation | RAGAS |

---

## 📅 Project Status

VidyaRAG is being developed as a one-year major project (Group RG73) under the supervision of **Mr. Shankar Kumar Jha**.

- ✅ Dataset (PYQs + study materials) collected and organized
- ✅ System architecture and design finalized
- 🚧 Implementation in progress

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/dakshraj00/vidyarag.git
cd vidyarag

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

> ⚠️ Setup instructions will be expanded as implementation progresses.

---

## 👥 Team

| Member | Role |
|---|---|
| Daksh Raj Chauhan | RAG architecture, pipeline design, hybrid retrieval |
| — | Dataset collection & preprocessing |
| — | Frontend/chat interface & evaluation |

**Supervisor:** Mr. Shankar Kumar Jha

---

## 📖 References

This project builds on ideas from:

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020
- Levonian et al., *RAG to Improve Math Question-Answering*, GAIED 2023
- Maity et al., *ICL + RAG for Automatic Question Generation in Educational Domains*, FIRE 2025
- Cheng et al., *A Survey on Knowledge-Oriented Retrieval-Augmented Generation*, ACM Computing Surveys 2025

---

## 📄 License

MIT License — feel free to fork and adapt for your own institution's exam prep needs.
