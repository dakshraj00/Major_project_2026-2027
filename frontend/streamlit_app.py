"""VidyaRAG chat UI. Run (with the API running): streamlit run frontend/streamlit_app.py"""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="VidyaRAG", page_icon="📘", layout="centered")
st.title("📘 VidyaRAG")
st.caption("Exam prep from your syllabus and PYQs. Answers are grounded in the material and cited.")


@st.cache_data(ttl=60)
def get_subjects() -> dict:
    try:
        return requests.get(f"{API}/subjects", timeout=10).json()
    except requests.RequestException:
        return {}


subjects = get_subjects()
if not subjects:
    st.error(f"Cannot reach the API at {API}, or the index is empty. Start it with: uvicorn app.main:app")
    st.stop()

with st.sidebar:
    mode = st.radio("Mode", ["Ask a doubt", "Generate questions"])
    subject = st.selectbox("Subject", (["All subjects"] if mode == "Ask a doubt" else []) + list(subjects))
    if mode == "Generate questions":
        topic = st.selectbox("Topic", subjects[subject])
        n = st.slider("How many questions", 1, 10, 5)
    else:
        retrieval_mode = st.selectbox("Retrieval", ["hybrid", "vector", "bm25"],
                                      help="hybrid = dense + BM25 + reranker")
    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}] {s['subject']} · {s['unit']} · {s['topic']}**  \n"
                        f"`{s['source_file']}` page {s['page']} · relevance {s['score']}")
            st.caption(s["text"])


def show_generated(r: dict):
    st.markdown(f"**{len(r['accepted'])} practice questions on _{r['topic']}_ ({r['subject']})**")
    if r.get("note"):
        st.warning(f"Generator declined: {r['note']}")
    for i, q in enumerate(r["accepted"], 1):
        st.markdown(f"{i}. {q['question']}  \n"
                    f"<small>{q.get('marks', '?')} marks · {q.get('question_type', '')} · ✅ {q['validation']['reason']}</small>",
                    unsafe_allow_html=True)
    if r["rejected"]:
        with st.expander(f"❌ Rejected by validator ({len(r['rejected'])})", expanded=True):
            for q in r["rejected"]:
                st.markdown(f"- ~~{q['question']}~~  \n  <small>{q['validation']['reason']}</small>",
                            unsafe_allow_html=True)
    else:
        st.caption("Validator rejected none: every question is answerable from the material.")
    if r["style_examples"]:
        with st.expander("PYQs used as style examples"):
            for e in r["style_examples"]:
                st.markdown(f"- {e['text']} _({e['year']}, {e['marks']} marks)_")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("generated"):
            show_generated(m["generated"])
        else:
            st.markdown(m["content"])
            show_sources(m.get("sources", []))

placeholder = "Ask a doubt, e.g. What is Belady's anomaly?" if mode == "Ask a doubt" \
    else "Press enter to generate (or type a custom topic)"
prompt = st.chat_input(placeholder)
if prompt is not None:
    if mode == "Generate questions":
        user_text = f"Generate {n} questions on {prompt.strip() or topic}"
    else:
        user_text = prompt
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if mode == "Ask a doubt":
                    r = requests.post(f"{API}/qa", timeout=120, json={
                        "question": prompt,
                        "subject": "" if subject == "All subjects" else subject,
                        "retrieval_mode": retrieval_mode}).json()
                    msg = {"role": "assistant", "content": r.get("answer", r.get("detail", "Error")),
                           "sources": r.get("sources", [])}
                    st.markdown(msg["content"])
                    show_sources(msg["sources"])
                else:
                    r = requests.post(f"{API}/generate", timeout=180, json={
                        "subject": subject, "topic": prompt.strip() or topic, "n": n}).json()
                    if "accepted" in r:
                        msg = {"role": "assistant", "content": "", "generated": r}
                        show_generated(r)
                    else:
                        msg = {"role": "assistant", "content": r.get("detail", "Error")}
                        st.error(msg["content"])
            except requests.RequestException as e:
                msg = {"role": "assistant", "content": f"API error: {e}"}
                st.error(msg["content"])
    st.session_state.messages.append(msg)
