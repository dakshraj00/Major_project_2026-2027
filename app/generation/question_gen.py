"""Practice-question generator: topic context + similar PYQs as style examples
-> LLM -> answerability validator."""
import json

import groq

from app.generation.llm import chat, load_prompt
from app.generation.validator import validate
from app.retrieval.hybrid import citation, context_text, retrieve


def generate(subject: str, topic: str, n: int = 5) -> dict:
    context_chunks = retrieve(topic, subject=subject, source_type="study_material", top_k=4)
    pyqs = retrieve(topic, subject=subject, source_type="pyq", top_k=5)
    seen, ctx = set(), []
    for c in context_chunks:
        t = context_text(c)
        if t not in seen:
            seen.add(t)
            ctx.append(t)
    pyq_examples = "\n".join(f"- {p['text']} [{p['metadata']['marks']} marks, {p['metadata']['year']}]"
                             for p in pyqs) or "- (none available)"
    prompt = load_prompt("question_gen_template.txt").format(
        subject=subject, topic=topic, n=n, context="\n\n".join(ctx), pyq_examples=pyq_examples)
    note = ""
    try:
        raw = chat(prompt, temperature=0.7, json_mode=True)
        questions = json.loads(raw).get("questions", [])
    except groq.BadRequestError as e:
        # Groq raises this when the model answers in prose instead of JSON,
        # which happens when it refuses because the topic is not in the material.
        body = e.body.get("error", {}) if isinstance(e.body, dict) else {}
        if body.get("code") != "json_validate_failed":
            raise
        questions, note = [], body.get("failed_generation", "The model did not return questions.")
    except json.JSONDecodeError:
        questions, note = [], "The model returned invalid JSON."
    accepted, rejected = [], []
    for q in questions[:n]:
        q["validation"] = validate(q.get("question", ""), subject)
        (accepted if q["validation"]["answerable"] else rejected).append(q)
    return {"subject": subject, "topic": topic, "accepted": accepted, "rejected": rejected, "note": note,
            "style_examples": [citation(p) for p in pyqs]}
