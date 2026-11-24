"""One LangGraph graph: a router node sends the request to the Q&A chain or
the question generator (which runs the validator internally).

    START -> router --(mode=qa)-------> qa ---------> END
                    \--(mode=generate)-> generate ---> END
"""
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.generation.qa_chain import answer
from app.generation.question_gen import generate


class State(TypedDict, total=False):
    mode: Literal["qa", "generate"]
    question: str
    subject: str
    topic: str
    n: int
    retrieval_mode: str
    result: dict[str, Any]


GEN_WORDS = ("generate", "practice question", "make questions", "give me questions", "mock test")


def router(state: State) -> State:
    """Use the explicit mode if given; otherwise guess from the wording."""
    if state.get("mode") not in ("qa", "generate"):
        q = state.get("question", "").lower()
        state["mode"] = "generate" if any(w in q for w in GEN_WORDS) else "qa"
    return state


def qa_node(state: State) -> State:
    return {"result": answer(state["question"], state.get("subject", ""),
                             state.get("retrieval_mode", "hybrid"))}


def generate_node(state: State) -> State:
    return {"result": generate(state["subject"], state.get("topic") or state.get("question", ""),
                               state.get("n", 5))}


def build_graph():
    g = StateGraph(State)
    g.add_node("router", router)
    g.add_node("qa", qa_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: s["mode"], {"qa": "qa", "generate": "generate"})
    g.add_edge("qa", END)
    g.add_edge("generate", END)
    return g.compile()


graph = build_graph()


def run(**inputs) -> dict:
    return graph.invoke(inputs)["result"]
