# backend/agents/qa/graph.py

from langgraph.graph import StateGraph, START, END

from backend.agents.qa.state import QAState
from backend.agents.qa.nodes import (
    classify_query_node,
    hyde_generate_node,
    multi_query_rewrite_node,
    retrieve_node,
    generate_rag_node,
    web_search_node,
    generate_direct_node,
    generate_general_node,
    enqueue_pending_node,
    save_memory_node,
)
from backend.core.memory import get_memory_saver


def _route_by_query_type(state: QAState) -> str:
    """Route after classify_query.

    GENERAL + enable_web_search=True  -> "GENERAL_WEB": search the web, then answer
    GENERAL + enable_web_search=False -> "GENERAL": answer with the LLM
    PRECISE / VAGUE / BROAD           -> the matching retrieval branch
    """
    qt = state.get("query_type", "PRECISE").upper()
    if qt == "GENERAL" and state.get("enable_web_search", False):
        return "GENERAL_WEB"
    return qt


def _route_by_confidence(state: QAState) -> str:
    """Route after retrieve.

    is_high_confidence=True              -> "high": answer from the knowledge base
    is_high_confidence=False
      + enable_web_search=True           -> "low_web": search the web, then answer directly
      + enable_web_search=False          -> "low_direct": answer with the LLM alone
    """
    if state.get("is_high_confidence", False):
        return "high"
    if state.get("enable_web_search", False):
        return "low_web"
    return "low_direct"


def _route_after_web_search(state: QAState) -> str:
    """web_search is shared by two paths; send each back to its own answer node:
      - from GENERAL_WEB (query_type=GENERAL) -> generate_general
      - from low confidence                   -> generate_direct
    """
    if state.get("query_type", "").upper() == "GENERAL":
        return "generate_general"
    return "generate_direct"


def build_qa_graph():
    """Build and compile the QA graph, used by the API and the Orchestrator."""
    builder = StateGraph(QAState)

    # ── Nodes ──
    builder.add_node("classify_query",       classify_query_node)
    builder.add_node("hyde_generate",        hyde_generate_node)
    builder.add_node("multi_query_rewrite",  multi_query_rewrite_node)
    builder.add_node("retrieve",             retrieve_node)
    builder.add_node("generate_rag",         generate_rag_node)
    builder.add_node("web_search",           web_search_node)
    builder.add_node("generate_direct",      generate_direct_node)
    builder.add_node("generate_general",     generate_general_node)
    builder.add_node("enqueue_pending",      enqueue_pending_node)
    builder.add_node("save_memory",          save_memory_node)

    # ── Entry ──
    builder.add_edge(START, "classify_query")

    # ── Conditional edge 1: route by query type ──
    builder.add_conditional_edges(
        "classify_query",
        _route_by_query_type,
        {
            "GENERAL":     "generate_general",
            "GENERAL_WEB": "web_search",
            "PRECISE":     "retrieve",
            "VAGUE":       "hyde_generate",
            "BROAD":       "multi_query_rewrite",
        },
    )

    # VAGUE / BROAD rejoin retrieve after preprocessing
    builder.add_edge("hyde_generate",       "retrieve")
    builder.add_edge("multi_query_rewrite", "retrieve")

    # ── Conditional edge 2: route by confidence ──
    builder.add_conditional_edges(
        "retrieve",
        _route_by_confidence,
        {
            "high":       "generate_rag",
            "low_web":    "web_search",
            "low_direct": "generate_direct",
        },
    )

    # ── Conditional edge 3: leave web_search ──
    builder.add_conditional_edges(
        "web_search",
        _route_after_web_search,
        {
            "generate_general": "generate_general",
            "generate_direct":  "generate_direct",
        },
    )

    # ── Answer nodes -> wrap-up nodes ──
    builder.add_edge("generate_rag",     "save_memory")
    builder.add_edge("generate_general", "save_memory")
    builder.add_edge("generate_direct",  "enqueue_pending")
    builder.add_edge("enqueue_pending",  "save_memory")
    builder.add_edge("save_memory",      END)

    # ── Compile with a checkpointer for multi-turn memory ──
    checkpointer = get_memory_saver("qa")
    return builder.compile(checkpointer=checkpointer)
