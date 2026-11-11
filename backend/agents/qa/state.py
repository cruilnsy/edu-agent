# backend/agents/qa/state.py

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class QAState(TypedDict):
    """State shared by every node of the QA agent."""

    # ── Message history: the add_messages reducer appends instead of overwriting ──
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Request context (set by the API or Orchestrator; read-only for nodes) ──
    student_id:  str            # Student id
    tenant_id:   str            # Tenant id (Milvus and DB isolation)
    session_id:  str            # Session id, part of the thread_id
    course_id:   Optional[str]  # Limits retrieval to one course (knowledge base topic); None = all

    # ── Query processing ──
    original_query:    str            # The user's question, unchanged throughout
    query_type:        str            # GENERAL / PRECISE / VAGUE / BROAD
    rewritten_queries: list[str]      # BROAD: sub-queries from Multi-Query
    hyde_document:     Optional[str]  # VAGUE: hypothetical document from HyDE

    # ── Retrieval and reranking (retrieve() encodes internally; no vectors in State) ──
    ranked_chunks:      list[dict]   # Top-K chunks after BGE reranking
    confidence:         float        # Rerank score of the top chunk, 0-1
    is_high_confidence: bool         # confidence >= HIGH_CONFIDENCE_THRESHOLD
    web_search_results: list[dict]   # Web Search MCP results (low confidence or GENERAL + web)

    # ── Answer and control flags ──
    answer:            str              # Final answer text
    sources:           list[str]        # Sources shown with the answer
    answer_mode:       str              # "rag" / "web_augmented" / "llm_direct" / "general"
    existing_summary:  Optional[str]    # Session summary loaded from the DB
    should_summarize:  bool             # Compress the history into the summary this turn
    enable_web_search: bool             # On low confidence, search the web before answering
    fallback_used:     bool             # A step failed and a fallback was used
    structured_output: Optional[dict]   # Structured result for the Orchestrator
