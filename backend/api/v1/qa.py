# backend/api/v1/qa.py
# QA endpoints: chat (JSON), chat/stream (SSE) and session history.

import json
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage

from backend.agents.qa.graph import build_qa_graph
from backend.core.memory import build_thread_id
from backend.dependencies import get_current_user
from backend.core.logger import get_logger
from backend.core.rate_limit import limit_chat

router = APIRouter()
logger = get_logger(__name__)


# ── Request / response models ──

class ChatRequest(BaseModel):
    session_id:        str        = Field(..., min_length=1, max_length=64, pattern=r"^[\w-]+$", description="Session id")
    course_id:         str | None = Field(None, description="Course (knowledge base topic) id; limits retrieval when set")
    message:           str        = Field(..., min_length=1, max_length=2000)
    enable_web_search: bool       = Field(False, description="On low confidence, search the web before answering")


class ChatResponse(BaseModel):
    session_id:    str
    answer:        str
    answer_mode:   str        # "rag" / "web_augmented" / "llm_direct" / "general"
    confidence:    float
    sources:       list[str]
    fallback_used: bool


class SessionMessage(BaseModel):
    role:       str   # "user" / "assistant"
    content:    str
    created_at: str


class HistoryResponse(BaseModel):
    session_id:  str
    messages:    list[SessionMessage]
    summary:     str | None
    total_turns: int


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(limit_chat)])
async def chat(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Ask a question; returns a knowledge base or direct LLM answer (not streamed)."""
    graph     = build_qa_graph()
    thread_id = build_thread_id(current_user["user_id"], req.session_id)

    initial_state = qa_state(req.message, current_user, req.session_id, req.course_id, req.enable_web_search)
    config: dict = {"configurable": {"thread_id": thread_id}}

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.error("chat.invoke_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "AGENT_ERROR", "message": "The assistant failed to answer. Please try again."},
        )

    return ChatResponse(
        session_id=req.session_id,
        answer=result.get("answer", ""),
        answer_mode=result.get("answer_mode", "llm_direct"),
        confidence=result.get("confidence", 0.0),
        sources=result.get("sources", []),
        fallback_used=result.get("fallback_used", False),
    )


# Stream tokens from these nodes only
GENERATE_NODES = {"generate_rag", "generate_direct", "generate_general"}

# Progress text sent when a node starts
PROGRESS_LABELS = {
    "classify_query":      "Understanding the question...",
    "hyde_generate":       "Understanding the question...",
    "multi_query_rewrite": "Rewriting the query...",
    "retrieve":            "Searching the knowledge base...",
    "web_search":          "Searching the web...",
    "generate_general":    "Thinking...",
}


def qa_state(message: str, user: dict, session_id: str, course_id: str | None = None,
             enable_web_search: bool = False) -> dict:
    return {
        "messages":            [HumanMessage(content=message)],
        "student_id":          user["user_id"],
        "tenant_id":           user["tenant_id"],
        "session_id":          session_id,
        "course_id":           course_id,
        "query_type":          "PRECISE",     # Placeholder; classify_query_node sets it
        "enable_web_search":   enable_web_search,
        "web_search_results":  [],            # Reset each turn so last turn's results do not leak into sources
        "fallback_used":       False,         # Reset each turn; a node sets it when a step falls back
    }


async def stream_qa_events(graph, initial_state: dict, config: dict, session_id: str):
    """Run the QA graph and yield its SSE events: progress, token, then meta (or error).
    Shared by /qa/chat/stream and the unified chat."""
    answer_mode, confidence, sources = "llm_direct", 0.0, []
    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            evt  = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")
            if evt == "on_chain_start" and node in PROGRESS_LABELS:
                yield {"data": json.dumps({"type": "progress", "stage": PROGRESS_LABELS[node]})}
            elif evt == "on_chat_model_stream" and node in GENERATE_NODES:
                chunk = event["data"].get("chunk")
                if chunk and chunk.content:
                    yield {"data": json.dumps({"type": "token", "content": chunk.content})}
            elif evt == "on_chain_end" and node in GENERATE_NODES:
                output = event["data"].get("output", {})
                if isinstance(output, dict):
                    answer_mode = output.get("answer_mode") or answer_mode
                    if output.get("sources") is not None:
                        sources = output["sources"]
                    conf = (output.get("structured_output") or {}).get("confidence")
                    if conf is not None:
                        confidence = conf
    except Exception as e:
        logger.error("chat_stream.error", error=str(e), exc_info=True)
        yield {"data": json.dumps({"type": "error", "message": "Streaming failed. Please try again."})}
        return
    yield {"data": json.dumps({"type": "meta", "session_id": session_id, "answer_mode": answer_mode,
                               "confidence": confidence, "sources": sources})}


@router.post("/chat/stream", dependencies=[Depends(limit_chat)])
async def chat_stream(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Same request as /chat; the answer streams as server-sent events."""
    graph  = build_qa_graph()
    config = {"configurable": {"thread_id": build_thread_id(current_user["user_id"], req.session_id)}}
    state  = qa_state(req.message, current_user, req.session_id, req.course_id, req.enable_web_search)

    async def event_generator():
        async for event in stream_qa_events(graph, state, config, req.session_id):
            yield event
        yield {"data": json.dumps({"type": "done"})}

    return EventSourceResponse(event_generator())


@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
async def get_session_history(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Messages and summary of one session."""
    from sqlalchemy import text as sa_text
    from langchain_core.messages import HumanMessage as LCHuman, AIMessage as LCAi
    from backend.dependencies import AsyncSessionLocal

    student_id = current_user["user_id"]
    thread_id  = build_thread_id(student_id, session_id)

    # Summary from the DB
    summary = None
    try:
        async with AsyncSessionLocal() as db_session:
            result = await db_session.execute(
                sa_text(
                    "SELECT summary FROM qa_sessions "
                    "WHERE thread_id = :tid AND student_id = :sid"
                ),
                {"tid": thread_id, "sid": student_id},
            )
            row = result.fetchone()
            if row:
                summary = row[0]
    except Exception as e:
        logger.warning("get_history.db_error", error=str(e))

    # Messages from the checkpointer
    messages: list[SessionMessage] = []
    total_turns = 0
    try:
        graph  = build_qa_graph()
        config = {"configurable": {"thread_id": thread_id}}
        state  = await graph.aget_state(config)
        if state and state.values:
            for msg in state.values.get("messages", []):
                content = (
                    msg.text
                    if hasattr(msg, "text") and not callable(msg.text)
                    else str(msg.content)
                )
                if isinstance(msg, LCHuman):
                    messages.append(SessionMessage(role="user", content=content, created_at=""))
                elif isinstance(msg, LCAi):
                    messages.append(SessionMessage(role="assistant", content=content, created_at=""))
            total_turns = sum(1 for m in messages if m.role == "user")
    except Exception as e:
        logger.warning("get_history.checkpoint_error", error=str(e))

    return HistoryResponse(
        session_id=session_id,
        messages=messages,
        summary=summary,
        total_turns=total_turns,
    )
