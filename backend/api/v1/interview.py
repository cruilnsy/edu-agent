# backend/api/v1/interview.py
# Mock interview API. Every query is scoped to the caller's tenant and student: a session,
# report or resume of another user returns 404, the same as one that does not exist.

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import text

from backend.agents.interview.graph import build_interview_graph
from backend.agents.interview.nodes import START_MESSAGE, stages_reached
from backend.agents.interview.state import InterviewStage
from backend.core.logger import get_logger
from backend.core.rate_limit import limit_chat
from backend.core.memory import build_config, build_thread_id
from backend.dependencies import AsyncSessionLocal, get_current_user

router = APIRouter()
logger = get_logger(__name__)
# Shared by all requests. Starts with the in-memory checkpointer; the app's startup swaps in
# the Postgres one (use_checkpointer) so interviews in progress survive a restart.
_graph = build_interview_graph()


def use_checkpointer(checkpointer) -> None:
    global _graph
    _graph = build_interview_graph(checkpointer=checkpointer)

MAX_MESSAGE_CHARS = 4000


class StartSessionRequest(BaseModel):
    target_position:  str = Field(min_length=1, max_length=100)
    resume_review_id: str | None = None    # Optional: links the interview to a resume review

    @field_validator("target_position")
    @classmethod
    def _one_line(cls, value: str) -> str:
        value = " ".join(value.split())    # It goes into prompts: no newlines
        if not value:
            raise ValueError("target_position is empty")
        return value


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


def _parse_uuid(value: str, what: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid {what}")


async def _load_session(session_id: str, user: dict) -> dict:
    """The caller's own session (404 otherwise)."""
    session_id = _parse_uuid(session_id, "session id")
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            text("""SELECT session_id, target_position, status, overall_score, report, finished_at
                    FROM interview_sessions
                    WHERE session_id = :sid AND tenant_id = :tenant_id AND student_id = :student_id"""),
            {"sid": session_id, "tenant_id": user["tenant_id"], "student_id": user["user_id"]},
        )).mappings().fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Interview session not found")
    return dict(row)


async def _session_for_chat(session_id: str, user: dict) -> str:
    session = await _load_session(session_id, user)
    if session["status"] != "in_progress":
        # A finished interview cannot continue, so its report is never regenerated
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This interview has already finished")
    return session["session_id"]


async def _check_resume_review(resume_review_id: str, user: dict) -> str:
    review_id = _parse_uuid(resume_review_id, "resume review id")
    async with AsyncSessionLocal() as db:
        found = (await db.execute(
            text("SELECT 1 FROM resume_reviews WHERE id = :id AND tenant_id = :tenant_id AND student_id = :student_id"),
            {"id": review_id, "tenant_id": user["tenant_id"], "student_id": user["user_id"]},
        )).fetchone()
    if not found:
        raise HTTPException(status_code=404, detail="Resume review not found")
    return review_id


def _last_ai_message(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
            return str(content)
    return ""


def _turn_payload(state: dict) -> dict:
    """What the client gets after each turn; the report summary once the interview ends."""
    stage = state.get("current_stage", InterviewStage.WARMUP.value)
    payload = {
        "reply":         _last_ai_message(state.get("messages", [])),
        "current_stage": stage,
        "total_turns":   state.get("total_turn_count", 0),
        "is_finished":   stage == InterviewStage.FINISHED.value,
        "stages_reached": stages_reached(state),
    }
    if payload["is_finished"]:
        report = state.get("report") or {}
        payload["report_summary"] = {
            "overall_score": report.get("overall_score"),
            "strengths":     report.get("strengths", []),
            "improvements":  report.get("improvements", []),
        }
    return payload


def _turn_input(message: str, user: dict, session_id: str) -> dict:
    """Only the new message: the checkpointer restores the rest of the state."""
    return {"messages": [HumanMessage(content=message)], "student_id": user["user_id"],
            "session_id": session_id, "tenant_id": user["tenant_id"]}


@router.post("/sessions", status_code=201, dependencies=[Depends(limit_chat)])
async def start_session(req: StartSessionRequest, current_user: dict = Depends(get_current_user)):
    """Create an interview session and return the interviewer's opening message."""
    student_id, tenant_id = current_user["user_id"], current_user["tenant_id"]
    review_id = await _check_resume_review(req.resume_review_id, current_user) if req.resume_review_id else None
    session_id = str(uuid.uuid4())

    async with AsyncSessionLocal() as db:
        async with db.begin():
            await db.execute(
                text("""INSERT INTO interview_sessions
                            (id, tenant_id, student_id, session_id, thread_id, target_position, resume_review_id, status)
                        VALUES (:id, :tenant_id, :student_id, :session_id, :thread_id, :target_position,
                                :resume_review_id, 'in_progress')"""),
                {"id": str(uuid.uuid4()), "tenant_id": tenant_id, "student_id": student_id, "session_id": session_id,
                 "thread_id": build_thread_id(student_id, session_id), "target_position": req.target_position,
                 "resume_review_id": review_id},
            )

    # First run: load_context initializes the rest of the state
    first_input = {"messages": [HumanMessage(content=START_MESSAGE)], "student_id": student_id,
                   "tenant_id": tenant_id, "session_id": session_id, "target_position": req.target_position,
                   "resume_review_id": review_id}
    try:
        result = await _graph.ainvoke(first_input, config=build_config(student_id, session_id))
    except Exception as e:
        logger.error("interview.start_failed", session_id=session_id, error=str(e), exc_info=True)
        async with AsyncSessionLocal() as db:
            async with db.begin():
                await db.execute(text("DELETE FROM interview_sessions WHERE session_id = :sid"), {"sid": session_id})
        raise HTTPException(status_code=503, detail="The interview could not be started; please try again")

    logger.info("interview.session_started", session_id=session_id)
    return {"session_id": session_id, "target_position": req.target_position, "status": "in_progress",
            "message": _last_ai_message(result.get("messages", []))}


@router.post("/sessions/{session_id}/chat", dependencies=[Depends(limit_chat)])
async def chat(session_id: str, req: ChatRequest, current_user: dict = Depends(get_current_user)):
    """The candidate's message in; the interviewer's reply out (and the report summary at the end)."""
    session_id = await _session_for_chat(session_id, current_user)
    try:
        result = await _graph.ainvoke(_turn_input(req.message, current_user, session_id),
                                      config=build_config(current_user["user_id"], session_id))
    except Exception as e:
        logger.error("interview.chat_failed", session_id=session_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="The interviewer could not reply; please send your message again")
    return {"session_id": session_id, **_turn_payload(result)}


@router.post("/sessions/{session_id}/chat/stream", dependencies=[Depends(limit_chat)])
async def chat_stream(session_id: str, req: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Server-sent events: type=token for each piece of the interviewer's reply, then type=done
    with the full turn payload, or type=error."""
    session_id = await _session_for_chat(session_id, current_user)
    config = build_config(current_user["user_id"], session_id)
    graph_input = _turn_input(req.message, current_user, session_id)

    async def events():
        try:
            async for event in _graph.astream_events(graph_input, config=config, version="v2"):
                # Only the interviewer's reply is streamed, not the grading or report calls
                if (event["event"] == "on_chat_model_stream"
                        and event.get("metadata", {}).get("langgraph_node") == "generate_response"):
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        yield {"data": json.dumps({"type": "token", "content": chunk.content})}
            final = await _graph.aget_state(config)
            yield {"data": json.dumps({"type": "done", **_turn_payload(final.values if final else {})})}
        except Exception as e:
            logger.error("interview.chat_stream_failed", session_id=session_id, error=str(e), exc_info=True)
            yield {"data": json.dumps({"type": "error", "message": "Streaming failed; please try again"})}

    return EventSourceResponse(events())


@router.get("/sessions/{session_id}/report")
async def get_report(session_id: str, current_user: dict = Depends(get_current_user)):
    session = await _load_session(session_id, current_user)
    if session["status"] != "finished":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The interview has not finished yet")
    report = session["report"]
    report = report if isinstance(report, dict) else (json.loads(report) if report else {})
    return {
        "session_id":         session["session_id"],
        "target_position":    session["target_position"],
        "overall_score":      session["overall_score"],
        "dimensions":         report.get("dimensions", []),
        "strengths":          report.get("strengths", []),
        "improvements":       report.get("improvements", []),
        "overall_comment":    report.get("overall_comment", ""),
        "recommended_topics": report.get("recommended_topics", []),
        "next_step_advice":   report.get("next_step_advice", ""),
        "fallback":           bool(report.get("fallback")),
        "stages_reached":     report.get("stages_reached", []),
    }


@router.get("/sessions")
async def list_sessions(current_user: dict = Depends(get_current_user)):
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            text("""SELECT session_id, target_position, overall_score, status, finished_at, created_at
                    FROM interview_sessions
                    WHERE tenant_id = :tenant_id AND student_id = :student_id
                    ORDER BY created_at DESC LIMIT 50"""),
            {"tenant_id": current_user["tenant_id"], "student_id": current_user["user_id"]},
        )).mappings().all()
    items = [{"session_id": r["session_id"], "target_position": r["target_position"],
              "overall_score": r["overall_score"], "status": r["status"],
              "finished_at": r["finished_at"].isoformat() if r["finished_at"] else None,
              "created_at": r["created_at"].isoformat() if r["created_at"] else None} for r in rows]
    return {"items": items, "total": len(items)}
