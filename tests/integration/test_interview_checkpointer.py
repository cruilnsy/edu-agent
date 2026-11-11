# tests/integration/test_interview_checkpointer.py
# Interview state in the Postgres checkpointer survives a restart (a new connection pool).
# The LLM and the summary/question-bank lookups are replaced, so no API calls are made.

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.agents.interview import nodes
from backend.agents.interview.graph import build_interview_graph
from backend.core.checkpointer import postgres_checkpointer

pytestmark = pytest.mark.integration


class ReplyLLM:
    async def ainvoke(self, messages):
        return AIMessage(content="Explain database indexes.")


async def _none(*_):
    return None


async def test_interview_state_survives_a_new_connection_pool(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: ReplyLLM())
    monkeypatch.setattr(nodes, "fetch_summary", _none)
    config = {"configurable": {"thread_id": f"test_{uuid.uuid4().hex}"}}
    turn = {"student_id": "s", "session_id": "x", "tenant_id": "t", "target_position": "Backend Engineer",
            "messages": [AIMessage(content="Tell me about yourself."), HumanMessage(content="I build APIs.")],
            "current_stage": "warmup", "stage_turn_count": 1, "total_turn_count": 1, "max_turns": 30,
            "question_bank": [{"id": "q0", "content": "Explain database indexes.", "tags": [], "asked": False}],
            "current_question": None, "projects_asked": [], "resume_projects": [],
            "last_answer_quality": None, "followup_count": 0, "answer_log": []}

    async with postgres_checkpointer(max_size=2) as saver:
        await build_interview_graph(checkpointer=saver).ainvoke(turn, config=config)

    try:
        async with postgres_checkpointer(max_size=2) as saver:          # As after a restart
            state = (await build_interview_graph(checkpointer=saver).aget_state(config)).values
            assert state["current_stage"] == "tech_base"
            assert state["messages"][-1].content == "Explain database indexes."
            assert state["question_bank"][0]["asked"] is True
    finally:                                 # adelete_thread is not implemented in this version
        async with postgres_checkpointer(max_size=2) as saver, saver.conn.connection() as conn:
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (config["configurable"]["thread_id"],))
