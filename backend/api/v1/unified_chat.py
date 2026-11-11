# backend/api/v1/unified_chat.py
# Unified assistant: one chat box that routes each message (see core/orchestrator.py).
#
# SSE events, in order:
#   token             a rule reply (greeting, thanks, ...) sent as one token, then done
#   routing_decision  the route: agent_type, agent_display, confidence, reason, execution_mode
#   progress / token / meta   for qa: the QA agent's stream (same as /qa/chat/stream)
#   guidance          for exam, resume, interview and clarify: a message and a link to the page
#   pipeline_plan     for multi_agent: the steps (resume review, then mock interview)
#   done              end of stream
#   error             the QA stream failed

import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.api.v1.qa import qa_state, stream_qa_events
from backend.core.logger import get_logger
from backend.core.rate_limit import limit_chat
from backend.core.memory import build_thread_id
from backend.core.orchestrator import AgentType, RouteDecision, get_agent_graph, route_message
from backend.dependencies import get_current_user

router = APIRouter()
logger = get_logger(__name__)


AGENT_DISPLAY = {
    AgentType.QA:        "Knowledge Q&A",
    AgentType.EXAM:      "Exam Grading",
    AgentType.RESUME:    "Resume Review",
    AgentType.INTERVIEW: "Mock Interview",
}

GUIDANCE = {
    "exam": {
        "message": "It looks like you want an exam graded. On the Exam page, upload your answer sheet as a Word "
                   "(.docx) file; the AI grades it and shows the topics to review.",
        "action_label": "Go to Exam Grading", "action_url": "/exam",
    },
    "resume": {
        "message": "It looks like you want a resume review. On the Resume page, upload your resume as a PDF; the AI "
                   "scores it on six dimensions and suggests changes.",
        "action_label": "Go to Resume Review", "action_url": "/resume",
    },
    "interview": {
        "message": "It looks like you want to practise an interview. On the Interview page, choose a target position "
                   "(and optionally a reviewed resume); the AI asks technical and project questions and writes a report.",
        "action_label": "Go to Mock Interview", "action_url": "/interview",
    },
    "clarify": {
        "message": "I'm not sure which feature fits yet. Could you say a bit more? For example: ask a technical "
                   "question, submit an exam for grading, review your resume, or practise an interview.",
        "action_label": "", "action_url": "",
    },
}

PIPELINE_PLAN = {
    "title": "Job preparation",
    "intro": "Here is a two-step plan. Your resume review feeds the mock interview: start the interview from the "
             "reviewed resume and the interviewer asks about your projects.",
    "steps": [
        {"step": 1, "agent_type": "resume", "label": "Resume review",
         "desc": "Upload your resume as a PDF; get six dimension scores and suggestions tied to your text.",
         "action_label": "Start resume review", "action_url": "/resume",
         "tip": "Do this first: the interview uses the review."},
        {"step": 2, "agent_type": "interview", "label": "Mock interview",
         "desc": "Technical questions plus a deep dive into your projects, with a five-dimension report.",
         "action_label": "Start mock interview", "action_url": "/interview",
         "tip": "Choose the reviewed resume when you start."},
    ],
}

_FEATURES = (
    "- **Technical Q&A**: ask a question and I'll answer from the knowledge base\n"
    "- **Exam grading**: say \"I want to submit my exam\"; the AI grades it and finds topics to review\n"
    "- **Resume review**: say \"review my resume\"; six dimension scores and suggestions\n"
    "- **Mock interview**: say \"I want a mock interview\"; technical and project questions with a report\n"
    "- **Job preparation**: say \"help me prepare for job hunting\"; a resume review, then a mock interview "
    "based on it\n"
)

RULE_REPLIES = {
    "hello": "Hi! I'm the EduAgent assistant.\n\nI can help with:\n" + _FEATURES + "\nWhat would you like to do?",
    "thanks": "You're welcome! Ask me anything else whenever you like.",
    "bye": "Goodbye, and good luck with your studies!",
    "identity": "I'm the **EduAgent assistant**, a learning assistant for software and AI engineering students. "
                "Several specialised agents work behind me:\n\n" + _FEATURES + "\nHow can I help?",
    "capability": "Here is what I can do:\n\n" + _FEATURES
                  + "\nJust tell me what you need and I'll send it to the right agent.",
}


class UnifiedChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[\w-]+$")
    message:    str = Field(..., min_length=1, max_length=2000)


def _sse(data: dict) -> dict:
    return {"data": json.dumps(data)}


def _routing_event(decision: RouteDecision) -> dict:
    agent = decision.agent_type
    return _sse({
        "type":           "routing_decision",
        "label":          decision.label,
        "agent_type":     agent.value if agent else decision.label,
        "agent_display":  AGENT_DISPLAY.get(agent, "Job preparation" if decision.label == "multi_agent" else "Clarify"),
        "confidence":     round(decision.confidence, 2),
        "reason":         decision.reason,
        "execution_mode": decision.execution_mode.value if decision.execution_mode else "",
    })


@router.post("/stream", dependencies=[Depends(limit_chat)])
async def unified_chat_stream(req: UnifiedChatRequest, current_user: dict = Depends(get_current_user)):
    """Route the message, then answer it (qa) or point to the right page."""

    async def events():
        decision = await route_message(req.message)

        if decision.source == "rule":            # Fixed reply, no LLM call
            yield _sse({"type": "token", "content": RULE_REPLIES[decision.label]})
            yield _sse({"type": "done"})
            return

        yield _routing_event(decision)
        if decision.label == "qa":
            config = {"configurable": {"thread_id": build_thread_id(current_user["user_id"], req.session_id)}}
            async for event in stream_qa_events(get_agent_graph(AgentType.QA),
                                                qa_state(req.message, current_user, req.session_id),
                                                config, req.session_id):
                yield event
        elif decision.label == "multi_agent":
            yield _sse({"type": "pipeline_plan", **PIPELINE_PLAN})
        else:
            yield _sse({"type": "guidance", **GUIDANCE[decision.label]})
        yield _sse({"type": "done"})

    return EventSourceResponse(events())
