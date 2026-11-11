# backend/agents/interview/nodes.py
# Nodes of the mock interview graph. One request = one candidate message = one graph run:
#   load_context -> evaluate_answer -> check_stage -> generate_response -> save_memory
#                                                 \-> generate_report -> save_report -> save_memory  (finished)
# Database access is in the small module-level functions below (fetch_*, store_*), so the eval
# and the tests can replace them.

import asyncio
import difflib
import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import text

from backend.agents.interview.prompts import (
    CLOSING_PROMPT, EVALUATE_ANSWER_PROMPT, GENERATE_REPORT_PROMPT, INTRO_EVAL_TECH_FIRST_PROMPT,
    PROJECT_PROMPT, STAGE_TRANSITION_TEXT, SYSTEM_PROMPT, TECH_BASE_GENERATE_PROMPT, TECH_BASE_PROMPT,
    WARMUP_PROMPT,
)
from backend.agents.interview.state import (
    REPORT_DIMENSIONS, AnswerEvaluation, AnswerQuality, InterviewReport, InterviewStage, InterviewState,
)
from backend.core.llm_factory import get_llm, get_structured_llm
from backend.core.logger import get_logger
from backend.core.memory import (
    build_thread_id, compress_to_summary, should_trigger_summary, trim_messages_to_window,
)
from backend.dependencies import AsyncSessionLocal

logger = get_logger(__name__)

START_MESSAGE = "[start interview]"       # Sent by the API to run the first turn
DEFAULT_MAX_TURNS = 30                    # Hard limit on candidate messages per interview
MAX_FOLLOWUP_PER_QUESTION = 2             # Follow-ups on one question or project
TECH_QUESTIONS = 6                        # Bank questions to ask before moving to projects
LLM_QUESTION_COUNT = 8                    # Questions the LLM writes at the start
MAX_ANSWER_CHARS = 1500                   # Candidate text sent to the LLM per answer
MAX_TRANSCRIPT_CHARS = 20000              # Transcript sent to the report; the most recent part is kept
STRUCTURED_ATTEMPTS = 2                   # One retry for empty or invalid structured output
DUPLICATE_SIMILARITY = 0.85               # Questions this similar count as the same question

# Candidate messages counted per stage before it may / must end
STAGE_MIN_TURNS = {
    InterviewStage.WARMUP.value:    1,    # The self-introduction
    InterviewStage.TECH_BASE.value: TECH_QUESTIONS,
    InterviewStage.PROJECT.value:   2,
    InterviewStage.CLOSING.value:   2,
}
STAGE_MAX_TURNS = {
    InterviewStage.WARMUP.value:    1,
    InterviewStage.TECH_BASE.value: 10,
    InterviewStage.PROJECT.value:   8,
    InterviewStage.CLOSING.value:   2,
}
STAGE_ORDER = [s.value for s in InterviewStage]

# The candidate asks to stop. Matching is on the whole message, so "end" inside an answer
# about something else does not count.
END_PATTERNS = [
    re.compile(r"\b(end|stop|finish|quit|wrap up)\b[^.?!]{0,20}\binterview\b", re.I),
    re.compile(r"\bcan we (stop|end|finish)\b", re.I),
    re.compile(r"\bi (want|need|would like|'d like) to (stop|leave|quit)\b", re.I),
]
NO_ANSWER_TEXTS = {
    "", "i don't know", "i dont know", "don't know", "no idea", "not sure", "i'm not sure", "im not sure",
    "pass", "skip", "sorry, i don't know", "sorry, i'm not sure", "i have no idea",
}

# Question tracks of the bank (interview_questions.target_position), picked from the position text
POSITION_TRACKS = [
    ("ai",       re.compile(r"\b(ai|llm|ml|machine learning|nlp|data scien\w*|algorithm\w*|genai)\b", re.I)),
    ("frontend", re.compile(r"\b(front[- ]?end|react|ui|web developer|typescript|javascript)\b", re.I)),
    ("backend",  re.compile(r"\b(back[- ]?end|python|server|api|full[- ]?stack|platform|devops)\b", re.I)),
]


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _msg_text(msg: BaseMessage) -> str:
    """Message text, for both str and multi-part content."""
    content = msg.content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _wrap(answer: str, limit: int = MAX_ANSWER_CHARS) -> str:
    """Candidate text for a prompt, cut to limit. The closing tag is removed so the text
    cannot end the block early."""
    return answer[:limit].replace("</candidate_answer>", "").strip() or "(empty)"


def _last_candidate_message(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _msg_text(msg)
    return ""


def _last_interviewer_message(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return _msg_text(msg)
    return ""


def wants_to_end(message: str) -> bool:
    return any(p.search(message) for p in END_PATTERNS)


def is_no_answer(message: str) -> bool:
    return message.replace("\u2019", "'").strip().strip(".!").lower() in NO_ANSWER_TEXTS


def position_track(target_position: str) -> str:
    """Bank track for a free-text position: ai, frontend, backend, or general."""
    for track, pattern in POSITION_TRACKS:
        if pattern.search(target_position):
            return track
    return "general"


def _normalize_question(content: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", content.lower()).strip()


def is_duplicate_question(content: str, others: list[str]) -> bool:
    norm = _normalize_question(content)
    return any(difflib.SequenceMatcher(None, norm, _normalize_question(o)).ratio() >= DUPLICATE_SIMILARITY
               for o in others)


def merge_question_bank(*sources: list[dict]) -> list[dict]:
    """Questions from all sources in order, without near-duplicates."""
    bank: list[dict] = []
    for source in sources:
        for q in source:
            if q.get("content") and not is_duplicate_question(q["content"], [b["content"] for b in bank]):
                bank.append(q)
    return bank


def parse_generated_questions(raw: str, target_position: str) -> list[dict]:
    """The LLM's JSON array of questions, also when it is wrapped in a ``` fence."""
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.S)
    data = json.loads(fenced.group(1) if fenced else raw.strip())
    questions = []
    for i, q in enumerate(data if isinstance(data, list) else []):
        if not isinstance(q, dict) or not str(q.get("content", "")).strip():
            continue
        difficulty = q.get("difficulty") if q.get("difficulty") in ("easy", "medium", "hard") else "medium"
        tags = [str(t) for t in q.get("tags", [])] if isinstance(q.get("tags"), list) else []
        questions.append({"id": f"llm_{i}", "content": str(q["content"]).strip(), "difficulty": difficulty,
                          "tags": tags or [target_position], "asked": False})
    return questions


async def _structured_call(schema, system: str, prompt: str):
    """Structured LLM call with one retry when the reply is empty or fails validation."""
    last_error: Exception | None = None
    for attempt in range(1, STRUCTURED_ATTEMPTS + 1):
        try:
            result = await get_structured_llm("interview", schema).ainvoke([
                SystemMessage(content=system), HumanMessage(content=prompt)])
            if result is not None:
                return result
            last_error = ValueError("the model returned no structured output")
        except Exception as e:
            last_error = e
        logger.warning("interview.structured_retry", schema=schema.__name__, attempt=attempt, error=str(last_error))
    raise last_error


def _system_prompt(state: InterviewState) -> str:
    system = SYSTEM_PROMPT.format(position=state.get("target_position") or "Software Engineer")
    if state.get("existing_summary"):
        system += f"\n\nSummary of the earlier interview:\n{state['existing_summary']}"
    return system


# ──────────────────────────────────────────────────────────────
# Database access (replaced by the eval and the tests)
# ──────────────────────────────────────────────────────────────

async def fetch_resume(resume_review_id: str, student_id: str, tenant_id: str) -> dict:
    """Projects and skills of the student's own resume review (empty if not theirs)."""
    async with AsyncSessionLocal() as s:
        row = (await s.execute(
            text("""SELECT structured_data FROM resume_reviews
                    WHERE id = :id AND student_id = :student_id AND tenant_id = :tenant_id"""),
            {"id": resume_review_id, "student_id": student_id, "tenant_id": tenant_id},
        )).mappings().fetchone()
    if not row or not row["structured_data"]:
        return {}
    data = row["structured_data"] if isinstance(row["structured_data"], dict) else json.loads(row["structured_data"])
    # Only projects and skills: the interview never needs the resume's contact details
    return {"resume_projects": data.get("projects", []), "resume_skills": data.get("skills_list", [])}


async def fetch_question_bank(track: str, tenant_id: str, limit: int = 20) -> list[dict]:
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(
            text("""SELECT id, content, difficulty, tags FROM interview_questions
                    WHERE tenant_id = :tenant_id AND is_active = TRUE
                      AND target_position IN (:track, 'general')
                    ORDER BY (target_position = 'general'),
                             CASE difficulty WHEN 'medium' THEN 1 WHEN 'easy' THEN 2 ELSE 3 END,
                             RANDOM()
                    LIMIT :limit"""),
            {"tenant_id": tenant_id, "track": track, "limit": limit},
        )).mappings().all()
    return [{"id": str(r["id"]), "content": r["content"], "difficulty": r["difficulty"],
             "tags": r["tags"] if isinstance(r["tags"], list) else [], "asked": False} for r in rows]


async def fetch_summary(thread_id: str) -> str | None:
    async with AsyncSessionLocal() as s:
        row = (await s.execute(text("SELECT summary FROM interview_sessions WHERE thread_id = :tid"),
                               {"tid": thread_id})).mappings().fetchone()
    return row["summary"] if row else None


async def store_report(thread_id: str, report: dict) -> bool:
    """Save the report and finish the session. Only an unfinished session is updated, so a
    finished interview's report is never overwritten. Returns False if nothing was updated."""
    async with AsyncSessionLocal() as s:
        async with s.begin():
            result = await s.execute(
                text("""UPDATE interview_sessions
                        SET report = :report, overall_score = :overall_score, status = 'finished',
                            finished_at = NOW(), updated_at = NOW()
                        WHERE thread_id = :thread_id AND status = 'in_progress'"""),
                {"report": json.dumps(report), "overall_score": report.get("overall_score"), "thread_id": thread_id},
            )
    return result.rowcount == 1


async def store_summary(thread_id: str, summary: str | None) -> None:
    async with AsyncSessionLocal() as s:
        async with s.begin():
            await s.execute(text("UPDATE interview_sessions SET summary = :summary, updated_at = NOW() "
                                 "WHERE thread_id = :thread_id"),
                            {"summary": summary, "thread_id": thread_id})


# ──────────────────────────────────────────────────────────────
# load_context
# ──────────────────────────────────────────────────────────────

async def generate_questions(target_position: str, count: int = LLM_QUESTION_COUNT) -> list[dict]:
    """Questions written by the LLM for the position; [] on failure (the DB bank is used then)."""
    try:
        response = await get_llm("interview", temperature=0.3).ainvoke([HumanMessage(
            content=TECH_BASE_GENERATE_PROMPT.format(position=target_position, count=count))])
        questions = parse_generated_questions(_msg_text(response), target_position)
        logger.info("generate_questions.done", count=len(questions))
        return questions
    except Exception as e:
        logger.warning("generate_questions.failed", error=str(e))
        return []


async def _safe(coro, default, event: str):
    """Await coro; on error log it and return default, so one failed lookup does not stop the interview."""
    try:
        return await coro
    except Exception as e:
        logger.warning(event, error=str(e))
        return default


async def load_context_node(state: InterviewState) -> dict:
    """First request: initialize the state, load the resume and build the question bank.
    Later requests: refresh the history summary."""
    student_id, session_id = state.get("student_id") or "", state.get("session_id") or ""
    if not student_id or not session_id:
        logger.error("load_context.missing_ids")
        return {}
    thread_id = build_thread_id(student_id, session_id)

    if state.get("current_stage"):          # Not the first request
        return {"existing_summary": await _safe(fetch_summary(thread_id), state.get("existing_summary"),
                                                "load_context.summary_failed")}

    target = state.get("target_position") or "Software Engineer"
    review_id = state.get("resume_review_id")
    resume_coro = (fetch_resume(review_id, student_id, state.get("tenant_id", "")) if review_id
                   else asyncio.sleep(0, result={}))
    # Resume, DB bank, LLM questions and summary in parallel; the LLM call is the slowest
    resume, db_questions, llm_questions, summary = await asyncio.gather(
        _safe(resume_coro, {}, "load_context.resume_failed"),
        _safe(fetch_question_bank(position_track(target), state.get("tenant_id", "")), [],
              "load_context.question_bank_failed"),
        generate_questions(target),
        _safe(fetch_summary(thread_id), None, "load_context.summary_failed"),
    )
    # LLM questions first (closest to the position), then the DB bank, without duplicates
    bank = merge_question_bank(llm_questions, db_questions)
    logger.info("load_context.initialized", questions=len(bank), llm_questions=len(llm_questions),
                projects=len(resume.get("resume_projects", [])))
    return {
        "current_stage":       InterviewStage.WARMUP.value,
        "stage_turn_count":    0,
        "total_turn_count":    0,
        "max_turns":           DEFAULT_MAX_TURNS,
        "question_bank":       bank,
        "current_question":    None,
        "projects_asked":      [],
        "last_answer_quality": AnswerQuality.ADEQUATE.value,
        "followup_count":      0,
        "answer_log":          [],
        "answered_stages":     [],
        "resume_projects":     resume.get("resume_projects", []),
        "resume_skills":       resume.get("resume_skills", []),
        "existing_summary":    summary,
        "report":              None,
        "fallback_used":       not llm_questions,      # The LLM bank failed; the DB bank is the fallback
        "structured_output":   None,
    }


# ──────────────────────────────────────────────────────────────
# evaluate_answer
# ──────────────────────────────────────────────────────────────

async def evaluate_answer_node(state: InterviewState) -> dict:
    """Count the candidate's message and grade it if it answers a question.

    Runs before check_stage, so the answer is graded against the question it answers, even
    when the stage then moves on. Only technical and project answers are graded."""
    messages = state.get("messages", [])
    stage = state.get("current_stage", InterviewStage.WARMUP.value)
    if not any(isinstance(m, AIMessage) for m in messages):
        return {}                            # The API's start message, not a candidate turn

    counts = {"total_turn_count": state.get("total_turn_count", 0) + 1,
              "stage_turn_count": state.get("stage_turn_count", 0) + 1}
    answer = _last_candidate_message(messages)
    question = state.get("current_question")
    answered = state.get("answered_stages", [])
    if not wants_to_end(answer) and stage not in answered:
        counts["answered_stages"] = answered + [stage]
    if (stage not in (InterviewStage.TECH_BASE.value, InterviewStage.PROJECT.value)
            or not question or wants_to_end(answer)):
        return counts

    # Grade against what the interviewer actually asked last (a follow-up differs from the
    # bank question); the bank question or project is context
    asked = _last_interviewer_message(messages)[:MAX_ANSWER_CHARS] or question.get("content", "")
    entry = {"stage": stage, "question": asked, "topic": question.get("content", ""), "quality": None,
             "interviewer_instructions_found": False, "fallback": False}
    updates: dict = {}
    if is_no_answer(answer):                 # Saves an LLM call
        entry["quality"] = AnswerQuality.NO_ANSWER.value
    else:
        try:
            result: AnswerEvaluation = await _structured_call(
                AnswerEvaluation, _system_prompt(state),
                EVALUATE_ANSWER_PROMPT.format(topic=question.get("content", ""), question=asked, answer=_wrap(answer)))
            entry["quality"] = result.quality
            entry["interviewer_instructions_found"] = result.interviewer_instructions_found
        except Exception as e:               # Neutral label: the interview goes on
            logger.warning("evaluate_answer.failed", error=str(e))
            entry["quality"] = AnswerQuality.ADEQUATE.value
            entry["fallback"] = True
            updates["fallback_used"] = True

    logger.info("evaluate_answer.done", stage=stage, quality=entry["quality"])
    return {**counts, **updates, "last_answer_quality": entry["quality"],
            "answer_log": state.get("answer_log", []) + [entry]}


# ──────────────────────────────────────────────────────────────
# check_stage
# ──────────────────────────────────────────────────────────────

def _tech_followup_due(state: InterviewState) -> bool:
    """Technical follow-ups only after an excellent answer: the bar is high."""
    return (state.get("current_question") is not None
            and state.get("followup_count", 0) < MAX_FOLLOWUP_PER_QUESTION
            and state.get("last_answer_quality") == AnswerQuality.EXCELLENT.value)


def _project_followup_due(state: InterviewState) -> bool:
    """Project follow-ups also after an adequate answer: digging is the point of this stage."""
    return (state.get("current_question") is not None
            and state.get("followup_count", 0) < MAX_FOLLOWUP_PER_QUESTION
            and state.get("last_answer_quality") in (AnswerQuality.EXCELLENT.value, AnswerQuality.ADEQUATE.value))


def _unasked(state: InterviewState) -> list[dict]:
    return [q for q in state.get("question_bank", []) if not q.get("asked")]


def _stage_done(stage: str, state: InterviewState) -> bool:
    """Whether the stage's content is covered (called once its minimum turns are reached)."""
    if stage == InterviewStage.WARMUP.value:
        return True
    if stage == InterviewStage.TECH_BASE.value:
        asked = sum(1 for q in state.get("question_bank", []) if q.get("asked"))
        return asked >= min(TECH_QUESTIONS, len(state.get("question_bank", []))) and not _tech_followup_due(state)
    if stage == InterviewStage.PROJECT.value:
        projects = state.get("resume_projects", [])
        if not projects:
            return True
        all_started = all(p.get("name") in state.get("projects_asked", []) for p in projects)
        return all_started and not _project_followup_due(state)
    return True                              # CLOSING ends after its minimum turns


def check_stage_node(state: InterviewState) -> dict:
    """Move to the next stage, or end the interview. Pure logic, no LLM."""
    stage = state.get("current_stage", InterviewStage.WARMUP.value)
    if stage == InterviewStage.FINISHED.value:
        return {}

    if (state.get("total_turn_count", 0) >= state.get("max_turns", DEFAULT_MAX_TURNS)
            or (state.get("total_turn_count", 0) > 0 and wants_to_end(_last_candidate_message(state.get("messages", []))))):
        logger.info("check_stage.ended_early", stage=stage)
        return {"current_stage": InterviewStage.FINISHED.value}

    turns = state.get("stage_turn_count", 0)
    bank_exhausted = stage == InterviewStage.TECH_BASE.value and not _unasked(state) and not _tech_followup_due(state)
    advance = (turns >= STAGE_MAX_TURNS[stage]
               or (turns >= STAGE_MIN_TURNS[stage] and _stage_done(stage, state))
               or (turns > 0 and bank_exhausted))
    if not advance:
        return {}

    next_stage = STAGE_ORDER[STAGE_ORDER.index(stage) + 1]
    if next_stage == InterviewStage.TECH_BASE.value and not _unasked(state):
        next_stage = InterviewStage.PROJECT.value     # No questions at all: skip the technical stage
    logger.info("check_stage.advance", from_stage=stage, to_stage=next_stage)
    return {"current_stage": next_stage, "stage_turn_count": 0, "followup_count": 0, "current_question": None}


def route_after_check_stage(state: InterviewState) -> str | list[str]:
    """The report on the last turn; otherwise the reply, plus the history summary when one is
    due. The summary runs next to the reply rather than after it, so it adds no wait."""
    if state.get("current_stage") == InterviewStage.FINISHED.value:
        return "generate_report"
    if should_trigger_summary(state.get("messages", [])):
        return ["generate_response", "save_memory"]
    return "generate_response"


# ──────────────────────────────────────────────────────────────
# generate_response
# Each _plan_* function decides what to say and returns a Reply: the LLM prompt (or None for
# a fixed text), the fallback text used if the LLM fails, the state updates, and the settings.
# ──────────────────────────────────────────────────────────────

class Reply:
    def __init__(self, prompt: str | None, fallback: str, updates: dict | None = None,
                 temperature: float = 0.5, window: int = 6):
        self.prompt, self.fallback, self.updates = prompt, fallback, updates or {}
        self.temperature, self.window = temperature, window


def _mark_asked(bank: list[dict], question: dict) -> list[dict]:
    return [{**q, "asked": True} if q["id"] == question["id"] else q for q in bank]


def _plan_warmup(state: InterviewState) -> Reply:
    position = state.get("target_position") or "Software Engineer"
    return Reply(WARMUP_PROMPT.format(position=position),
                 f"Welcome to this mock interview for the {position} position. "
                 "Please start with a short self-introduction.", temperature=0.7, window=2)


def _plan_tech(state: InterviewState) -> Reply:
    messages = state.get("messages", [])
    current = state.get("current_question")
    answer = _last_candidate_message(messages)

    if current is not None and _tech_followup_due(state):
        return Reply(TECH_BASE_PROMPT["followup"].format(
                         question=current["content"], answer=_wrap(answer), tech_stack=", ".join(current.get("tags", []))),
                     "Could you go into more detail on how that works?",
                     {"followup_count": state.get("followup_count", 0) + 1}, temperature=0.4, window=8)

    unasked = _unasked(state)
    if not unasked:
        return Reply(None, "OK, that's it for the technical questions.", {"current_question": None})
    next_q = unasked[0]
    updates = {"question_bank": _mark_asked(state.get("question_bank", []), next_q),
               "current_question": next_q, "followup_count": 0}

    if current is None:                      # First question: comment on the self-introduction
        return Reply(INTRO_EVAL_TECH_FIRST_PROMPT.format(
                         intro=_wrap(answer, 800), position=state.get("target_position", ""),
                         first_question=next_q["content"]),
                     f"Thanks for the introduction. Let's start with the technical part. {next_q['content']}",
                     updates, temperature=0.5, window=2)
    return Reply(TECH_BASE_PROMPT["ask_with_feedback"].format(
                     question=current["content"], answer=_wrap(answer),
                     quality=state.get("last_answer_quality"), next_question=next_q["content"]),
                 f"OK, next question. {next_q['content']}", updates, temperature=0.4, window=8)


def _plan_project(state: InterviewState) -> Reply:
    messages = state.get("messages", [])
    projects = state.get("resume_projects", [])
    current = state.get("current_question")
    answer = _last_candidate_message(messages)
    position = state.get("target_position", "")

    if not projects:                         # No resume: ask from what the candidate said
        return Reply(PROJECT_PROMPT["no_resume"].format(position=position, answer=_wrap(answer, 600)),
                     "Tell me about a project you have worked on: what it did and what you built yourself.",
                     {"current_question": {"content": "Project experience (no resume)", "project_name": "", "tech_stack": []}},
                     window=10)

    if current is not None and _project_followup_due(state):
        count = state.get("followup_count", 0) + 1
        return Reply(PROJECT_PROMPT["followup_with_feedback"].format(
                         project_name=current.get("project_name", ""), question=current["content"],
                         answer=_wrap(answer), quality=state.get("last_answer_quality"),
                         tech_stack=", ".join(current.get("tech_stack", [])), question_num=count),
                     "What was the hardest technical problem in that project, and how did you solve it?",
                     {"followup_count": count}, window=10)

    asked = state.get("projects_asked", [])
    next_project = next((p for p in projects if p.get("name") not in asked), None)
    if next_project is None:                 # Every project covered; check_stage moves on next turn
        return Reply(PROJECT_PROMPT["synthesis"].format(position=position),
                     "Across your projects, what was the most complex technical problem, and how did you solve it?",
                     window=10)
    name = next_project.get("name", "your project")
    return Reply(PROJECT_PROMPT["new_project"].format(
                     project_name=name, project_role=next_project.get("role", "developer"),
                     tech_stack=", ".join(next_project.get("tech_stack", [])),
                     highlights="; ".join(next_project.get("highlights", [])) or "(none)",
                     description=str(next_project.get("description", ""))[:300]),
                 f'Let\'s talk about your project "{name}". What was it for, and what did you personally build?',
                 {"current_question": {"content": f"Project: {name}", "project_name": name,
                                       "tech_stack": next_project.get("tech_stack", [])},
                  "projects_asked": asked + [name], "followup_count": 0},
                 window=10)


def _plan_closing(state: InterviewState) -> Reply:
    if state.get("stage_turn_count", 0) == 0:
        return Reply(CLOSING_PROMPT["opening"], "That covers my questions. Is there anything you would like to ask me?",
                     temperature=0.6, window=2)
    return Reply(CLOSING_PROMPT["respond_question"].format(
                     question=_wrap(_last_candidate_message(state.get("messages", [])), 600),
                     position=state.get("target_position", "")),
                 "Good question. Is there anything else you would like to ask?", temperature=0.6, window=4)


PLANNERS = {
    InterviewStage.WARMUP.value:    _plan_warmup,
    InterviewStage.TECH_BASE.value: _plan_tech,
    InterviewStage.PROJECT.value:   _plan_project,
    InterviewStage.CLOSING.value:   _plan_closing,
}


def _chat_messages(state: InterviewState, prompt: str, window: int) -> list:
    """System prompt, the recent dialogue (candidate turns wrapped in tags), then the task.
    The latest candidate message is left out: the task prompt already contains it."""
    result: list = [SystemMessage(content=_system_prompt(state))]
    history = [m for m in state.get("messages", [])[:-1] if not isinstance(m, SystemMessage)]
    for msg in trim_messages_to_window(history, window_size=window):
        if isinstance(msg, HumanMessage):
            text_ = _msg_text(msg)
            result.append(HumanMessage(content=text_ if text_ == START_MESSAGE
                                       else f"<candidate_answer>\n{_wrap(text_)}\n</candidate_answer>"))
        else:
            result.append(msg)
    result.append(HumanMessage(content=prompt))
    return result


async def generate_response_node(state: InterviewState) -> dict:
    """The interviewer's reply for the current stage. If the LLM fails, a fixed reply keeps
    the interview going (it still asks the next question)."""
    stage = state.get("current_stage", InterviewStage.WARMUP.value)
    reply = PLANNERS[stage](state)
    updates = dict(reply.updates)

    text_ = reply.fallback
    if reply.prompt is not None:
        try:
            # streaming=True so /chat/stream receives tokens; ainvoke still returns the whole reply
            response = await get_llm("interview", temperature=reply.temperature, streaming=True).ainvoke(
                _chat_messages(state, reply.prompt, reply.window))
            text_ = _msg_text(response).strip() or reply.fallback
        except Exception as e:
            logger.warning("generate_response.failed", stage=stage, error=str(e))
            updates["fallback_used"] = True

    # Fixed transition line on the first reply of the project and closing stages
    if state.get("stage_turn_count", 0) == 0 and stage in STAGE_TRANSITION_TEXT:
        text_ = f"{STAGE_TRANSITION_TEXT[stage]}\n\n{text_}"

    updates["messages"] = [AIMessage(content=text_)]
    logger.info("generate_response.done", stage=stage, length=len(text_))
    return updates


# ──────────────────────────────────────────────────────────────
# generate_report / save_report / save_memory
# ──────────────────────────────────────────────────────────────

def transcript(messages: list, limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    """Interviewer and candidate turns as text. If too long, the most recent part is kept
    (older turns are in the summary)."""
    lines = []
    for m in messages:
        if isinstance(m, AIMessage):
            lines.append(f"Interviewer: {_msg_text(m)}")
        elif isinstance(m, HumanMessage) and _msg_text(m) != START_MESSAGE:
            lines.append(f"Candidate: <candidate_answer>{_wrap(_msg_text(m))}</candidate_answer>")
    text_ = "\n".join(lines)
    return text_[-limit:] if len(text_) > limit else text_


def overall_score(dimensions: list[dict]) -> tuple[int | None, bool]:
    """Weighted average of the dimension scores, using REPORT_DIMENSIONS. Returns the score and
    whether every dimension was found. Missing dimensions are left out of the average."""
    weights = {name.lower(): w for name, w in REPORT_DIMENSIONS.items()}
    found = {}
    for d in dimensions:
        name = str(d.get("dimension", "")).strip().lower()
        if name in weights and name not in found:
            found[name] = max(0, min(int(d.get("score", 0)), 100))
    if not found:
        return None, False
    total_weight = sum(weights[n] for n in found)
    return round(sum(found[n] * weights[n] for n in found) / total_weight), len(found) == len(weights)


FALLBACK_REPORT = {
    "dimensions": [], "overall_score": None, "strengths": [], "improvements": [],
    "overall_comment": "The report could not be generated. Your answers are saved; please try another interview later.",
    "recommended_topics": [], "next_step_advice": "", "interviewer_instructions_found": False, "fallback": True,
}


def stages_reached(state: dict) -> list[str]:
    """Stages the interview actually went through, in order: those the candidate answered in,
    then the current one. Stages missing from it were skipped (e.g. no projects, or an early end)."""
    stages = list(state.get("answered_stages", []))
    stages.append(state.get("current_stage") or InterviewStage.WARMUP.value)
    return list(dict.fromkeys(stages))


async def generate_report_node(state: InterviewState) -> dict:
    """Five-dimension report from the transcript. The overall score is computed from the
    dimension scores here; a score in the model's output would be ignored."""
    position = state.get("target_position") or "Software Engineer"
    prompt = GENERATE_REPORT_PROMPT.format(
        position=position, total_turns=state.get("total_turn_count", 0),
        history_summary=state.get("existing_summary") or "(none)",
        conversation=transcript(state.get("messages", [])),
    )
    updates: dict = {}
    try:
        result: InterviewReport = await _structured_call(InterviewReport, _system_prompt(state), prompt)
        report = result.model_dump()
        report["overall_score"], complete = overall_score(report["dimensions"])
        if not complete:
            logger.warning("generate_report.dimensions_incomplete", found=[d["dimension"] for d in report["dimensions"]])
        report["interviewer_instructions_found"] = (
            result.interviewer_instructions_found
            or any(e.get("interviewer_instructions_found") for e in state.get("answer_log", [])))
        report["fallback"] = report["overall_score"] is None
    except Exception as e:
        logger.warning("generate_report.failed", error=str(e))
        report = dict(FALLBACK_REPORT)
    if report["fallback"]:
        updates["fallback_used"] = True
    report["stages_reached"] = stages_reached(state)

    score = report["overall_score"]
    closing = ("That's the end of the mock interview. Thank you for taking part!\n\n"
               + (f"**Overall score: {score} / 100**\n\n" if score is not None else "")
               + "The detailed report, with each dimension and what to improve, is in your interview history.")
    logger.info("generate_report.done", overall_score=score, fallback=report["fallback"])
    return {**updates, "report": report, "messages": [AIMessage(content=closing)],
            "structured_output": {"session_id": state.get("session_id"), "report": report,
                                  "total_turns": state.get("total_turn_count", 0)}}


async def save_report_node(state: InterviewState) -> dict:
    thread_id = build_thread_id(state["student_id"], state["session_id"])
    try:
        if not await store_report(thread_id, state.get("report") or {}):
            logger.warning("save_report.not_in_progress", thread_id=thread_id)
    except Exception as e:
        logger.error("save_report.db_failed", error=str(e))
    return {}


async def save_memory_node(state: InterviewState) -> dict:
    """Compress the history and store the summary. Runs in parallel with generate_response on
    the turns route_after_check_stage picks, so it sees the history up to the candidate's message."""
    thread_id = build_thread_id(state["student_id"], state["session_id"])
    try:
        summary = await compress_to_summary(state.get("messages", []), state.get("existing_summary"))
        await store_summary(thread_id, summary)
        return {"existing_summary": summary}
    except Exception as e:
        logger.warning("save_memory.failed", error=str(e))
        return {}
