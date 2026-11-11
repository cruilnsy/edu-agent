# backend/api/v1/exam.py
# Exam grading API. Students submit a .docx answer sheet; the graph grades it in the background
# and pauses for the teacher; the teacher reviews and confirms, which publishes the result.
# Every query is scoped to the caller's tenant, and students only see their own submissions.

import asyncio
import json
import os
import tempfile
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.agents.exam.graph import build_exam_graph
from backend.core.logger import get_logger
from backend.core.rate_limit import limit_exam_submit
from backend.core.memory import build_config
from backend.dependencies import AsyncSessionLocal, get_current_user, require_teacher

router = APIRouter()
logger = get_logger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024      # An answer sheet is text; 5 MB is plenty
MAX_COMMENT_CHARS = 2000

# Compiled once at import time, not per request
_graph = build_exam_graph()

# Keep references to background tasks so they are not garbage-collected before they finish
_background_tasks: set[asyncio.Task] = set()


def _parse_uuid(value: str, what: str) -> str:
    """Reject malformed ids with 400 instead of letting the DB fail with 500."""
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid {what}")


@router.post("/submit", status_code=202, dependencies=[Depends(limit_exam_submit)])
async def submit_exam(
    exam_id: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Submit a .docx answer sheet. Returns 202 with a submission_id; grading runs in the background."""
    exam_id = _parse_uuid(exam_id, "exam id")
    if not (file.filename or "").lower().endswith(".docx"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only .docx files are accepted")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    student_id = current_user["user_id"]
    tenant_id = current_user["tenant_id"]

    async with AsyncSessionLocal() as session:
        exam_row = (await session.execute(
            text("SELECT id FROM exams WHERE id = :exam_id AND tenant_id = :tenant_id AND is_active"),
            {"exam_id": exam_id, "tenant_id": tenant_id},
        )).fetchone()
        if not exam_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")

        # One submission per student per exam
        existing = (await session.execute(
            text("SELECT id, status FROM exam_submissions WHERE exam_id = :exam_id AND student_id = :student_id"),
            {"exam_id": exam_id, "student_id": student_id},
        )).fetchone()

    if existing and existing[1] in ("pending_review", "reviewed", "published"):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="This exam already has a result waiting for review or published")

    submission_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as session:
        async with session.begin():
            if existing:   # 'submitted' or 'ai_processing': replace the earlier attempt
                await session.execute(text("DELETE FROM exam_submissions WHERE id = :id"), {"id": str(existing[0])})
            await session.execute(
                text("""
                    INSERT INTO exam_submissions (id, tenant_id, exam_id, student_id, status, submitted_at)
                    VALUES (:id, :tenant_id, :exam_id, :student_id, 'ai_processing', NOW())
                """),
                {"id": submission_id, "tenant_id": tenant_id, "exam_id": exam_id, "student_id": student_id},
            )

    # The file name is our own uuid, never the uploaded name
    tmp_path = os.path.join(tempfile.gettempdir(), f"exam_{submission_id}.docx")
    with open(tmp_path, "wb") as f:
        f.write(content)

    initial_state = {
        "messages":            [],
        "student_id":          student_id,
        "tenant_id":           tenant_id,
        "session_id":          submission_id,
        "exam_id":             exam_id,
        "submission_id":       submission_id,
        "word_file_path":      tmp_path,
        "parsed_questions":    [],
        "objective_results":   [],
        "subjective_results":  [],
        "code_results":        [],
        "pre_review_summary":  {},
        "weak_points":         [],
        "weak_points_summary": "",
        "teacher_notified":    False,
        "teacher_decision":    None,
        "final_results":       [],
        "published":           False,
        "fallback_used":       False,
        "structured_output":   None,
    }

    def _on_task_done(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
        failed = not t.cancelled() and t.exception() is not None
        if failed:
            logger.error("exam.background_task_failed", submission_id=submission_id, error=str(t.exception()))

        async def _cleanup():
            if failed:
                # Let the student resubmit: move the submission back from ai_processing
                try:
                    async with AsyncSessionLocal() as db:
                        async with db.begin():
                            await db.execute(
                                text("""
                                    UPDATE exam_submissions SET status = 'submitted', updated_at = NOW()
                                    WHERE id = :sid AND status = 'ai_processing'
                                """),
                                {"sid": submission_id},
                            )
                except Exception as db_err:
                    logger.warning("exam.status_reset_failed", submission_id=submission_id, error=str(db_err))
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        cleanup = asyncio.ensure_future(_cleanup())
        _background_tasks.add(cleanup)
        cleanup.add_done_callback(_background_tasks.discard)

    # ainvoke returns when the graph reaches interrupt() in teacher_review; the paused state
    # stays in the checkpointer until /confirm resumes it.
    task = asyncio.create_task(_graph.ainvoke(initial_state, config=build_config(student_id, submission_id)))
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)

    logger.info("exam.submitted", submission_id=submission_id, exam_id=exam_id)
    return {
        "submission_id": submission_id,
        "status":        "ai_processing",
        "message":       "Submitted. The AI is grading it; a teacher will review the result.",
    }


@router.get("/exams")
async def list_exams(current_user: dict = Depends(get_current_user)):
    """Active exams in the caller's tenant, so a student can pick one to submit."""
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            text("""
                SELECT e.id, e.title, e.description, COUNT(q.id) AS question_count,
                       COALESCE(SUM(q.score), 0) AS full_score
                FROM exams e LEFT JOIN questions q ON q.exam_id = e.id
                WHERE e.tenant_id = :tenant_id AND e.is_active = TRUE
                GROUP BY e.id ORDER BY e.created_at DESC
            """),
            {"tenant_id": current_user["tenant_id"]},
        )).mappings().all()
    return {"items": [{"exam_id": str(r["id"]), "title": r["title"], "description": r["description"] or "",
                       "question_count": r["question_count"], "full_score": r["full_score"]} for r in rows]}


@router.get("/my-submissions")
async def list_my_submissions(current_user: dict = Depends(get_current_user)):
    """The caller's 20 most recent submissions."""
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            text("""
                SELECT es.id AS submission_id, e.title AS exam_title, es.exam_id, es.status, es.submitted_at
                FROM exam_submissions es
                LEFT JOIN exams e ON e.id = es.exam_id
                WHERE es.student_id = :student_id AND es.tenant_id = :tenant_id
                ORDER BY es.submitted_at DESC
                LIMIT 20
            """),
            {"student_id": current_user["user_id"], "tenant_id": current_user["tenant_id"]},
        )).mappings().all()

    return {"items": [{
        "submission_id": str(r["submission_id"]),
        "exam_id":       str(r["exam_id"]),
        "exam_title":    r["exam_title"] or "",
        "status":        r["status"],
        "submitted_at":  r["submitted_at"].isoformat() if r["submitted_at"] else None,
    } for r in rows]}


@router.get("/my-submissions/{submission_id}")
async def get_my_submission(submission_id: str, current_user: dict = Depends(get_current_user)):
    """Status of one of the caller's submissions; the full result once it is published."""
    submission_id = _parse_uuid(submission_id, "submission id")
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            text("""
                SELECT id, status, weak_points, weak_points_summary FROM exam_submissions
                WHERE id = :sid AND student_id = :student_id AND tenant_id = :tenant_id
            """),
            {"sid": submission_id, "student_id": current_user["user_id"], "tenant_id": current_user["tenant_id"]},
        )).mappings().fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Submission not found")
    if row["status"] != "published":       # Not published yet: the student keeps polling
        return {"submission_id": submission_id, "status": row["status"]}

    async with AsyncSessionLocal() as session:
        reviews = (await session.execute(
            text("""
                SELECT er.question_id, er.question_type, er.student_answer, er.ai_score, er.ai_feedback,
                       er.teacher_comment, er.final_score, er.needs_review, er.ai_raw_result,
                       q.question_no, q.correct_answer, q.score AS full_score
                FROM exam_reviews er
                JOIN questions q ON q.id = er.question_id
                WHERE er.submission_id = :sid
                ORDER BY q.question_no
            """),
            {"sid": submission_id},
        )).mappings().all()

    by_question, total, full = [], 0, 0
    for r in reviews:
        raw = r["ai_raw_result"] or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                raw = {}
        final = r["final_score"] if r["final_score"] is not None else r["ai_score"]
        by_question.append({
            "question_id":      str(r["question_id"]),
            "question_no":      r["question_no"],
            "question_type":    r["question_type"],
            "full_score":       r["full_score"],
            "score":            r["ai_score"],
            "final_score":      final,
            "student_answer":   r["student_answer"] or "",
            "correct_answer":   r["correct_answer"] or "",
            "ai_feedback":      r["ai_feedback"] or "",
            "teacher_comment":  r["teacher_comment"] or "",
            "needs_review":     r["needs_review"],
            "point_results":    raw.get("point_results", []),
            "quality_feedback": raw.get("quality_feedback", []),
        })
        total += final or 0
        full += r["full_score"] or 0

    return {
        "submission_id": submission_id,
        "status":        "published",
        "pre_review_summary": {
            "total_score": total,
            "full_score":  full,
            "score_rate":  round(total / full, 4) if full else 0,
            "by_question": by_question,
        },
        "weak_points":         row["weak_points"] if isinstance(row["weak_points"], list) else [],
        "weak_points_summary": row["weak_points_summary"] or "",
    }


# ── Teacher endpoints ──

async def _load_submission(submission_id: str, tenant_id: str) -> dict:
    """The submission's student and status, only within the teacher's tenant (404 otherwise,
    so a teacher cannot tell whether another tenant's submission exists)."""
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            text("SELECT student_id, status FROM exam_submissions WHERE id = :sid AND tenant_id = :tenant_id"),
            {"sid": submission_id, "tenant_id": tenant_id},
        )).mappings().fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Submission not found")
    return {"student_id": str(row["student_id"]), "status": row["status"]}


async def _paused_state(submission_id: str, student_id: str) -> dict:
    """State saved when the graph paused at teacher_review."""
    snapshot = await _graph.aget_state(build_config(student_id, submission_id))
    if not snapshot or not snapshot.values:
        # The in-memory checkpointer loses paused graphs when the server restarts
        raise HTTPException(status_code=404, detail="No AI pre-review found for this submission")
    return snapshot.values


@router.get("/pending-reviews")
async def get_pending_reviews(current_user: dict = Depends(require_teacher)):
    """Submissions in the teacher's tenant that wait for review."""
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            text("""
                SELECT es.id AS submission_id, es.student_id, u.username AS student_name,
                       e.title AS exam_title, es.submitted_at
                FROM exam_submissions es
                JOIN users u ON u.id = es.student_id
                JOIN exams e ON e.id = es.exam_id
                WHERE es.tenant_id = :tenant_id AND es.status = 'pending_review'
                ORDER BY es.submitted_at DESC
            """),
            {"tenant_id": current_user["tenant_id"]},
        )).mappings().all()

    items = []
    for row in rows:
        submission_id = str(row["submission_id"])
        pre_review = {"total_score": 0, "full_score": 0, "needs_review_count": 0}
        weak_points = []
        try:
            values = await _paused_state(submission_id, str(row["student_id"]))
            summary = values.get("pre_review_summary", {})
            pre_review = {k: summary.get(k, 0) for k in pre_review}
            weak_points = values.get("weak_points", [])
        except HTTPException:
            logger.warning("exam.pending_review_state_missing", submission_id=submission_id)

        items.append({
            "submission_id": submission_id,
            "student_name":  row["student_name"],
            "exam_title":    row["exam_title"],
            "submitted_at":  row["submitted_at"].isoformat() if row["submitted_at"] else None,
            "pre_review":    pre_review,
            "weak_points":   weak_points[:3],     # The list view shows the top 3
        })
    return {"items": items, "total": len(items)}


@router.get("/submissions/{submission_id}/review")
async def get_submission_review(submission_id: str, current_user: dict = Depends(require_teacher)):
    """The AI pre-review of one submission, read from the paused graph state."""
    submission_id = _parse_uuid(submission_id, "submission id")
    submission = await _load_submission(submission_id, current_user["tenant_id"])
    if submission["status"] == "ai_processing":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The AI is still grading this submission")

    values = await _paused_state(submission_id, submission["student_id"])
    return {
        "submission_id":       submission_id,
        "status":              submission["status"],
        "student_id":          submission["student_id"],
        "pre_review_summary":  values.get("pre_review_summary", {}),
        "weak_points":         values.get("weak_points", []),
        "weak_points_summary": values.get("weak_points_summary", ""),
    }


class Modification(BaseModel):
    question_id: str
    new_score:   int | None = Field(default=None, ge=0)
    comment:     str | None = Field(default=None, max_length=MAX_COMMENT_CHARS)


class ConfirmRequest(BaseModel):
    action:        Literal["approve", "modify"]
    modifications: list[Modification] = []


def validate_modifications(modifications: list[Modification], by_question: list[dict]) -> None:
    """Every modified question must belong to this submission, and new scores must fit the full score."""
    full_scores = {r["question_id"]: r["full_score"] for r in by_question}
    for mod in modifications:
        if mod.question_id not in full_scores:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Question {mod.question_id} is not part of this submission")
        if mod.new_score is not None and mod.new_score > full_scores[mod.question_id]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Score for question {mod.question_id} exceeds its full score")


async def _set_status(submission_id: str, tenant_id: str, old: str, new: str) -> bool:
    """Move a submission from one status to another; False if it was not in the old status."""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            moved = (await session.execute(
                text("""
                    UPDATE exam_submissions SET status = :new, updated_at = NOW()
                    WHERE id = :sid AND tenant_id = :tenant_id AND status = :old
                    RETURNING id
                """),
                {"sid": submission_id, "tenant_id": tenant_id, "old": old, "new": new},
            )).fetchone()
    return moved is not None


@router.post("/submissions/{submission_id}/confirm")
async def confirm_review(submission_id: str, req: ConfirmRequest, current_user: dict = Depends(require_teacher)):
    """Approve or adjust the AI grades, then publish: resumes the graph from interrupt()."""
    submission_id = _parse_uuid(submission_id, "submission id")
    tenant_id = current_user["tenant_id"]
    submission = await _load_submission(submission_id, tenant_id)
    if submission["status"] != "pending_review":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This submission is not waiting for review")

    values = await _paused_state(submission_id, submission["student_id"])
    if req.action == "modify":
        validate_modifications(req.modifications, values.get("pre_review_summary", {}).get("by_question", []))

    # Claim the submission first, so two teachers confirming at once cannot both publish
    if not await _set_status(submission_id, tenant_id, "pending_review", "reviewed"):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="This submission is not waiting for review")

    decision = {
        "action":        req.action,
        "modifications": [m.model_dump() for m in req.modifications] if req.action == "modify" else [],
        "teacher_id":    current_user["user_id"],
    }
    try:
        result = await _graph.ainvoke(Command(resume=decision), config=build_config(submission["student_id"], submission_id))
    except Exception as e:
        logger.error("exam.confirm_failed", submission_id=submission_id, error=str(e))
        await _set_status(submission_id, tenant_id, "reviewed", "pending_review")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Publishing failed; please retry")

    structured = result.get("structured_output") or {}
    logger.info("exam.published", submission_id=submission_id, final_score=structured.get("final_score", 0))
    return {
        "submission_id":       submission_id,
        "status":              "published",
        "final_score":         structured.get("final_score", 0),
        "full_score":          structured.get("full_score", 0),
        "score_rate":          structured.get("score_rate", 0),
        "weak_points":         structured.get("weak_points", []),
        "weak_points_summary": structured.get("weak_points_summary", ""),
    }
