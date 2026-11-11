# scripts/demo/seed_demo.py
# Loads the demo data for student01: one resume review, one graded exam (confirmed by
# teacher01) and one finished mock interview with its report. No LLM calls: the results come
# from demo_data.json, recorded once from the real agents by generate_demo_data.py.
#
#     python -m scripts.demo.seed_demo
#
# Run scripts/seed_data.py and scripts/seed_standard_exam.py first. This DELETES everything
# student01 already has (resume reviews, exam submissions, interviews, Q&A sessions), so
# rerunning it always gives the same demo.

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text

from backend.core.memory import build_thread_id
from backend.dependencies import AsyncSessionLocal

DATA_PATH = Path(__file__).parent / "demo_data.json"
STUDENT, TEACHER = "student01", "teacher01"

# Placeholders for the account ids in demo_data.json, which differ in every database
STUDENT_ID, TEACHER_ID = "{{student_id}}", "{{teacher_id}}"

# How long ago each part of the story happened
DAYS_AGO = {"resume": 3, "exam_submitted": 2, "exam_published": 1.9, "interview": 1}


async def user_ids(db) -> tuple[str, str, str]:
    rows = (await db.execute(text("SELECT username, id, tenant_id FROM users WHERE username IN (:s, :t)"),
                             {"s": STUDENT, "t": TEACHER})).mappings().all()
    found = {r["username"]: r for r in rows}
    missing = {STUDENT, TEACHER} - set(found)
    if missing:
        raise SystemExit(f"Accounts {sorted(missing)} not found: run scripts/seed_data.py first")
    return str(found[STUDENT]["id"]), str(found[TEACHER]["id"]), found[STUDENT]["tenant_id"]


async def clear_student(db, student_id: str) -> None:
    """Delete everything the student has; exam reviews go with their submissions (cascade)."""
    for table in ("interview_sessions", "exam_submissions", "resume_reviews", "qa_sessions",
                  "knowledge_pending_queue"):
        await db.execute(text(f"DELETE FROM {table} WHERE student_id = :sid"), {"sid": student_id})
    exists = (await db.execute(text("SELECT to_regclass('checkpoints') IS NOT NULL"))).scalar()
    if exists:                                   # Interview state, created on the app's first start
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            await db.execute(text(f"DELETE FROM {table} WHERE thread_id LIKE :prefix"),
                             {"prefix": build_thread_id(student_id, "") + "%"})


def _load_data(student_id: str, teacher_id: str) -> dict:
    raw = DATA_PATH.read_text(encoding="utf-8")
    return json.loads(raw.replace(STUDENT_ID, student_id).replace(TEACHER_ID, teacher_id))


def _ago(days: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def _json(value) -> str | None:
    return None if value is None else json.dumps(value)


async def seed() -> None:
    async with AsyncSessionLocal() as db:
        async with db.begin():
            student_id, teacher_id, tenant_id = await user_ids(db)
            await clear_student(db, student_id)
            data = _load_data(student_id, teacher_id)
            ids = {"student_id": student_id, "tenant_id": tenant_id}

            for r in data["resume_reviews"]:
                await db.execute(text("""
                    INSERT INTO resume_reviews (id, tenant_id, student_id, pdf_minio_path, structured_data,
                                                scores, issues, summary, status, created_at, updated_at)
                    VALUES (:id, :tenant_id, :student_id, :path, CAST(:structured AS JSONB), CAST(:scores AS JSONB),
                            CAST(:issues AS JSONB), CAST(:summary AS JSONB), 'done', :at, :at)"""),
                    {**ids, "id": r["id"], "path": f"resumes/{student_id}/{r['id']}.pdf",
                     "structured": _json(r["structured_data"]), "scores": _json(r["scores"]),
                     "issues": _json(r["issues"]), "summary": _json(r["summary"]), "at": _ago(DAYS_AGO["resume"])})

            for s in data["exam_submissions"]:
                exam = (await db.execute(text("SELECT id FROM exams WHERE id = :id"), {"id": s["exam_id"]})).fetchone()
                if not exam:
                    raise SystemExit("The sample exam is missing: run python -m scripts.seed_standard_exam first")
                await db.execute(text("""
                    INSERT INTO exam_submissions (id, tenant_id, exam_id, student_id, source, status, submitted_at,
                                                  published_at, weak_points, weak_points_summary, created_at, updated_at)
                    VALUES (:id, :tenant_id, :exam_id, :student_id, 'word', 'published', :submitted, :published,
                            CAST(:weak AS JSONB), :weak_summary, :submitted, :published)"""),
                    {**ids, "id": s["id"], "exam_id": s["exam_id"], "submitted": _ago(DAYS_AGO["exam_submitted"]),
                     "published": _ago(DAYS_AGO["exam_published"]), "weak": _json(s["weak_points"]),
                     "weak_summary": s["weak_points_summary"]})
                for q in s["reviews"]:
                    await db.execute(text("""
                        INSERT INTO exam_reviews (id, submission_id, question_id, question_type, knowledge_tag,
                                                  student_answer, ai_score, ai_feedback, ai_raw_result, teacher_score,
                                                  teacher_comment, final_score, needs_review, reviewed_by, reviewed_at,
                                                  created_at, updated_at)
                        VALUES (:id, :submission_id, :question_id, :question_type, :knowledge_tag, :student_answer,
                                :ai_score, :ai_feedback, CAST(:ai_raw AS JSONB), :teacher_score, :teacher_comment,
                                :final_score, :needs_review, :reviewed_by, :reviewed_at, :created, :reviewed_at)"""),
                        {**{k: q[k] for k in ("id", "question_id", "question_type", "knowledge_tag", "student_answer",
                                              "ai_score", "ai_feedback", "teacher_score", "teacher_comment",
                                              "final_score", "needs_review")},
                         "submission_id": s["id"], "ai_raw": _json(q["ai_raw_result"]),
                         "reviewed_by": teacher_id if q["reviewed_by_teacher"] else None,
                         "reviewed_at": _ago(DAYS_AGO["exam_published"]) if q["reviewed_by_teacher"] else None,
                         "created": _ago(DAYS_AGO["exam_submitted"])})

            for i in data["interview_sessions"]:
                finished = _ago(DAYS_AGO["interview"])
                await db.execute(text("""
                    INSERT INTO interview_sessions (id, tenant_id, student_id, session_id, thread_id, target_position,
                                                    resume_review_id, summary, report, overall_score, status,
                                                    finished_at, created_at, updated_at)
                    VALUES (:id, :tenant_id, :student_id, :session_id, :thread_id, :position, :resume_review_id,
                            :summary, CAST(:report AS JSONB), :score, 'finished', :finished, :started, :finished)"""),
                    {**ids, "id": i["id"], "session_id": i["session_id"],
                     "thread_id": build_thread_id(student_id, i["session_id"]), "position": i["target_position"],
                     "resume_review_id": i["resume_review_id"], "summary": i["summary"],
                     "report": _json(i["report"]), "score": i["overall_score"],
                     "finished": finished, "started": finished - timedelta(minutes=25)})

    print(f"Demo data loaded for {STUDENT}: {len(data['resume_reviews'])} resume review, "
          f"{len(data['exam_submissions'])} graded exam, {len(data['interview_sessions'])} finished interview")


if __name__ == "__main__":
    asyncio.run(seed())
