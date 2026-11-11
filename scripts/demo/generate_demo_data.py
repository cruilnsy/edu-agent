# scripts/demo/generate_demo_data.py
# Records demo_data.json by running the real agents once through the API (calls the LLM; a
# few minutes, well under one US cent). Only needed when the agents or tables change; the demo
# itself is loaded offline by seed_demo.py.
#
#     python -m scripts.demo.generate_demo_data
#
# The story uses fictional people from the eval datasets: Jordan Avery (resume r01) uploads a
# resume, submits the sample exam (answers s02), teacher01 confirms the grades, and Jordan
# takes a backend mock interview played by the candidate simulator. DELETES student01's data
# first, like seed_demo.py.

import asyncio
import copy
import json
import os
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv
from sqlalchemy import text

from backend.config import ENV_LOCAL_PATH
from backend.dependencies import AsyncSessionLocal
from scripts.demo.seed_demo import DATA_PATH, STUDENT, STUDENT_ID, TEACHER, TEACHER_ID, clear_student, user_ids
from scripts.evaluation.exam_docx import load_exam, load_submissions, render_answer_sheet
from scripts.evaluation.interview_sim import candidate_reply, load_candidates
from scripts.evaluation.resume_pdf import render_pdf, resume_sources

RESUME_ID, EXAM_ANSWERS_ID, CANDIDATE_ID = "r01", "s02", "c01"
MAX_INTERVIEW_REPLIES = 30
POLL_SECONDS, POLL_LIMIT = 2, 150

# Jordan's projects from resume r01, so the simulated answers match what the interviewer reads
JORDAN = {
    "persona": ("A backend engineer with three years of experience in Python, FastAPI and PostgreSQL. "
                "Answers are precise and concrete: explains how things work, names trade-offs and gives "
                "numbers from their work. Admits the edge of their knowledge honestly."),
    "intro": ("Hi, I'm a backend engineer with three years at a logistics company. I built our order-tracking "
              "API in FastAPI and PostgreSQL, moved six services from cron jobs to Celery, and I maintain a "
              "small open-source rate limiter for asyncio services."),
    "resume": {
        "skills_list": ["Python", "FastAPI", "PostgreSQL", "Redis", "Celery", "Docker", "AWS"],
        "projects": [
            {"name": "Order-tracking API", "role": "Backend engineer", "tech_stack": ["FastAPI", "PostgreSQL", "Redis"],
             "description": "Serves 1,200 requests per second at a p95 of 85 ms; Redis caches carrier rate lookups."},
            {"name": "Celery migration", "role": "Lead", "tech_stack": ["Celery", "Python"],
             "description": "Moved 6 cron jobs to Celery with retries and dead-letter queues; failed shipments 2.1% -> 0.3%."},
            {"name": "Rate limiter library", "role": "Author", "tech_stack": ["Python", "asyncio", "Redis"],
             "description": "Open-source token-bucket rate limiter for asyncio services with Redis and in-memory backends."},
        ],
    },
}


async def login(client: httpx.AsyncClient, username: str, password_env: str) -> dict:
    resp = await client.post("/api/v1/auth/login", json={"username": username, "password": os.environ[password_env]})
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def poll(client: httpx.AsyncClient, url: str, headers: dict, done) -> dict:
    for _ in range(POLL_LIMIT):
        body = (await client.get(url, headers=headers)).json()
        if done(body):
            return body
        await asyncio.sleep(POLL_SECONDS)
    raise TimeoutError(f"{url} did not finish")


async def review_resume(client, student: dict, workdir: Path) -> str:
    pdf = render_pdf(resume_sources([RESUME_ID])[0], workdir / "resume.pdf")
    resp = await client.post("/api/v1/resume/upload", headers=student,
                             files={"file": ("jordan_avery_resume.pdf", pdf.read_bytes(), "application/pdf")})
    resp.raise_for_status()
    review_id = resp.json()["review_id"]
    review = await poll(client, f"/api/v1/resume/reviews/{review_id}", student, lambda b: b.get("status") != "processing")
    assert review["status"] == "done", review
    print(f"resume review done: weighted score {review.get('weighted_score')}")
    return review_id


async def grade_exam(client, student: dict, teacher: dict, workdir: Path) -> None:
    exam = load_exam()
    answers = next(s for s in load_submissions() if s["id"] == EXAM_ANSWERS_ID)["answers"]
    labels = next(l for l in json.loads((Path("scripts/evaluation/datasets/exam/labels.json")).read_text())
                  if l["id"] == EXAM_ANSWERS_ID)["scores"]
    async with AsyncSessionLocal() as db:
        exam_id = str((await db.execute(text("SELECT id FROM exams WHERE title = :t"), {"t": exam["title"]})).scalar())
        question_ids = {str(no): str(qid) for no, qid in (await db.execute(
            text("SELECT question_no, id FROM questions WHERE exam_id = :e"), {"e": exam_id})).all()}

    sheet = render_answer_sheet(exam, answers, workdir / "answers.docx")
    resp = await client.post("/api/v1/exam/submit", headers=student, data={"exam_id": exam_id},
                             files={"file": ("answers.docx", sheet.read_bytes(),
                                             "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    resp.raise_for_status()
    submission_id = resp.json()["submission_id"]
    await poll(client, f"/api/v1/exam/my-submissions/{submission_id}", student,
               lambda b: b.get("status") not in ("submitted", "ai_processing"))

    # The teacher corrects every AI mark that differs from the hand label
    review = (await client.get(f"/api/v1/exam/submissions/{submission_id}/review", headers=teacher)).json()
    ai_scores = {r["question_id"]: r["score"] for r in review["pre_review_summary"]["by_question"]}
    modifications = [{"question_id": question_ids[no], "new_score": score,
                      "comment": "Adjusted after checking the scoring points."}
                     for no, score in labels.items() if ai_scores.get(question_ids[no]) != score]
    action = "modify" if modifications else "approve"
    resp = await client.post(f"/api/v1/exam/submissions/{submission_id}/confirm", headers=teacher,
                             json={"action": action, "modifications": modifications})
    resp.raise_for_status()
    print(f"exam published: {resp.json()['final_score']}/{resp.json()['full_score']}, "
          f"{len(modifications)} marks changed by the teacher")


async def interview(client, student: dict, resume_review_id: str) -> None:
    candidate = copy.deepcopy(next(c for c in load_candidates() if c["id"] == CANDIDATE_ID))
    candidate.update(copy.deepcopy(JORDAN))
    resp = await client.post("/api/v1/interview/sessions", headers=student,
                             json={"target_position": candidate["target_position"], "resume_review_id": resume_review_id})
    resp.raise_for_status()
    session_id, dialogue = resp.json()["session_id"], [("Interviewer", resp.json()["message"])]
    for reply_no in range(MAX_INTERVIEW_REPLIES):
        answer = await candidate_reply(candidate, dialogue, reply_no)
        dialogue.append(("Candidate", answer))
        resp = await client.post(f"/api/v1/interview/sessions/{session_id}/chat", headers=student,
                                 json={"message": answer})
        resp.raise_for_status()
        turn = resp.json()
        dialogue.append(("Interviewer", turn["reply"]))
        if turn["is_finished"]:
            print(f"interview finished after {reply_no + 1} replies: {turn['report_summary']['overall_score']}")
            return
    raise RuntimeError("The interview did not finish")


async def export(student_id: str, teacher_id: str) -> None:
    """Write the student's rows to demo_data.json, with the account ids as placeholders."""
    async with AsyncSessionLocal() as db:
        resumes = (await db.execute(text("""SELECT id, structured_data, scores, issues, summary FROM resume_reviews
                                            WHERE student_id = :s"""), {"s": student_id})).mappings().all()
        submissions = (await db.execute(text("""SELECT id, exam_id, weak_points, weak_points_summary
                                                FROM exam_submissions WHERE student_id = :s"""),
                                        {"s": student_id})).mappings().all()
        exam_data = []
        for s in submissions:
            reviews = (await db.execute(text("""
                SELECT id, question_id, question_type, knowledge_tag, student_answer, ai_score, ai_feedback,
                       ai_raw_result, teacher_score, teacher_comment, final_score, needs_review,
                       reviewed_by IS NOT NULL AS reviewed_by_teacher
                FROM exam_reviews WHERE submission_id = :id"""), {"id": s["id"]})).mappings().all()
            exam_data.append({**dict(s), "reviews": [dict(r) for r in reviews]})
        interviews = (await db.execute(text("""SELECT id, session_id, target_position, resume_review_id, summary,
                                                      report, overall_score
                                               FROM interview_sessions WHERE student_id = :s AND status = 'finished'"""),
                                       {"s": student_id})).mappings().all()

    data = {"resume_reviews": [dict(r) for r in resumes], "exam_submissions": exam_data,
            "interview_sessions": [dict(i) for i in interviews]}
    raw = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    raw = raw.replace(student_id, STUDENT_ID).replace(teacher_id, TEACHER_ID)
    DATA_PATH.write_text(raw + "\n", encoding="utf-8")
    print(f"wrote {DATA_PATH}")


async def main() -> None:
    load_dotenv(ENV_LOCAL_PATH)                      # SEED_*_PASSWORD for the two logins
    async with AsyncSessionLocal() as db:
        async with db.begin():
            student_id, teacher_id, _ = await user_ids(db)
            await clear_student(db, student_id)

    from backend.main import app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://demo", timeout=120) as client:
        student = await login(client, STUDENT, "SEED_STUDENT_PASSWORD")
        teacher = await login(client, TEACHER, "SEED_TEACHER_PASSWORD")
        with tempfile.TemporaryDirectory() as tmp:
            review_id = await review_resume(client, student, Path(tmp))
            await grade_exam(client, student, teacher, Path(tmp))
        await interview(client, student, review_id)
    await export(student_id, teacher_id)


if __name__ == "__main__":
    asyncio.run(main())
