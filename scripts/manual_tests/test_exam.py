# scripts/manual_tests/test_exam.py
# End-to-end exam flow against a running backend (port 8010):
#   student submits -> AI grades -> student is denied teacher endpoints -> teacher reviews
#   -> teacher approves (or modifies with --modify) -> student reads the published result.
#
# Needs: python -m scripts.seed_standard_exam, and seed passwords in .env.local.
# Run from the project root:
#     python -m scripts.manual_tests.test_exam
#     python -m scripts.manual_tests.test_exam --modify --student s03

import argparse
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv(".env.local")

from backend.dependencies import AsyncSessionLocal  # noqa: E402  (needs the env loaded first)
from scripts.evaluation.exam_docx import load_exam, load_submissions, render_answer_sheet  # noqa: E402

BASE_URL = "http://localhost:8010/api/v1"     # Port 8000 is taken by another app on this machine
STUDENT = ("student01", os.environ["SEED_STUDENT_PASSWORD"])
TEACHER = ("teacher01", os.environ["SEED_TEACHER_PASSWORD"])


def login(username: str, password: str) -> dict:
    resp = httpx.post(f"{BASE_URL}/auth/login", json={"username": username, "password": password}, trust_env=False)
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def clear_old_submission(exam_id: str) -> None:
    """Delete the student's earlier submission; otherwise /submit returns 409 on a rerun."""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(
                text("""
                    DELETE FROM exam_submissions
                    WHERE exam_id = :exam_id AND student_id = (SELECT id FROM users WHERE username = :username)
                """),
                {"exam_id": exam_id, "username": STUDENT[0]},
            )


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="s01", help="Which answer set in datasets/exam/submissions.json to submit")
    parser.add_argument("--modify", action="store_true", help="Teacher changes one short-answer score instead of approving")
    args = parser.parse_args()

    exam = load_exam()
    answers = next(s["answers"] for s in load_submissions() if s["id"] == args.student)
    asyncio.run(clear_old_submission(exam["exam_id"]))

    section("1. Student submits")
    student = login(*STUDENT)
    with tempfile.TemporaryDirectory() as tmp:
        sheet = render_answer_sheet(exam, answers, Path(tmp) / "answer_sheet.docx")
        with open(sheet, "rb") as f:
            resp = httpx.post(f"{BASE_URL}/exam/submit", headers=student, data={"exam_id": exam["exam_id"]},
                              files={"file": ("answer_sheet.docx", f)}, timeout=30, trust_env=False)
    resp.raise_for_status()
    submission_id = resp.json()["submission_id"]
    print(f"submission_id: {submission_id}")

    section("2. Wait for AI grading")
    for attempt in range(30):
        time.sleep(5)
        poll = httpx.get(f"{BASE_URL}/exam/my-submissions/{submission_id}", headers=student, trust_env=False).json()
        print(f"  poll {attempt + 1}: {poll['status']}")
        if poll["status"] == "pending_review":
            break
    else:
        sys.exit("Timed out; check the backend log")

    section("3. Student is denied the teacher endpoints")
    denied = [
        httpx.get(f"{BASE_URL}/exam/pending-reviews", headers=student, trust_env=False),
        httpx.get(f"{BASE_URL}/exam/submissions/{submission_id}/review", headers=student, trust_env=False),
        httpx.post(f"{BASE_URL}/exam/submissions/{submission_id}/confirm", headers=student,
                   json={"action": "approve"}, timeout=30, trust_env=False),
    ]
    for r in denied:
        print(f"  {r.request.method} {r.request.url.path} -> {r.status_code}")
        assert r.status_code == 403

    section("4. Teacher reads the AI pre-review")
    teacher = login(*TEACHER)
    review = httpx.get(f"{BASE_URL}/exam/submissions/{submission_id}/review", headers=teacher, trust_env=False)
    review.raise_for_status()
    summary = review.json()["pre_review_summary"]
    print(f"AI total: {summary['total_score']}/{summary['full_score']}, needs review: {summary['needs_review_count']}")
    for q in summary["by_question"]:
        flag = " (needs review)" if q.get("needs_review") else ""
        print(f"  Q{q['question_no']} [{q['question_type']}] {q['score']}/{q['full_score']}{flag}")

    section("5. Teacher confirms")
    body = {"action": "approve"}
    if args.modify:
        target = next(q for q in summary["by_question"] if q["question_type"] == "short_answer")
        body = {"action": "modify", "modifications": [
            {"question_id": target["question_id"], "new_score": target["full_score"] // 2,
             "comment": "Teacher adjustment: half marks."}]}
        bad = httpx.post(f"{BASE_URL}/exam/submissions/{submission_id}/confirm", headers=teacher, timeout=30,
                         json={"action": "modify", "modifications": [
                             {"question_id": target["question_id"], "new_score": target["full_score"] + 1}]},
                         trust_env=False)
        print(f"  score above full -> {bad.status_code}")
        assert bad.status_code == 400
    confirm = httpx.post(f"{BASE_URL}/exam/submissions/{submission_id}/confirm", headers=teacher, json=body,
                         timeout=60, trust_env=False)
    confirm.raise_for_status()
    print(f"Published: {confirm.json()['final_score']}/{confirm.json()['full_score']}")
    again = httpx.post(f"{BASE_URL}/exam/submissions/{submission_id}/confirm", headers=teacher, json=body,
                       timeout=30, trust_env=False)
    print(f"  confirm twice -> {again.status_code}")
    assert again.status_code == 409

    section("6. Student reads the result")
    final = httpx.get(f"{BASE_URL}/exam/my-submissions/{submission_id}", headers=student, trust_env=False).json()
    assert final["status"] == "published"
    for q in final["pre_review_summary"]["by_question"]:
        print(f"  Q{q['question_no']} {q['final_score']}/{q['full_score']}  {q['ai_feedback'][:60]}")
    print("\nEnd-to-end exam flow passed")


if __name__ == "__main__":
    main()
