# scripts/seed_standard_exam.py
# Loads the sample exam (questions, answer keys, scoring points) into exams / questions /
# scoring_points, so load_questions_meta has something to read. The exam itself lives in
# scripts/evaluation/datasets/exam/exam.json, shared with the exam eval.
# Idempotent: deletes the exam by its fixed id, then inserts. Submissions to this sample exam
# (e.g. from scripts/manual_tests/test_exam.py) are deleted first, since they reference the exam;
# their reviews, questions and scoring points cascade.
#
# Run from the project root:
#     python -m scripts.seed_standard_exam

import asyncio

from sqlalchemy import text

from backend.config import get_settings
from backend.dependencies import AsyncSessionLocal
from scripts.evaluation.exam_docx import load_exam


async def main() -> None:
    exam = load_exam()
    tenant = get_settings().default_tenant_id

    async with AsyncSessionLocal() as session:
        async with session.begin():
            removed = (await session.execute(
                text("DELETE FROM exam_submissions WHERE exam_id = :id"), {"id": exam["exam_id"]},
            )).rowcount
            await session.execute(text("DELETE FROM exams WHERE id = :id"), {"id": exam["exam_id"]})
            await session.execute(
                text("""INSERT INTO exams (id, tenant_id, title, description, is_active)
                        VALUES (:id, :tenant, :title, :description, TRUE)"""),
                {"id": exam["exam_id"], "tenant": tenant, "title": exam["title"], "description": exam["description"]},
            )
            for q in exam["questions"]:
                await session.execute(
                    text("""INSERT INTO questions (id, tenant_id, exam_id, question_no, question_type,
                                                   content, correct_answer, score, knowledge_tag)
                            VALUES (:id, :tenant, :exam_id, :no, :type, :content, :answer, :score, :tag)"""),
                    {"id": q["question_id"], "tenant": tenant, "exam_id": exam["exam_id"],
                     "no": q["question_no"], "type": q["question_type"], "content": q["content"],
                     "answer": q["correct_answer"], "score": q["score"], "tag": q["knowledge_tag"]},
                )
                # clock_timestamp(), not NOW(): NOW() is fixed for the whole transaction, and
                # load_questions_meta orders the points by created_at to keep the rubric order
                for sp in q.get("scoring_points", []):
                    await session.execute(
                        text("""INSERT INTO scoring_points (question_id, point_desc, point_score, is_active, created_at)
                                VALUES (:qid, :desc, :score, TRUE, clock_timestamp())"""),
                        {"qid": q["question_id"], "desc": sp["desc"], "score": sp["score"]},
                    )

        total = (await session.execute(
            text("SELECT COUNT(*), SUM(score) FROM questions WHERE exam_id = :id"), {"id": exam["exam_id"]},
        )).one()

    if removed:
        print(f"Deleted {removed} earlier submission(s) to the sample exam")
    print(f"Seeded exam {exam['exam_id']} ({exam['title']}): {total[0]} questions, {total[1]} points")


if __name__ == "__main__":
    asyncio.run(main())
