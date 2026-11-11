# scripts/seed_interview_questions.py
# Loads the English interview question bank (datasets/interview/question_bank.json) into
# interview_questions. The interview agent uses these questions when LLM question generation
# fails, and adds them after the generated ones otherwise.
# Idempotent: replaces the default tenant's bank.
#
# Run from the project root:
#     python -m scripts.seed_interview_questions

import asyncio
import json

from sqlalchemy import text

from backend.config import get_settings
from backend.dependencies import AsyncSessionLocal
from scripts.evaluation.exam_docx import DATASET_DIR

BANK_PATH = DATASET_DIR.parent / "interview" / "question_bank.json"


def load_bank() -> list[dict]:
    return json.loads(BANK_PATH.read_text(encoding="utf-8"))["questions"]


async def main() -> None:
    questions = load_bank()
    tenant = get_settings().default_tenant_id
    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(text("DELETE FROM interview_questions WHERE tenant_id = :tenant"), {"tenant": tenant})
            for q in questions:
                await session.execute(
                    text("""INSERT INTO interview_questions (tenant_id, content, difficulty, tags, target_position)
                            VALUES (:tenant, :content, :difficulty, CAST(:tags AS JSONB), :track)"""),
                    {"tenant": tenant, "content": q["content"], "difficulty": q["difficulty"],
                     "tags": json.dumps(q["tags"]), "track": q["track"]},
                )
    tracks = {}
    for q in questions:
        tracks[q["track"]] = tracks.get(q["track"], 0) + 1
    print(f"Seeded {len(questions)} interview questions: {tracks}")


if __name__ == "__main__":
    asyncio.run(main())
