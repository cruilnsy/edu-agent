# backend/db/migrations.py
#
# Schema patches run at startup. All are idempotent and safe to rerun.
# Rules:
#   - Only idempotent DDL such as ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT EXISTS
#   - Never DROP or TRUNCATE
#   - When init_db.sql gets a new column, add a matching entry to _MIGRATIONS

from sqlalchemy import text
from backend.dependencies import AsyncSessionLocal
from backend.core.logger import get_logger

logger = get_logger(__name__)

# Patches in the order they were added. Every statement must be idempotent (IF NOT EXISTS).
_MIGRATIONS: list[tuple[str, str]] = [
    (
        "exam_submissions.weak_points",
        "ALTER TABLE exam_submissions ADD COLUMN IF NOT EXISTS weak_points JSONB",
    ),
    (
        "exam_reviews.knowledge_tag",
        "ALTER TABLE exam_reviews ADD COLUMN IF NOT EXISTS knowledge_tag VARCHAR(128)",
    ),
    (
        "exam_submissions.weak_points_summary",
        "ALTER TABLE exam_submissions ADD COLUMN IF NOT EXISTS weak_points_summary TEXT",
    ),
    (
        "idx_exam_submissions_student_created",
        "CREATE INDEX IF NOT EXISTS idx_exam_submissions_student_created "
        "ON exam_submissions (student_id, created_at DESC)",
    ),
    (
        "idx_resume_reviews_student_created",
        "CREATE INDEX IF NOT EXISTS idx_resume_reviews_student_created "
        "ON resume_reviews (student_id, created_at DESC)",
    ),
    (
        "idx_interview_sessions_student_created",
        "CREATE INDEX IF NOT EXISTS idx_interview_sessions_student_created "
        "ON interview_sessions (student_id, created_at DESC)",
    ),
]


async def run_migrations() -> None:
    """
    Apply every schema patch at startup.
    A failed patch is logged as a warning; startup continues.
    """
    async with AsyncSessionLocal() as session:
        for desc, sql in _MIGRATIONS:
            try:
                await session.execute(text(sql))
                await session.commit()
                logger.debug("db.migration_applied", column=desc)
            except Exception as e:
                await session.rollback()
                err = str(e)
                if "already exists" not in err:
                    logger.warning("db.migration_failed", column=desc, error=err)

    logger.info("db.migrations_done", count=len(_MIGRATIONS))