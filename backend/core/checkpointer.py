# backend/core/checkpointer.py
# Postgres checkpointer for graphs whose state must outlive one request, so an interview in
# progress survives a backend restart or a deploy. Tests and evals keep using MemorySaver.

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


def _conninfo() -> str:
    """libpq connection string for the app database. Contains the password, so never log it."""
    s = get_settings()
    return make_conninfo(host=s.db_host, port=s.db_port, dbname=s.db_name, user=s.db_user, password=s.db_password,
                         application_name="eduagent-checkpointer")


@asynccontextmanager
async def postgres_checkpointer(max_size: int = 10) -> AsyncIterator[AsyncPostgresSaver]:
    """Open a connection pool, create the checkpoint tables if needed, and close the pool on exit.

    The settings are the ones AsyncPostgresSaver requires: autocommit, rows as dicts, and no
    prepared statements (which break behind connection poolers such as PgBouncer)."""
    pool = AsyncConnectionPool(_conninfo(), min_size=1, max_size=max_size, open=False,
                               kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None})
    await pool.open(wait=True, timeout=10)
    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()                  # Idempotent: creates or migrates the checkpoint tables
        logger.info("checkpointer.postgres_ready")
        yield saver
    finally:
        await pool.close()
