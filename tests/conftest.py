# tests/conftest.py
# Shared fixtures. Tests are grouped by what they need:
#   tests/unit         no services
#   tests/integration  PostgreSQL (auto-skipped when unreachable)
#   tests/llm          real LLM API (opt-in with --run-llm)

import asyncio
import os

import httpx
import pytest
from dotenv import load_dotenv

from backend.config import ENV_LOCAL_PATH, get_settings

load_dotenv(ENV_LOCAL_PATH)   # Provides SEED_*_PASSWORD for the integration tests


def pytest_addoption(parser):
    parser.addoption("--run-llm", action="store_true", help="Run tests that call the real LLM API")


def _database_reachable() -> bool:
    """Try a real connection with a short timeout, so a stopped container skips instead of hanging."""
    import asyncpg

    s = get_settings()

    async def _probe():
        conn = await asyncpg.connect(
            user=s.db_user, password=s.db_password, host=s.db_host,
            port=s.db_port, database=s.db_name, timeout=3,
        )
        await conn.close()

    try:
        asyncio.run(_probe())
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    needs_db = [i for i in items if "integration" in i.keywords]
    if needs_db and not _database_reachable():
        if os.getenv("CI"):                  # Set by GitHub Actions, which starts a database
            raise pytest.UsageError("PostgreSQL is not reachable, and CI must run the integration tests")
        skip = pytest.mark.skip(reason="PostgreSQL is not reachable")
        for item in needs_db:
            item.add_marker(skip)
    if not config.getoption("--run-llm"):
        skip = pytest.mark.skip(reason="needs --run-llm")
        for item in items:
            if "llm" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
async def _dispose_db_pool():
    """Each async test gets its own event loop, but the engine's pool is global.

    Pooled connections belong to the loop that opened them, so drop them after every test.
    """
    yield
    from backend.dependencies import engine
    await engine.dispose()


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Request limits count per process; start every test from zero."""
    from backend.core.rate_limit import ALL_LIMITERS
    yield
    for limiter in ALL_LIMITERS:
        limiter.reset()


@pytest.fixture
async def client():
    """HTTP client wired straight into the app (no server, no lifespan startup)."""
    from backend.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _login(client: httpx.AsyncClient, username: str, env_name: str) -> dict:
    resp = await client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": os.environ[env_name]},
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
async def student_headers(client):
    return await _login(client, "student01", "SEED_STUDENT_PASSWORD")


@pytest.fixture
async def teacher_headers(client):
    return await _login(client, "teacher01", "SEED_TEACHER_PASSWORD")
