# tests/integration/test_auth_and_db.py
# Database session and login behaviour against the real PostgreSQL.

import os
import uuid

import pytest
from sqlalchemy import text

from backend.api.v1.auth import _create_access_token
from backend.dependencies import AsyncSessionLocal, get_db

pytestmark = pytest.mark.integration

LOGIN = "/api/v1/auth/login"


async def test_get_db_yields_a_working_session():
    gen = get_db()
    session = await anext(gen)
    assert (await session.execute(text("SELECT 1"))).scalar() == 1
    await gen.aclose()


async def test_login_by_username_and_email(client):
    password = os.environ["SEED_STUDENT_PASSWORD"]
    for username in ("student01", "student01@eduagent.local"):
        resp = await client.post(LOGIN, json={"username": username, "password": password})
        assert resp.status_code == 200
        assert resp.json()["role"] == "student"


async def test_wrong_password_and_unknown_user_look_the_same(client):
    wrong = await client.post(LOGIN, json={"username": "student01", "password": "wrong-password"})
    unknown = await client.post(LOGIN, json={"username": "nobody", "password": "x"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


async def test_disabled_account_is_revealed_only_with_the_right_password(client):
    async def set_active(value: bool):
        async with AsyncSessionLocal() as db:
            await db.execute(text("UPDATE users SET is_active = :v WHERE username = 'student02'"), {"v": value})
            await db.commit()

    await set_active(False)
    try:
        wrong = await client.post(LOGIN, json={"username": "student02", "password": "wrong-password"})
        right = await client.post(LOGIN, json={"username": "student02",
                                               "password": os.environ["SEED_STUDENT_PASSWORD"]})
    finally:
        await set_active(True)
    assert wrong.status_code == 401
    assert right.status_code == 403


async def test_signed_token_for_unknown_user_is_rejected(client):
    """A token signed with our secret but for a user not in this DB (e.g. another deployment)."""
    token = _create_access_token({"sub": str(uuid.uuid4()), "role": "student",
                                  "tenant_id": "tenant_default"}, expires_minutes=5)
    resp = await client.get("/api/v1/resume/reviews", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


async def test_disabling_a_user_revokes_existing_tokens(client):
    password = os.environ["SEED_STUDENT_PASSWORD"]
    login = await client.post(LOGIN, json={"username": "student02", "password": password})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    async with AsyncSessionLocal() as db:
        await db.execute(text("UPDATE users SET is_active = false WHERE username = 'student02'"))
        await db.commit()
    try:
        resp = await client.get("/api/v1/auth/me", headers=headers)
    finally:
        async with AsyncSessionLocal() as db:
            await db.execute(text("UPDATE users SET is_active = true WHERE username = 'student02'"))
            await db.commit()
    assert resp.status_code == 401


async def test_me_and_bad_token(client, student_headers):
    me = await client.get("/api/v1/auth/me", headers=student_headers)
    assert me.status_code == 200 and set(me.json()) == {"user_id", "role", "tenant_id"}
    bad = await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer bad.token.here"})
    assert bad.status_code == 401


async def test_repeated_failures_lock_the_account_but_not_others(client):
    from backend.core.rate_limit import LOGIN_FAILURES_PER_ACCOUNT, LOGIN_PER_ADDRESS
    address_limit, LOGIN_PER_ADDRESS.limit = LOGIN_PER_ADDRESS.limit, 100   # Isolate the per-account limit
    try:
        for _ in range(LOGIN_FAILURES_PER_ACCOUNT.limit):
            resp = await client.post(LOGIN, json={"username": "Student01", "password": "wrong-password"})
            assert resp.status_code == 401
        # Even the right password is refused now, and the check ignores case
        right = await client.post(LOGIN, json={"username": "student01",
                                               "password": os.environ["SEED_STUDENT_PASSWORD"]})
        other = await client.post(LOGIN, json={"username": "student02",
                                               "password": os.environ["SEED_STUDENT_PASSWORD"]})
    finally:
        LOGIN_PER_ADDRESS.limit = address_limit
    assert right.status_code == 429 and int(right.headers["Retry-After"]) > 0
    assert other.status_code == 200


async def test_too_many_attempts_from_one_address(client):
    from backend.core.rate_limit import LOGIN_PER_ADDRESS
    codes = [(await client.post(LOGIN, json={"username": f"nobody{i}", "password": "x"})).status_code
             for i in range(LOGIN_PER_ADDRESS.limit + 1)]
    assert codes[:-1] == [401] * LOGIN_PER_ADDRESS.limit
    assert codes[-1] == 429
