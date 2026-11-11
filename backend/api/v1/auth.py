# backend/api/v1/auth.py
# Auth endpoints: /login issues a token, /me checks it.

import asyncio
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.rate_limit import (
    LOGIN_FAILURES_PER_ACCOUNT, LOGIN_PER_ADDRESS, client_address, too_many_requests,
)
from backend.dependencies import get_current_user, get_db

# ── Compatibility patch: passlib 1.7.4 reads bcrypt.__about__.__version__, which bcrypt>=4 removed ──
import bcrypt as _bcrypt_mod, types as _types
if not hasattr(_bcrypt_mod, "__about__"):
    _bcrypt_mod.__about__ = _types.SimpleNamespace(__version__=getattr(_bcrypt_mod, "__version__", "4.x"))

router = APIRouter()
logger = get_logger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Checked when the user does not exist, so that case takes as long as a wrong password.
# Otherwise response time would reveal which usernames exist.
_DUMMY_HASH = pwd_context.hash("not-a-real-password")

_INVALID_LOGIN = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Incorrect username or password",
)


class LoginRequest(BaseModel):
    username: str = Field(..., max_length=255, description="Username or email")
    password: str = Field(..., max_length=128, description="Password")


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int                 # Seconds
    role:         str
    user_id:      str


def _create_access_token(data: dict, expires_minutes: int) -> str:
    """Sign the identity claims plus an expiry into a JWT."""
    settings = get_settings()
    payload = data.copy()                                            # Do not mutate the caller's dict
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)   # Standard expiry claim
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


@router.post("/login", response_model=TokenResponse)
async def login(
    req: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Log in with a username or email and return a JWT access token."""
    settings = get_settings()

    # Checked before the password, so a blocked caller learns nothing about it
    account = req.username.strip().lower()
    wait = LOGIN_PER_ADDRESS.hit(client_address(request)) or LOGIN_FAILURES_PER_ACCOUNT.retry_after(account)
    if wait:
        logger.warning("auth.login_rate_limited")
        raise too_many_requests(wait, "Too many login attempts; please wait and try again")

    # Bound parameter (:val), never string formatting, so the input cannot inject SQL
    result = await db.execute(
        text(
            "SELECT id, password_hash, role, tenant_id, is_active "
            "FROM users WHERE username = :val OR email = :val LIMIT 1"
        ),
        {"val": req.username},
    )
    row = result.fetchone()

    # bcrypt takes ~100 ms of CPU, so run it in a thread to keep the event loop free
    loop = asyncio.get_running_loop()
    password_ok = await loop.run_in_executor(
        None, pwd_context.verify, req.password, row.password_hash if row else _DUMMY_HASH
    )
    if not row or not password_ok:
        LOGIN_FAILURES_PER_ACCOUNT.record(account)
        raise _INVALID_LOGIN

    # Checked only after the password, so a disabled account is not revealed to someone who lacks it
    if not row.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled. Contact an admin.")

    token = _create_access_token(
        data={"sub": str(row.id), "role": row.role, "tenant_id": row.tenant_id},
        expires_minutes=settings.jwt_access_token_expire_minutes,
    )
    logger.info("auth.login_success", user_id=str(row.id), role=row.role)

    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
        role=row.role,
        user_id=str(row.id),
    )


@router.get("/me")
async def get_me(
    current_user: dict = Depends(get_current_user),
):
    """Return the current user; useful for checking that a token is valid."""
    return current_user


# ── Self-check: password hashing and token round trip, no database needed ──
if __name__ == "__main__":
    import secrets
    sample = secrets.token_urlsafe(16)   # Random sample, so no password lives in code
    h = pwd_context.hash(sample)
    print("correct password verifies:", pwd_context.verify(sample, h))
    print("wrong password verifies:  ", pwd_context.verify("wrong", h))

    s = get_settings()
    tk = _create_access_token({"sub": "u-1", "role": "student", "tenant_id": "tenant_default"}, 10)
    decoded = jwt.decode(tk, s.jwt_secret_key, algorithms=[s.jwt_algorithm])
    print("decoded sub:", decoded["sub"], "| role:", decoded["role"])
