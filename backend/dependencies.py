# backend/dependencies.py
# FastAPI dependencies: get_db (database session), get_current_user (JWT auth),
# require_teacher (role check).

import uuid
from typing import AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.config import get_settings

settings = get_settings()

# ── PostgreSQL async connection pool ──
engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    echo=False,                      # True would log every SQL statement
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a session that commits on success, rolls back on error, and always closes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── JWT auth ──
bearer_scheme = HTTPBearer()   # Parses "Authorization: Bearer <token>"; rejects missing headers


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """Validate the JWT and return {"user_id", "role", "tenant_id"}. Raises 401 if invalid.

    The user is looked up on every request, so a deleted or disabled user loses access at once,
    and a validly signed token for a user this database does not know (e.g. one issued by another
    deployment sharing the secret) gets 401 instead of failing later on a foreign key.
    Role and tenant come from the database, not from the token claims.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Raises JWTError for a bad signature or an expired token
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],   # Pinned list: blocks algorithm-switching attacks
        )
        user_id = str(uuid.UUID(payload.get("sub") or ""))   # ValueError for a missing or malformed id
    except (JWTError, ValueError):
        raise credentials_exception

    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            text("SELECT role, tenant_id, is_active FROM users WHERE id = :id"), {"id": user_id},
        )).first()
    if row is None or not row.is_active:
        raise credentials_exception

    return {"user_id": user_id, "role": row.role, "tenant_id": row.tenant_id}


async def require_teacher(
    current_user: dict = Depends(get_current_user),   # Auth runs first, so a bad token is still 401
) -> dict:
    """Allow only teachers and admins; everyone else gets 403."""
    if current_user["role"] not in ("teacher", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only teachers or admins can perform this action",
        )
    return current_user
