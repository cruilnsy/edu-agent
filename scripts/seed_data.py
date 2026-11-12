# scripts/seed_data.py
# Run: python scripts/seed_data.py
# Seeds the local development accounts. Passwords come from .env.local, never from code.

import asyncio
import os
import sys
import uuid
from pathlib import Path

import asyncpg                       # Plain async driver; this script does not need SQLAlchemy
from dotenv import load_dotenv
from passlib.context import CryptContext

# Compatibility patch, same as auth.py: lets passlib read the bcrypt version
import bcrypt as _b, types as _t
if not hasattr(_b, "__about__"):
    _b.__about__ = _t.SimpleNamespace(__version__=getattr(_b, "__version__", "4.x"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env.local")   # Works from any working directory

MIN_PASSWORD_LENGTH = 12
TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "tenant_default")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# (username, role, env var holding the password); emails use the reserved .local domain
ACCOUNTS = [
    ("admin",     "admin",   "SEED_ADMIN_PASSWORD"),
    ("teacher01", "teacher", "SEED_TEACHER_PASSWORD"),
    ("student01", "student", "SEED_STUDENT_PASSWORD"),
    ("student02", "student", "SEED_STUDENT_PASSWORD"),
]


def require_env(name: str) -> str:
    """Return a required environment variable, or exit with a clear message."""
    value = os.getenv(name)
    if not value:
        sys.exit(f"Missing {name}. Set it in .env.local (see .env.example).")
    return value


def load_passwords() -> dict[str, str]:
    """Read every seed password up front so the script fails before touching the database."""
    passwords = {}
    for _, _, env_name in ACCOUNTS:
        value = require_env(env_name)
        if len(value) < MIN_PASSWORD_LENGTH:
            sys.exit(f"{env_name} must be at least {MIN_PASSWORD_LENGTH} characters.")
        if value.lower().startswith("change-me"):      # An unedited placeholder is a known password
            sys.exit(f"{env_name} is still a placeholder; choose a real password.")
        passwords[env_name] = value
    return passwords


async def seed_users() -> None:
    """Create the dev accounts, or reset their passwords to the values in .env.local."""
    passwords = load_passwords()
    conn = await asyncpg.connect(
        user=require_env("DB_USER"),
        password=require_env("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        database=require_env("DB_NAME"),
    )
    try:
        for username, role, env_name in ACCOUNTS:
            await conn.execute(
                """
                INSERT INTO users (id, tenant_id, username, email, password_hash, role)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (tenant_id, email) DO UPDATE
                    SET password_hash = EXCLUDED.password_hash
                """,
                str(uuid.uuid4()), TENANT_ID, username, f"{username}@eduagent.local",
                pwd_context.hash(passwords[env_name]),   # Store only the hash
                role,
            )
        print(f"Seeded {len(ACCOUNTS)} accounts into '{os.getenv('DB_NAME')}':")
        for username, role, env_name in ACCOUNTS:
            print(f"  {username:<10} {role:<8} password from {env_name}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(seed_users())
