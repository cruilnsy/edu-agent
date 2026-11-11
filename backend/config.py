# backend/config.py
# Single source of configuration: every setting is read from .env.local and shared through get_settings().

import os
from functools import lru_cache              # Caches the Settings object so .env.local is read only once
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict   # Reads env vars / .env files with type checks

# Project root (two levels up). Absolute, so it does not depend on the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_LOCAL_PATH = PROJECT_ROOT / ".env.local"

MIN_JWT_SECRET_LENGTH = 32


def resolve_path(path: str) -> str:
    """Resolve a configured path (e.g. ./models/embedding/bge-m3) against the project root.

    Absolute paths are returned unchanged. Only normalizes the path; symlinks are not followed.
    """
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


class Settings(BaseSettings):
    """Each field maps to one key in .env.local (case-insensitive).

    Fields without a default are required: startup fails if they are missing.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_LOCAL_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,        # DB_HOST in the file maps to db_host
        extra="ignore",              # Keys used only by scripts or Docker are ignored
    )

    # ── PostgreSQL ──
    db_host: str = "localhost"
    db_port: int = 5433              # 5433 avoids clashing with a local PostgreSQL on 5432
    db_name: str = "eduagent_en"
    db_user: str
    db_password: str

    @property
    def database_url(self) -> str:
        """SQLAlchemy async URL. Contains the password, so never log it."""
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # ── Milvus ──
    milvus_host: str = "localhost"
    milvus_port: int = 19531         # 19531 avoids clashing with a local Milvus on 19530
    milvus_collection: str = "knowledge_domain_en"

    # ── LLM (DeepSeek, OpenAI-compatible API) ──
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model_chat: str = "deepseek-v4-flash"   # Sent with thinking disabled; see llm_factory.py

    # ── Local model weights (relative to the project root) ──
    reranker_model_path: str = "./models/reranker/bge-reranker-v2-m3"
    classifier_model_path: str = "./models/classifier/all-MiniLM-L6-v2"   # Base model for intent classification
    finetuned_classifier_path: str = "./models/classifier/finetune-en"    # Output of QueryClassifier.train()
    bge_m3_model_path: str = "./models/embedding/bge-m3"

    # ── JWT ──
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 10080   # 7 days

    # ── App ──
    app_env: str = "local"
    app_debug: bool = False
    app_host: str = "127.0.0.1"     # Loopback only; set 0.0.0.0 explicitly to expose on the network
    app_port: int = 8010
    log_level: str = "INFO"
    default_tenant_id: str = "tenant_default"

    # ── MCP servers (mounted on this app) ──
    # None means "derive from app_port"; an explicit empty string disables the server.
    kb_mcp_server_url: str | None = None
    web_search_mcp_url: str | None = None

    # ── Web search (Tavily is optional; empty falls back to free DuckDuckGo) ──
    tavily_api_key: str = ""

    @field_validator("jwt_secret_key")
    @classmethod
    def _check_jwt_secret(cls, value: str) -> str:
        """Reject short signing keys, which make tokens easy to forge."""
        if len(value) < MIN_JWT_SECRET_LENGTH:
            raise ValueError(f"JWT_SECRET_KEY must be at least {MIN_JWT_SECRET_LENGTH} characters")
        return value

    @model_validator(mode="after")
    def _derive_mcp_urls(self) -> "Settings":
        """Point the MCP URLs at this app's own port unless they are configured explicitly."""
        base = f"http://127.0.0.1:{self.app_port}"
        if self.kb_mcp_server_url is None:
            self.kb_mcp_server_url = f"{base}/mcp/kb"
        if self.web_search_mcp_url is None:
            self.web_search_mcp_url = f"{base}/mcp/web-search"
        return self


@lru_cache()
def get_settings() -> Settings:
    """Return the shared Settings instance; created on the first call, cached afterwards."""
    return Settings()
