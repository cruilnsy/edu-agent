# tests/unit/test_core.py
# Config, logging and the LLM factory. No services needed.

import logging

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError

from backend.config import Settings, get_settings, resolve_path, PROJECT_ROOT
from backend.core.llm_factory import LLMFactory
from backend.core.logger import get_logger


def test_settings_load_and_derive_mcp_urls():
    s = get_settings()
    assert s.db_name and s.milvus_collection
    assert s.web_search_mcp_url == f"http://127.0.0.1:{s.app_port}/mcp/web-search"


def test_empty_mcp_url_disables_server():
    assert Settings(web_search_mcp_url="").web_search_mcp_url == ""


def test_short_jwt_secret_is_rejected():
    with pytest.raises(ValidationError):
        Settings(jwt_secret_key="too-short")


def test_resolve_path_is_relative_to_project_root():
    assert resolve_path("./models/x") == str(PROJECT_ROOT / "models" / "x")


def test_structured_log_line(caplog):
    with caplog.at_level(logging.INFO):
        get_logger("demo").info("user.login", user_id="u1", role="student")
    assert "user.login | user_id='u1' role='student'" in caplog.text


def test_llm_factory_caches_and_validates():
    LLMFactory.clear_cache()
    llm = LLMFactory.get_llm("qa")                 # Builds the client only; no network call
    assert isinstance(llm, BaseChatModel)
    assert LLMFactory.get_llm("qa") is llm
    assert LLMFactory.get_llm("qa", temperature=0.5) is not llm
    with pytest.raises(ValueError):
        LLMFactory.get_llm("not_exist")

    class Demo(BaseModel):
        name: str = Field(description="Name")

    assert isinstance(LLMFactory.get_structured_llm("resume", Demo), Runnable)
    LLMFactory.clear_cache()
    assert LLMFactory._instances == {}
