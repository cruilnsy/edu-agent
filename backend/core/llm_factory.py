# backend/core/llm_factory.py
# LLM factory: one place that creates chat models, routed by agent type.
# Rule: every agent gets its model from here; never call init_chat_model directly.

from typing import Any, Type

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from backend.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)

# ── Timeouts and retries for every LLM call ──
# A normal call answers within seconds, so one still waiting after 60 s is treated as failed. The OpenAI client then retries it:
# connection errors, timeouts, 408, 409, 429 and 5xx, with exponential backoff that honours
# the API's Retry-After header. A stream is retried only if it fails before the first token.
LLM_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
LLM_MAX_RETRIES = 2                                   # 3 attempts in total

# ── Custom httpx clients that ignore system proxies ──
# httpx picks up system proxy settings and HTTPS_PROXY by default, which can break the TLS
# handshake with the LLM API. trust_env=False ignores them.
_HTTP_ASYNC_CLIENT = httpx.AsyncClient(trust_env=False, timeout=LLM_TIMEOUT)   # ainvoke / astream
_HTTP_SYNC_CLIENT = httpx.Client(trust_env=False, timeout=LLM_TIMEOUT)         # invoke

# ── Agent type -> model key. To switch the model for one use case, change one line here. ──
_AGENT_MODEL_ROUTING: dict[str, str] = {
    "qa":               "chat",
    "exam_subjective":  "chat",   # Short-answer grading
    "exam_code":        "chat",   # Code grading
    "resume":           "chat",
    "interview":        "chat",
    "intent":           "chat",   # Intent classification fallback
    "summarize":        "chat",   # Conversation summarization
}


def _model_id(model_key: str) -> str:
    """Model key -> model name the API accepts (from DEEPSEEK_MODEL_CHAT)."""
    return {"chat": get_settings().deepseek_model_chat}[model_key]


# DeepSeek V4 models think before answering by default. Every use here (grading, routing,
# interview replies) was tuned without thinking, and thinking adds latency and tokens, so it is
# turned off. The retired "deepseek-chat" name was the same model with thinking off.
_NO_THINKING = {"thinking": {"type": "disabled"}}


class LLMFactory:
    """The only entry point for getting a model. All methods are class methods.

    Usage:
        llm = LLMFactory.get_llm("qa")
        structured = LLMFactory.get_structured_llm("resume", SomeSchema)
        response = await llm.ainvoke(messages)
    """

    _instances: dict[str, BaseChatModel] = {}   # Cache shared by the whole class

    @classmethod
    def _build_model_kwargs(cls, model_key: str) -> dict[str, Any]:
        """Arguments for init_chat_model. DeepSeek is called through its OpenAI-compatible API."""
        settings = get_settings()
        return {
            "model": _model_id(model_key),
            "model_provider": "openai",
            "temperature": 0,                      # Grading and scoring need stable output
            "api_key": settings.deepseek_api_key,
            "base_url": settings.deepseek_base_url,
            "timeout": LLM_TIMEOUT,
            "max_retries": LLM_MAX_RETRIES,
            "extra_body": _NO_THINKING,
            "http_async_client": _HTTP_ASYNC_CLIENT,
            "http_client": _HTTP_SYNC_CLIENT,
        }

    @classmethod
    def get_llm(
        cls,
        agent_type: str,
        temperature: float = 0,      # 0.3-0.7 for conversation, 0 for scoring
        streaming: bool = False,
    ) -> BaseChatModel:
        """Return a cached model for agent_type.

        Each (model, temperature, streaming) combination is created once and reused.
        """
        if agent_type not in _AGENT_MODEL_ROUTING:   # Fail early on typos
            raise ValueError(
                f"Unknown agent_type: '{agent_type}'. "
                f"Available: {list(_AGENT_MODEL_ROUTING.keys())}"
            )

        model_key = _AGENT_MODEL_ROUTING[agent_type]
        cache_key = f"{model_key}_{temperature}_{streaming}"
        if cache_key not in cls._instances:
            kwargs = cls._build_model_kwargs(model_key)
            kwargs["temperature"] = temperature
            kwargs["streaming"] = streaming
            cls._instances[cache_key] = init_chat_model(**kwargs)
            logger.info(
                "llm_factory.model_initialized",
                agent_type=agent_type, model_key=model_key,
                temperature=temperature, streaming=streaming,
            )

        return cls._instances[cache_key]

    @classmethod
    def get_structured_llm(
        cls,
        agent_type: str,
        output_schema: Type[BaseModel],
        temperature: float = 0,
    ) -> Runnable:
        """Return a model bound to output_schema: ainvoke returns an output_schema instance, not text."""
        llm = cls.get_llm(agent_type, temperature=temperature)
        # DeepSeek needs method="function_calling" for structured output
        return llm.with_structured_output(output_schema, method="function_calling")

    @classmethod
    def clear_cache(cls) -> None:
        """Drop all cached models (used at shutdown and in tests)."""
        cls._instances.clear()
        logger.info("llm_factory.cache_cleared")


# ── Module-level shortcuts, the preferred style in agent code ──

def get_llm(agent_type: str, temperature: float = 0, streaming: bool = False) -> BaseChatModel:
    return LLMFactory.get_llm(agent_type, temperature=temperature, streaming=streaming)


def get_structured_llm(agent_type: str, output_schema: Type[BaseModel]) -> Runnable:
    return LLMFactory.get_structured_llm(agent_type, output_schema)
