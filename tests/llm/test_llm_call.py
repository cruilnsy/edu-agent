# tests/llm/test_llm_call.py
# Real LLM round trip. Costs tokens, so it runs only with: pytest --run-llm

import pytest
from langchain_core.messages import HumanMessage

from backend.core.llm_factory import LLMFactory, get_llm

pytestmark = pytest.mark.llm


async def test_llm_answers():
    LLMFactory.clear_cache()   # The shared async HTTP client must not carry state across loops
    try:
        reply = await get_llm("qa").ainvoke([HumanMessage(content="Reply with the single word: pong")])
        assert "pong" in reply.content.lower()
    finally:
        LLMFactory.clear_cache()
