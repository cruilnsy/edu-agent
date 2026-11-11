# tests/unit/test_llm_factory.py
# The factory's model arguments: model name from settings, thinking turned off, timeout and
# retries set. No network.

import pytest

from backend.core import llm_factory


def test_model_comes_from_settings_with_thinking_disabled(monkeypatch):
    settings = llm_factory.get_settings().model_copy(update={"deepseek_model_chat": "some-model"})
    monkeypatch.setattr(llm_factory, "get_settings", lambda: settings)
    kwargs = llm_factory.LLMFactory._build_model_kwargs("chat")
    assert kwargs["model"] == "some-model"
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kwargs["timeout"].read == 60.0
    assert kwargs["max_retries"] == 2                # Retried by the OpenAI client, with backoff


def test_every_agent_type_maps_to_a_known_model():
    for model_key in llm_factory._AGENT_MODEL_ROUTING.values():
        assert llm_factory._model_id(model_key)


def test_unknown_agent_type_fails_early():
    with pytest.raises(ValueError):
        llm_factory.LLMFactory.get_llm("no_such_agent")
