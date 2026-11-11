# scripts/evaluation/tracking.py
# Collects LLM call counts, token usage (with cache hits) and latency while an agent runs.
# Cost is computed later from these counts, by eval_system_health.py (see pricing.py).
#
# Usage:
#     tracker = UsageTracker()
#     with tracker.timed():
#         await graph.ainvoke(state, config={"callbacks": [tracker]})
#     tracker.snapshot()   # {"llm_calls": 9, "input_tokens": ..., "cached_input_tokens": ..., "latency_s": ...}

import time
from contextlib import contextmanager
from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class UsageTracker(BaseCallbackHandler):
    """LangChain callback that adds up token usage across every LLM call in one run.

    Callbacks passed to graph.ainvoke propagate to the LLM calls inside the nodes.
    """

    def __init__(self):
        self._lock = Lock()            # Parallel nodes (asyncio.gather) report concurrently
        self.llm_calls = 0
        self.input_tokens = 0
        self.cached_input_tokens = 0   # Part of input_tokens served from the provider's cache
        self.output_tokens = 0
        self.latency_s = 0.0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        input_tokens, cached, output_tokens = _usage_from(response)
        with self._lock:
            self.llm_calls += 1
            self.input_tokens += input_tokens
            self.cached_input_tokens += cached
            self.output_tokens += output_tokens

    @contextmanager
    def timed(self):
        """Measure wall-clock time of the wrapped block and add it to latency_s."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.latency_s += time.perf_counter() - start

    def snapshot(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": round(self.latency_s, 2),
        }


def _usage_from(response: LLMResult) -> tuple[int, int, int]:
    """Read (input, cached input, output) token counts; providers put them in different places."""
    for generations in response.generations:
        for gen in generations:
            usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if usage:
                cached = (usage.get("input_token_details") or {}).get("cache_read") or 0
                return usage.get("input_tokens", 0), cached, usage.get("output_tokens", 0)
    token_usage = (response.llm_output or {}).get("token_usage") or {}
    cached = token_usage.get("prompt_cache_hit_tokens") or 0        # DeepSeek's own field
    return token_usage.get("prompt_tokens", 0), cached, token_usage.get("completion_tokens", 0)
