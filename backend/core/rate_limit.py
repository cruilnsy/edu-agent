# backend/core/rate_limit.py
# Request limits per user (LLM endpoints) and per client address (login).
#
# Counters live in this process's memory: enough for the single uvicorn worker this app runs.
# With several workers or hosts each would count on its own, so a shared store (Redis) would
# be needed then.

import math
import time
from collections import deque
from typing import Callable

from fastapi import Depends, HTTPException, Request, status

from backend.dependencies import get_current_user

_MAX_KEYS = 10_000          # Above this, keys whose window has emptied are dropped


class SlidingWindowLimiter:
    """At most `limit` hits per key in any `window` seconds."""

    def __init__(self, name: str, limit: int, window: float, clock: Callable[[], float] = time.monotonic):
        self.name, self.limit, self.window, self._clock = name, limit, window, clock
        self._hits: dict[str, deque[float]] = {}

    def _recent(self, key: str, now: float) -> deque[float]:
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        return hits

    def retry_after(self, key: str) -> float:
        """Seconds until `key` may make another hit; 0 if it may now. Records nothing."""
        now = self._clock()
        hits = self._recent(key, now)
        return hits[0] + self.window - now if len(hits) >= self.limit else 0.0

    def record(self, key: str) -> None:
        now = self._clock()
        self._recent(key, now).append(now)
        if len(self._hits) > _MAX_KEYS:
            self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > now - self.window}

    def hit(self, key: str) -> float:
        """Record a hit unless the key is over its limit. Returns the wait in seconds, 0 if allowed."""
        wait = self.retry_after(key)
        if not wait:
            self.record(key)
        return wait

    def reset(self) -> None:
        self._hits.clear()


def too_many_requests(wait: float, message: str = "Too many requests; please wait and try again") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=message,
        headers={"Retry-After": str(max(1, math.ceil(wait)))},
    )


# ── Limits ──
# Chat turns: a person typing stays far below this; a script looping on the API does not.
CHAT = SlidingWindowLimiter("chat", limit=20, window=60)
# Each resume review makes about ten LLM calls, and each exam submission one per question.
RESUME_UPLOAD = SlidingWindowLimiter("resume_upload", limit=10, window=3600)
EXAM_SUBMIT = SlidingWindowLimiter("exam_submit", limit=20, window=3600)
# Login: attempts per client address, and failed attempts per account name. The second stops
# password guessing spread over many addresses; it can lock an account for 15 minutes.
LOGIN_PER_ADDRESS = SlidingWindowLimiter("login_per_address", limit=10, window=60)
LOGIN_FAILURES_PER_ACCOUNT = SlidingWindowLimiter("login_failures_per_account", limit=10, window=900)

ALL_LIMITERS = (CHAT, RESUME_UPLOAD, EXAM_SUBMIT, LOGIN_PER_ADDRESS, LOGIN_FAILURES_PER_ACCOUNT)


def per_user(limiter: SlidingWindowLimiter) -> Callable:
    """Route dependency: 429 once the signed-in user is over `limiter`."""
    async def check(current_user: dict = Depends(get_current_user)) -> None:
        wait = limiter.hit(f"{current_user['tenant_id']}:{current_user['user_id']}")
        if wait:
            raise too_many_requests(wait)
    return check


def client_address(request: Request) -> str:
    """The caller's address. Behind the Docker nginx, uvicorn's --proxy-headers sets it from
    the X-Forwarded-For header that nginx overwrites with the real peer address."""
    return request.client.host if request.client else "unknown"


limit_chat = per_user(CHAT)
limit_resume_upload = per_user(RESUME_UPLOAD)
limit_exam_submit = per_user(EXAM_SUBMIT)
