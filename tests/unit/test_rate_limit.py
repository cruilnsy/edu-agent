# tests/unit/test_rate_limit.py
# Sliding-window request limits. A fake clock replaces real waiting.

import pytest
from fastapi import HTTPException

from backend.core.rate_limit import SlidingWindowLimiter, per_user


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_allows_up_to_the_limit_then_reports_the_wait():
    clock = Clock()
    limiter = SlidingWindowLimiter("t", limit=3, window=60, clock=clock)
    assert [limiter.hit("a") for _ in range(3)] == [0, 0, 0]
    clock.now += 10
    assert limiter.hit("a") == pytest.approx(50)     # The first hit leaves the window in 50 s


def test_window_slides_rather_than_resetting():
    clock = Clock()
    limiter = SlidingWindowLimiter("t", limit=2, window=60, clock=clock)
    limiter.hit("a")
    clock.now += 30
    limiter.hit("a")
    clock.now += 31                                   # Only the first hit has left the window
    assert limiter.hit("a") == 0
    assert limiter.hit("a") > 0


def test_blocked_hits_are_not_counted():
    clock = Clock()
    limiter = SlidingWindowLimiter("t", limit=1, window=60, clock=clock)
    limiter.hit("a")
    for _ in range(5):
        clock.now += 1
        limiter.hit("a")                              # Refused, so the window does not grow
    clock.now = 1000.0 + 60
    assert limiter.hit("a") == 0


def test_keys_are_counted_separately():
    limiter = SlidingWindowLimiter("t", limit=1, window=60, clock=Clock())
    assert limiter.hit("a") == 0
    assert limiter.hit("b") == 0
    assert limiter.hit("a") > 0


def test_retry_after_does_not_record():
    limiter = SlidingWindowLimiter("t", limit=1, window=60, clock=Clock())
    assert limiter.retry_after("a") == 0
    assert limiter.retry_after("a") == 0
    limiter.record("a")
    assert limiter.retry_after("a") > 0


def test_idle_keys_are_dropped_when_there_are_many(monkeypatch):
    import backend.core.rate_limit as rate_limit
    monkeypatch.setattr(rate_limit, "_MAX_KEYS", 3)
    clock = Clock()
    limiter = SlidingWindowLimiter("t", limit=1, window=60, clock=clock)
    for key in "abc":
        limiter.hit(key)
    clock.now += 61
    limiter.hit("d")
    assert set(limiter._hits) == {"d"}


async def test_per_user_dependency_returns_429_with_retry_after():
    limiter = SlidingWindowLimiter("t", limit=1, window=60, clock=Clock())
    check = per_user(limiter)
    user = {"tenant_id": "t1", "user_id": "u1"}
    await check(current_user=user)
    with pytest.raises(HTTPException) as caught:
        await check(current_user=user)
    assert caught.value.status_code == 429
    assert caught.value.headers["Retry-After"] == "60"
    await check(current_user={"tenant_id": "t1", "user_id": "u2"})   # Another user is not affected
