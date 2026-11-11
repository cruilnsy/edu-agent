# tests/unit/test_resume_fallback.py
# When an LLM call fails, each node must still return usable output and report fallback_used,
# so the review never claims a full result it did not produce.

import pytest

from backend.agents.resume import nodes


class _FailingLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        raise RuntimeError("LLM unavailable")


@pytest.fixture
def failing_llm(monkeypatch):
    llm = _FailingLLM()
    monkeypatch.setattr(nodes, "get_structured_llm", lambda *args, **kwargs: llm)
    monkeypatch.setattr(nodes, "get_llm", lambda *args, **kwargs: llm)

    async def no_sleep(_seconds):
        pass
    monkeypatch.setattr(nodes.asyncio, "sleep", no_sleep)   # Skip the 1 s retry wait
    return llm


DIMENSION_SCORES = [
    {"key": "expression", "dimension": "Writing", "score": 60, "weight": 1.0,
     "issues": ["Weak verbs"], "suggestions": []},
]


async def test_extract_structured_retries_then_falls_back(failing_llm):
    result = await nodes.extract_structured_node({"raw_text": "resume"})
    assert failing_llm.calls == 2                       # One retry before giving up
    assert result["fallback_used"] is True
    assert result["structured"]["name"] == "(extraction failed)"


async def test_six_dimensions_reports_fallback(failing_llm):
    result = await nodes.run_six_dimensions_node({"raw_text": "resume", "structured": {}})
    assert result["fallback_used"] is True
    assert len(result["dimension_scores"]) == len(nodes.SIX_DIMENSIONS)


async def test_diagnose_falls_back_to_dimension_issues(failing_llm):
    result = await nodes.diagnose_issues_node(
        {"raw_text": "resume", "structured": {}, "dimension_scores": DIMENSION_SCORES})
    assert result["fallback_used"] is True
    assert [i["description"] for i in result["issues"]] == ["Weak verbs"]


async def test_generate_summary_reports_fallback(failing_llm):
    result = await nodes.generate_summary_node(
        {"structured": {}, "dimension_scores": DIMENSION_SCORES, "issues": [], "weighted_score": 60})
    assert result["fallback_used"] is True
    assert result["summary"]["overall_comment"]


class _FlakyLLM:
    """Fails the first `fail_times` calls, then returns `result`."""

    def __init__(self, result, fail_times=1):
        self.result, self.fail_times, self.calls = result, fail_times, 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient error")
        return self.result


def _use(monkeypatch, structured_llm, plain_llm=None):
    monkeypatch.setattr(nodes, "get_structured_llm", lambda *args, **kwargs: structured_llm)
    monkeypatch.setattr(nodes, "get_llm", lambda *args, **kwargs: plain_llm or structured_llm)

    async def no_sleep(_seconds):
        pass
    monkeypatch.setattr(nodes.asyncio, "sleep", no_sleep)


async def test_successful_retry_is_not_a_fallback(monkeypatch):
    _use(monkeypatch, _FlakyLLM(nodes.ResumeStructured(name="Alex Doe")))
    result = await nodes.extract_structured_node({"raw_text": "resume"})
    assert result["structured"]["name"] == "Alex Doe"
    assert not result.get("fallback_used")


async def test_six_dimensions_success_path(monkeypatch):
    score = nodes.DimensionScore(score=80, issues=[], suggestions=[])
    _use(monkeypatch, _FlakyLLM(score, fail_times=0))
    result = await nodes.run_six_dimensions_node({"raw_text": "resume", "structured": {}})
    assert result["weighted_score"] == 80
    assert not result.get("fallback_used")


async def test_earlier_fallback_is_not_cleared(monkeypatch):
    score = nodes.DimensionScore(score=80, issues=[], suggestions=[])
    _use(monkeypatch, _FlakyLLM(score, fail_times=0))
    result = await nodes.run_six_dimensions_node(
        {"raw_text": "resume", "structured": {}, "fallback_used": True})
    assert result.get("fallback_used", True) is True    # Either keep True or leave the key out


class _Reply:
    text = "Analysis."


async def test_diagnose_reports_fallback_when_only_the_issue_list_fails(monkeypatch):
    _use(monkeypatch, _FailingLLM(), plain_llm=_FlakyLLM(_Reply(), fail_times=0))
    result = await nodes.diagnose_issues_node(
        {"raw_text": "resume", "structured": {}, "dimension_scores": DIMENSION_SCORES})
    assert result["fallback_used"] is True
