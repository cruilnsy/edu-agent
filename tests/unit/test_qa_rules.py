# tests/unit/test_qa_rules.py
# QA agent pieces that need no model or network: prompt placeholders, layer-0 rules, web-search
# requests, graph routing, memory helpers, and the low-confidence notice.

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.agents.qa import graph, nodes, prompts
from backend.core import memory


def test_prompts_format_with_the_arguments_nodes_pass():
    filled = [
        prompts.RAG_STRATEGY_PROMPT.format(query="q"),
        prompts.HYDE_PROMPT.format(history="h", query="q"),
        prompts.MULTI_QUERY_REWRITE_PROMPT.format(last_answer="a", query="q"),
        prompts.RAG_ANSWER_PROMPT.format(context="c", query="q"),
        prompts.DIRECT_ANSWER_PROMPT.format(query="q"),
        prompts.GENERAL_ANSWER_PROMPT.format(query="q", history="h", current_time="t", web_context=""),
    ]
    for text in filled:
        assert "{" not in text and "}" not in text


@pytest.mark.parametrize("raw, question, web", [
    ("What is Spring AOP? If you don't know, search the web", "What is Spring AOP?", True),
    ("latest React version, please look it up", "latest React version", True),
    ("Can you google it: newest Python release", "Can you google it: newest Python release", True),
    ("How do I search the docs for a function?", "How do I search the docs for a function?", False),
    ("search the web", "search the web", True),                 # Nothing left: keep the raw text
])
def test_web_search_request_is_detected_and_removed(raw, question, web):
    assert nodes._extract_query_and_web_flag(raw) == (question, web)


@pytest.mark.parametrize("query, expected", [
    ("hi", True), ("Thanks!", True), ("What time is it?", True), ("who are you", True),
    ("this is hard", False),                                     # "hi" inside "this"
    ("What is a React hook?", False),
    ("How does the weather API example handle errors?", True),  # Known limitation: "weather" is a GENERAL phrase
])
def test_general_rules(query, expected):
    assert nodes._rule_classify_general(query) is expected


def test_specialized_rules():
    assert nodes._rule_classify_specialized("What does the knowledge base say about CORS?")
    assert nodes._rule_classify_specialized("In the docs, how is a reducer defined?")
    assert not nodes._rule_classify_specialized("What is a reducer?")


@pytest.mark.parametrize("query, strategy", [
    ("I don't get it", "VAGUE"),
    ("explain that again?", "VAGUE"),
    ("Can you explain how asyncio.gather handles exceptions from tasks?", "PRECISE"),  # Long: not vague
    ("overview of React hooks", "BROAD"),
    ("difference between interface and type", "BROAD"),
    ("What does asyncio.gather return?", "PRECISE"),
])
def test_fast_rag_strategy(query, strategy):
    assert nodes._fast_rag_strategy(query) == strategy


async def test_classify_query_routes_small_talk_without_the_classifier(monkeypatch):
    def no_classifier():
        raise AssertionError("the classifier must not run for small talk")
    monkeypatch.setattr(nodes, "get_query_classifier", no_classifier)
    state = {"messages": [HumanMessage(content="hello!")], "student_id": "s", "session_id": "x"}
    result = await nodes.classify_query_node(state)
    assert result["query_type"] == "GENERAL"


@pytest.mark.parametrize("state, route", [
    ({"query_type": "GENERAL", "enable_web_search": True}, "GENERAL_WEB"),
    ({"query_type": "general"}, "GENERAL"),
    ({"query_type": "BROAD"}, "BROAD"),
])
def test_route_by_query_type(state, route):
    assert graph._route_by_query_type(state) == route


@pytest.mark.parametrize("state, route", [
    ({"is_high_confidence": True, "enable_web_search": True}, "high"),
    ({"is_high_confidence": False, "enable_web_search": True}, "low_web"),
    ({"is_high_confidence": False}, "low_direct"),
])
def test_route_by_confidence(state, route):
    assert graph._route_by_confidence(state) == route


def test_memory_window_keeps_system_and_last_turns():
    messages = [SystemMessage(content="system")]
    for i in range(1, 7):
        messages += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]
    trimmed = memory.trim_messages_to_window(messages, window_size=3)
    assert isinstance(trimmed[0], SystemMessage)
    assert [m.content for m in trimmed[1:]] == ["q4", "a4", "q5", "a5", "q6", "a6"]


@pytest.mark.parametrize("turns, expected", [(9, False), (10, True), (11, False), (20, True)])
def test_summary_triggers_every_ten_turns(turns, expected):
    messages = [HumanMessage(content="q"), AIMessage(content="a")] * turns
    assert memory.should_trigger_summary(messages, threshold=10) is expected


def test_thread_ids_and_savers():
    assert memory.build_thread_id("s1", "x1") == "student_s1_session_x1"
    assert memory.get_memory_saver("qa") is memory.get_memory_saver("qa")
    assert memory.get_memory_saver("qa") is not memory.get_memory_saver("exam")


async def test_direct_answer_without_web_results_carries_a_notice(monkeypatch):
    class FakeLLM:
        async def ainvoke(self, messages):
            return AIMessage(content="An answer.")
    monkeypatch.setattr(nodes, "get_llm", lambda *args, **kwargs: FakeLLM())
    state = {"original_query": "q", "messages": [HumanMessage(content="q")], "confidence": 0.2}
    result = await nodes.generate_direct_node(state)
    assert result["answer_mode"] == "llm_direct"
    assert result["answer"].startswith("An answer.") and "knowledge base" in result["answer"]


def _fake_retrieve(monkeypatch, scores):
    from backend.core import reranker
    docs = [reranker.RankedDocument(content=f"chunk {i}", score=s, original_index=i, metadata={})
            for i, s in enumerate(scores)]
    monkeypatch.setattr(reranker, "retrieve", lambda *args, **kwargs: (docs, scores[0] if scores else 0.0))


RETRIEVE_STATE = {"query_type": "PRECISE", "tenant_id": "t", "original_query": "q"}


def test_min_chunk_score_sits_below_the_high_confidence_threshold():
    assert nodes.MIN_CHUNK_SCORE < nodes.HIGH_CONFIDENCE_THRESHOLD


async def test_retrieve_drops_low_scoring_chunks(monkeypatch):
    _fake_retrieve(monkeypatch, [0.9, 0.6, 0.1])
    result = await nodes.retrieve_node(RETRIEVE_STATE)
    assert [c["score"] for c in result["ranked_chunks"]] == [0.9, 0.6]
    assert result["confidence"] == 0.9 and result["is_high_confidence"]


async def test_low_confidence_keeps_the_top_score_even_when_every_chunk_is_dropped(monkeypatch):
    _fake_retrieve(monkeypatch, [0.3, 0.2])
    result = await nodes.retrieve_node(RETRIEVE_STATE)
    assert result["ranked_chunks"] == []
    assert result["confidence"] == 0.3 and not result["is_high_confidence"]


async def test_retrieve_with_no_candidates(monkeypatch):
    _fake_retrieve(monkeypatch, [])
    result = await nodes.retrieve_node(RETRIEVE_STATE)
    assert result["ranked_chunks"] == [] and result["confidence"] == 0.0


# ── Fallback flag ─────────────────────────────────────────────

class FailingLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("LLM down")


class NoSummaryDB:
    """Stands in for AsyncSessionLocal: the session has no stored summary."""
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *args, **kwargs):
        class Result:
            def fetchone(self):
                return None
        return Result()


async def test_failed_strategy_check_falls_back_to_precise(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: FailingLLM())
    assert await nodes._determine_rag_strategy("How do I structure a large project?") == ("PRECISE", True)


async def test_failed_hyde_and_rewrite_search_with_the_question(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: FailingLLM())
    state = {"original_query": "that thing", "messages": [HumanMessage(content="that thing")]}
    assert await nodes.hyde_generate_node(state) == {"hyde_document": None, "fallback_used": True}
    assert await nodes.multi_query_rewrite_node(state) == {"rewritten_queries": ["that thing"], "fallback_used": True}


async def test_failed_web_search_is_flagged(monkeypatch):
    from backend.mcp import client

    async def down(**kwargs):
        raise ConnectionError("MCP server down")
    monkeypatch.setattr(client, "call_mcp_tool", down)
    monkeypatch.setattr(nodes, "get_settings", lambda: type("S", (), {"web_search_mcp_url": "http://127.0.0.1:1"})())
    assert await nodes.web_search_node({"original_query": "q"}) == {"web_search_results": [], "fallback_used": True}


async def test_classify_query_resets_the_flag_and_flags_a_lost_summary(monkeypatch):
    from backend import dependencies
    state = {"messages": [HumanMessage(content="hello!")], "student_id": "s", "session_id": "x", "fallback_used": True}
    monkeypatch.setattr(dependencies, "AsyncSessionLocal", NoSummaryDB)
    assert (await nodes.classify_query_node(state))["fallback_used"] is False   # A new turn starts clean

    class BrokenDB(NoSummaryDB):
        async def execute(self, *args, **kwargs):
            raise ConnectionError("database down")
    monkeypatch.setattr(dependencies, "AsyncSessionLocal", BrokenDB)
    assert (await nodes.classify_query_node(state))["fallback_used"] is True
