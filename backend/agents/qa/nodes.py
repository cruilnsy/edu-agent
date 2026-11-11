# backend/agents/qa/nodes.py

import asyncio
import re
import uuid
from datetime import datetime

from sqlalchemy import text
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.agents.qa.state import QAState
from backend.agents.qa.prompts import (
    HYDE_PROMPT,
    MULTI_QUERY_REWRITE_PROMPT,
    RAG_ANSWER_PROMPT,
    DIRECT_ANSWER_PROMPT,
    GENERAL_ANSWER_PROMPT,
    RAG_STRATEGY_PROMPT,
    SYSTEM_PROMPT,
)
from backend.core.llm_factory import get_llm
from backend.core.memory import (
    trim_messages_to_window,
    should_trigger_summary,
    compress_to_summary,
    build_thread_id,
)
from backend.config import get_settings
from backend.core.logger import get_logger
from backend.core.query_classifier import get_query_classifier

logger = get_logger(__name__)

# ── Retrieval settings ──
MAX_BROAD_QUERIES        = 3   # BROAD: sub-queries searched in parallel
RECALL_TOP_K_PRECISE     = 8   # PRECISE: candidates recalled for reranking
RECALL_TOP_K_VAGUE       = 10  # VAGUE: recall more after HyDE widens the meaning
RECALL_TOP_K_BROAD_PER   = 4   # BROAD: candidates per sub-query
RERANK_TOP_K             = 3   # Chunks kept after reranking
HIGH_CONFIDENCE_THRESHOLD = 0.75   # Top rerank score needed to answer from the knowledge base
MIN_CHUNK_SCORE          = 0.35   # Chunks scoring lower are left out of the answer prompt and sources
LLM_STRATEGY_MIN_WORDS   = 8   # Ask the LLM to confirm VAGUE / BROAD only for longer questions
SHORT_QUERY_MAX_WORDS    = 5   # "Vague" hints count only in short messages


# ── Helpers ──

def _get_message_content(msg) -> str:
    """Message text, whether the model exposes .text or only .content."""
    if hasattr(msg, "text") and not callable(getattr(msg, "text", None)):
        return msg.text
    if isinstance(msg.content, str):
        return msg.content
    return str(msg.content)


def _format_history_for_prompt(messages: list) -> str:
    """Recent turns as prompt text; assistant answers are cut short to keep the prompt small."""
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"Student: {_get_message_content(msg)}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {_get_message_content(msg)[:200]}...")
    return "\n".join(lines) if lines else "(no earlier conversation)"


def _current_datetime_str() -> str:
    """Current local time for the prompts, e.g. "Friday, November 6, 2026, 13:45"."""
    now = datetime.now()
    return f"{now:%A, %B} {now.day}, {now:%Y, %H:%M}"


def _build_system_content(summary: str | None = None) -> str:
    """System prompt with the current time, plus the session summary when there is one."""
    content = SYSTEM_PROMPT + f"\n\nCurrent time: {_current_datetime_str()}"
    if summary:
        content += f"\n\nSummary of this student's earlier sessions:\n{summary}"
    return content


# ── Detect a request to search the web ──

_WEB_SEARCH_REQUEST = re.compile(
    r"[\s,;:-]*"                       # Keep the question's own ending punctuation
    r"(?:\b(?:and|or|please|if needed|if you (?:don't|do not|dont) know(?: it)?|you can|can you|feel free to)\b[\s,]*)*"
    r"\b(?:search (?:the web|online|the internet)|look (?:it|this|that) up|look online|"
    r"google (?:it|this|that)|web search|check online|browse the web)\b.*$",
    re.IGNORECASE,
)


def _extract_query_and_web_flag(raw: str) -> tuple[str, bool]:
    """Return (question without the search request, whether to search the web).

    The request is removed so the LLM does not treat it as part of the question, e.g.
    "What is Spring AOP? If you don't know, search the web" -> ("What is Spring AOP", True).
    """
    match = _WEB_SEARCH_REQUEST.search(raw)
    if not match:
        return raw, False
    clean = raw[:match.start()].strip()
    return (clean or raw), True


# ── Rules (layer 0, before the classifier) ──

_GENERAL_EXACT = {
    "hi", "hello", "hey", "hi there", "hello there", "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "thanks a lot", "thank you so much", "thx", "ty",
    "who are you", "what are you", "what is your name", "what's your name",
    "what can you do", "how can you help", "what can you help me with",
    "bye", "goodbye", "see you", "ok", "okay", "cool", "got it",
}

_GENERAL_PHRASES = (
    "who are you", "what can you do", "what can you help", "introduce yourself", "your name",
    "weather", "tell me a joke", "tell me a story",
    "what time is it", "current time", "what day is it", "what's the date", "what is the date",
    "today's date", "todays date", "what day of the week",
)

# A match means a knowledge base question: skip the classifier and go to the strategy step
_SPECIALIZED_PHRASES = (
    "knowledge base", "in the docs", "the documentation", "according to the docs",
    "in the course", "this course", "the lesson", "this lesson", "the chapter",
)

_VAGUE_QUERY_HINTS = (
    "don't get", "dont get", "don't understand", "dont understand", "confused", "lost me",
    "explain", "what does that mean", "what do you mean", "huh", "not clear",
)
_BROAD_QUERY_HINTS = (
    "overview", "summarize", "summary", "roadmap", "big picture", "compare", "comparison",
    "difference between", "differences between", "pros and cons", "everything about", "all about",
    "what are the",
)


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower()).strip(" .!?,")


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    """Whole-word phrase match, so "hi" never matches inside "this"."""
    return any(re.search(rf"\b{re.escape(p)}\b", text) for p in phrases)


def _rule_classify_general(query: str) -> bool:
    """Small talk, greetings, date and time -> GENERAL."""
    q = _normalize(query)
    return q in _GENERAL_EXACT or _contains_phrase(q, _GENERAL_PHRASES)


def _rule_classify_specialized(query: str) -> bool:
    """Explicit reference to the course or its docs -> knowledge base, skipping the classifier."""
    return _contains_phrase(_normalize(query), _SPECIALIZED_PHRASES)


def _fast_rag_strategy(query: str) -> str:
    """Rule-based strategy (< 1 ms)."""
    q = _normalize(query)
    if len(q.split()) <= SHORT_QUERY_MAX_WORDS and _contains_phrase(q, _VAGUE_QUERY_HINTS):
        return "VAGUE"
    if _contains_phrase(q, _BROAD_QUERY_HINTS):
        return "BROAD"
    return "PRECISE"


async def _determine_rag_strategy(query: str) -> tuple[str, bool]:
    """LLM strategy check; PRECISE when the call fails or the answer is not a strategy.
    Returns (strategy, fell_back)."""
    try:
        llm = get_llm("qa", temperature=0)
        resp = await llm.ainvoke([HumanMessage(content=RAG_STRATEGY_PROMPT.format(query=query))])
        label = _get_message_content(resp).strip().upper()
        if label in ("PRECISE", "VAGUE", "BROAD"):
            return label, False
        logger.warning("classify_query.rag_strategy_unknown", label=label[:20])
    except Exception as e:
        logger.warning("classify_query.rag_strategy_failed", error=str(e))
    return "PRECISE", True   # Safest default


async def _determine_rag_strategy_fast(query: str) -> tuple[str, bool]:
    """Two-step strategy:
    1. Rules say PRECISE -> PRECISE, no LLM call.
    2. Rules say VAGUE / BROAD on a longer question -> let the LLM confirm (rules misfire there).
    3. Rules say VAGUE / BROAD on a short question -> trust the rules.
    """
    strategy = _fast_rag_strategy(query)
    if strategy == "PRECISE":
        return strategy, False
    if len(query.split()) >= LLM_STRATEGY_MIN_WORDS:
        return await _determine_rag_strategy(query)
    return strategy, False


# ── Node: classify_query - pick the path, load the session summary ──

async def classify_query_node(state: QAState) -> dict:
    """Decide the path for this question and load the session summary from the DB.

    query_type:
      GENERAL -> answer with the LLM, no retrieval
      PRECISE -> search with the question
      VAGUE   -> HyDE first, then search
      BROAD   -> Multi-Query rewrite, then search each sub-query
    """
    messages = state.get("messages", [])
    raw_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            raw_query = _get_message_content(msg)
            break

    original_query, auto_web = _extract_query_and_web_flag(raw_query)

    existing_summary: str | None = None
    fallback = False            # Set when a step below fails and a default is used
    try:
        from backend.dependencies import AsyncSessionLocal
        thread_id = build_thread_id(state["student_id"], state["session_id"])
        async with AsyncSessionLocal() as db:
            row = (await db.execute(
                text("SELECT summary FROM qa_sessions WHERE thread_id = :tid"),
                {"tid": thread_id},
            )).fetchone()
            existing_summary = row[0] if row else None
    except Exception as e:
        logger.warning("classify_query.load_memory_failed", error=str(e))
        fallback = True         # Answers without the session summary

    _base: dict = {
        "original_query":    original_query,
        "existing_summary":  existing_summary,
        "rewritten_queries": [],
        "hyde_document":     None,
        "fallback_used":     fallback,   # Reset every turn (the checkpointer keeps state between turns)
    }
    if auto_web and not state.get("enable_web_search", False):
        _base["enable_web_search"] = True
        logger.info("classify_query.auto_web_enabled", query=original_query[:50])

    # Layer 0a: rules -> GENERAL (small talk, date and time)
    if _rule_classify_general(original_query):
        logger.info("classify_query.general_by_rule", query=original_query[:50])
        return {**_base, "query_type": "GENERAL"}

    # Layer 0b: rules -> knowledge base (explicit course or docs reference)
    if _rule_classify_specialized(original_query):
        logger.info("classify_query.specialized_by_keyword", query=original_query[:50])
        strategy, failed = await _determine_rag_strategy_fast(original_query)
        logger.info("classify_query.rag_strategy", strategy=strategy)
        return {**_base, "query_type": strategy, "fallback_used": fallback or failed}

    # Layer 1: MiniLM classifier (CPU; run in a thread so the event loop stays free)
    loop = asyncio.get_running_loop()
    label, confidence = await loop.run_in_executor(None, get_query_classifier().classify, original_query)

    if label == "general":
        logger.info("classify_query.general_by_minilm", query=original_query[:50],
                    confidence=round(confidence, 4))
        return {**_base, "query_type": "GENERAL"}

    # Layer 2: knowledge base question -> pick the retrieval strategy
    logger.info("classify_query.specialized_by_minilm", query=original_query[:50],
                confidence=round(confidence, 4))
    strategy, failed = await _determine_rag_strategy_fast(original_query)
    logger.info("classify_query.rag_strategy", strategy=strategy)
    return {**_base, "query_type": strategy, "fallback_used": fallback or failed}


# ── Node: hyde_generate - hypothetical document (VAGUE) ──

async def hyde_generate_node(state: QAState) -> dict:
    """For a vague question, have the LLM write a hypothetical answer and search with its vector
    instead of the question's.

    temperature=0.3: some variety to cover more meanings, but not so random it drifts off topic.
    """
    query    = state["original_query"]
    messages = state.get("messages", [])

    history_text = _format_history_for_prompt(messages[-6:])  # Last 3 turns

    prompt = HYDE_PROMPT.format(history=history_text, query=query)

    try:
        response = await get_llm("qa", temperature=0.3).ainvoke([HumanMessage(content=prompt)])
        hyde_doc = _get_message_content(response).strip()
    except Exception as e:      # Retrieval then searches with the question itself
        logger.warning("hyde_generate.failed", error=str(e))
        return {"hyde_document": None, "fallback_used": True}

    logger.info(
        "hyde_generate.done",
        query=query[:50],
        hyde_doc_length=len(hyde_doc),
    )

    return {"hyde_document": hyde_doc}


# ── Node: multi_query_rewrite - Multi-Query rewrite (BROAD) ──

async def multi_query_rewrite_node(state: QAState) -> dict:
    """Rewrite a broad or very short question as 3-5 specific sub-queries; retrieve_node searches
    them in parallel and merges the results."""
    query    = state["original_query"]
    messages = state.get("messages", [])

    # The previous answer tells us what "this" or "that" refers to
    last_ai_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_text = _get_message_content(msg)
            break

    prompt = MULTI_QUERY_REWRITE_PROMPT.format(
        last_answer=last_ai_text[:500] if last_ai_text else "(no previous answer)",
        query=query,
    )

    try:
        response = await get_llm("qa", temperature=0.3).ainvoke([HumanMessage(content=prompt)])
    except Exception as e:      # Retrieval then searches with the question itself
        logger.warning("multi_query_rewrite.failed", error=str(e))
        return {"rewritten_queries": [query], "fallback_used": True}

    # One sub-query per line; strip numbering or bullets such as "1. " or "- "
    raw = _get_message_content(response).strip()
    rewritten = [
        line.lstrip("0123456789.-)•* ").strip()
        for line in raw.split("\n")
        if line.strip() and len(line.strip()) > 3
    ][:MAX_BROAD_QUERIES]

    if not rewritten:
        rewritten = [query]   # Rewrite produced nothing: fall back to the question

    logger.info(
        "multi_query_rewrite.done",
        original=query,
        count=len(rewritten),
        queries=rewritten,
    )

    return {"rewritten_queries": rewritten}


# ── Node: retrieve - hybrid recall + rerank ──

async def retrieve_node(state: QAState) -> dict:
    """Run the retrieve() pipeline (hybrid recall + rerank) and store ranked_chunks.

    PRECISE -> search with original_query
    VAGUE   -> search with hyde_document instead (wider meaning)
    BROAD   -> search every rewritten query in parallel, then merge and deduplicate

    retrieve() is synchronous (BGE-M3 on CPU + blocking Milvus calls), so it runs in an executor
    to keep the event loop free.
    """
    from backend.core.reranker import retrieve, RankedDocument

    query_type     = state.get("query_type", "PRECISE").upper()
    tenant_id      = state["tenant_id"]
    course_id      = state.get("course_id")
    original_query = state["original_query"]

    loop = asyncio.get_running_loop()

    # ── BROAD: search the sub-queries in parallel, merge and deduplicate ──
    if query_type == "BROAD" and state.get("rewritten_queries"):
        broad_queries = state["rewritten_queries"][:MAX_BROAD_QUERIES]

        async def retrieve_one(sub_query: str) -> tuple[list, float]:
            return await loop.run_in_executor(
                None,
                lambda: retrieve(
                    sub_query,
                    tenant_id,
                    course_id,
                    recall_top_k=RECALL_TOP_K_BROAD_PER,
                    rerank_top_k=RECALL_TOP_K_BROAD_PER,
                ),
            )

        results = await asyncio.gather(*[retrieve_one(q) for q in broad_queries])

        # Deduplicate on the first 100 characters, keeping the highest score
        seen: dict[str, RankedDocument] = {}
        for ranked_docs, _ in results:
            for doc in ranked_docs:
                key = doc.content[:100]
                if key not in seen or doc.score > seen[key].score:
                    seen[key] = doc

        merged = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:RERANK_TOP_K]

    # ── PRECISE / VAGUE: one search ──
    else:
        if query_type == "VAGUE" and state.get("hyde_document"):
            query_text   = state["hyde_document"]
            recall_top_k = RECALL_TOP_K_VAGUE
        else:
            query_text   = original_query
            recall_top_k = RECALL_TOP_K_PRECISE

        merged, _ = await loop.run_in_executor(
            None,
            lambda: retrieve(
                query_text,
                tenant_id,
                course_id,
                recall_top_k=recall_top_k,
                rerank_top_k=RERANK_TOP_K,
            ),
        )

    # ── RankedDocument -> dict for the State ──
    ranked_chunks = [
        {
            "content":  doc.content,
            "score":    doc.score,
            "metadata": doc.metadata,
        }
        for doc in merged
    ]

    # Confidence is the top chunk's score, taken before filtering
    confidence         = ranked_chunks[0]["score"] if ranked_chunks else 0.0
    is_high_confidence = confidence >= HIGH_CONFIDENCE_THRESHOLD

    # Drop weak chunks so they neither mislead the answer nor show up as sources. On the
    # high-confidence path the top chunk scores >= HIGH_CONFIDENCE_THRESHOLD > MIN_CHUNK_SCORE,
    # so at least one chunk always remains.
    recalled = len(ranked_chunks)
    ranked_chunks = [c for c in ranked_chunks if c["score"] >= MIN_CHUNK_SCORE]

    logger.info(
        "retrieve.done",
        query_type=query_type,
        ranked=len(ranked_chunks),
        dropped_low_score=recalled - len(ranked_chunks),
        confidence=round(confidence, 4),
        is_high_confidence=is_high_confidence,
    )

    return {
        "ranked_chunks":      ranked_chunks,
        "confidence":         confidence,
        "is_high_confidence": is_high_confidence,
    }


# ── Node: generate_rag - answer from the knowledge base (high confidence) ──

async def generate_rag_node(state: QAState) -> dict:
    """Answer from the top reranked chunks, list the sources at the end, and include the session
    summary so multi-turn answers stay consistent."""
    ranked_chunks = state.get("ranked_chunks", [])
    query         = state["original_query"]
    messages      = state.get("messages", [])
    summary       = state.get("existing_summary")

    # Context and source list
    context_parts = []
    sources = []
    for i, chunk in enumerate(ranked_chunks, 1):
        context_parts.append(f"[Excerpt {i}]\n{chunk['content']}")
        source_name = chunk.get("metadata", {}).get("source_name", "Knowledge base")
        if source_name not in sources:
            sources.append(source_name)

    context_text = "\n\n".join(context_parts)

    # System message with the current time and the session summary
    llm_messages = [SystemMessage(content=_build_system_content(summary))]

    # Recent turns, without the last HumanMessage: it is already in RAG_ANSWER_PROMPT's {query},
    # and sending it twice hurts the answer.
    windowed = trim_messages_to_window(messages[:-1], window_size=10)
    for msg in windowed:
        if not isinstance(msg, SystemMessage):
            llm_messages.append(msg)

    rag_prompt = RAG_ANSWER_PROMPT.format(context=context_text, query=query)
    llm_messages.append(HumanMessage(content=rag_prompt))

    llm = get_llm("qa", streaming=True)
    response = await llm.ainvoke(llm_messages)
    answer_text = _get_message_content(response).strip()

    sources_text = "\n".join([f"  • {s}" for s in sources])
    final_answer = f"{answer_text}\n\n📚 **Sources**\n{sources_text}"

    logger.info(
        "generate_rag.done",
        answer_length=len(final_answer),
        sources=sources,
        confidence=round(state.get("confidence", 0), 4),
    )

    return {
        "answer":      final_answer,
        "sources":     sources,
        "answer_mode": "rag",
        "messages":    [AIMessage(content=final_answer)],
        "should_summarize": should_trigger_summary(messages),
        "structured_output": {
            "answer":      final_answer,
            "sources":     sources,
            "confidence":  state.get("confidence", 0),
            "answer_mode": "rag",
        },
    }


# ── Node: web_search - web results for low confidence or GENERAL + web ──

async def web_search_node(state: QAState) -> dict:
    """Fetch web results through the Web Search MCP server. If the server is unavailable, return no
    results and let the answer node continue without them."""
    from backend.mcp.client import call_mcp_tool

    settings = get_settings()

    if not settings.web_search_mcp_url:
        return {"web_search_results": []}

    try:
        results = await call_mcp_tool(
            server_url=settings.web_search_mcp_url,
            tool_name="web_search",
            arguments={
                "query":       state["original_query"],
                "max_results": 3,
            },
            timeout=10.0,
        )
        count = len(results or [])
        logger.info("web_search.done", count=count)
        return {"web_search_results": results or []}
    except Exception as e:
        logger.warning("web_search.failed", error=str(e))
        return {"web_search_results": [], "fallback_used": True}


# ── Node: generate_direct - LLM answer when retrieval is not good enough ──

async def generate_direct_node(state: QAState) -> dict:
    """Answer from the LLM's own knowledge when the knowledge base has nothing good enough.
    With web results they go into the prompt (web_augmented); without, the answer ends with a
    notice that it does not come from the knowledge base (llm_direct)."""
    query    = state["original_query"]
    messages = state.get("messages", [])
    summary  = state.get("existing_summary")

    llm_messages = [SystemMessage(content=_build_system_content(summary))]

    windowed = trim_messages_to_window(messages[:-1], window_size=10)
    for msg in windowed:
        if not isinstance(msg, SystemMessage):
            llm_messages.append(msg)

    # ── Web results ──
    web_results = state.get("web_search_results") or []
    web_context = ""
    web_sources: list[str] = []
    if web_results:
        snippets = "\n".join(
            f"  [{i + 1}] {r.get('title', '')} ({r.get('url', '')})\n"
            f"      {r.get('snippet', '')[:300]}"
            for i, r in enumerate(web_results)
        )
        web_context = f"\n\nWeb search results:\n{snippets}"
        web_sources = [r.get("url", "") for r in web_results if r.get("url")]

    direct_prompt = DIRECT_ANSWER_PROMPT.format(query=query) + web_context
    llm_messages.append(HumanMessage(content=direct_prompt))

    llm = get_llm("qa", streaming=True)
    response = await llm.ainvoke(llm_messages)
    answer_text = _get_message_content(response).strip()

    if web_sources:
        # URLs go to the frontend in sources (shown in a collapsible panel), not in the text
        final_answer = answer_text
        answer_mode  = "web_augmented"
    else:
        final_answer = (
            f"{answer_text}\n\n"
            f"⚠️ **Note**: this answer comes from the AI's general knowledge; the knowledge base "
            f"has nothing on it yet. Check it against your course materials or ask a teacher."
        )
        answer_mode = "llm_direct"

    logger.info(
        "generate_direct.done",
        answer_length=len(final_answer),
        confidence=round(state.get("confidence", 0), 4),
        web_sources=len(web_sources),
    )

    return {
        "answer":      final_answer,
        "sources":     web_sources,
        "answer_mode": answer_mode,
        "messages":    [AIMessage(content=final_answer)],
        "should_summarize": should_trigger_summary(messages),
        "structured_output": {
            "answer":      final_answer,
            "sources":     web_sources,
            "confidence":  state.get("confidence", 0),
            "answer_mode": answer_mode,
        },
    }


# ── Node: generate_general - answer GENERAL questions without retrieval ──

async def generate_general_node(state: QAState) -> dict:
    """Answer GENERAL questions (greetings, date and time, off-topic) without retrieval; web results,
    when present, add up-to-date information."""
    query       = state["original_query"]
    messages    = state.get("messages", [])
    web_results = state.get("web_search_results") or []

    web_context = ""
    web_sources: list[str] = []
    if web_results:
        snippets = "\n".join(
            f"  [{i + 1}] {r.get('title', '')} ({r.get('url', '')})\n"
            f"      {r.get('snippet', '')[:300]}"
            for i, r in enumerate(web_results)
        )
        web_context = f"Web search results:\n{snippets}\n\n"
        web_sources = [r.get("url", "") for r in web_results if r.get("url")]

    history_text = _format_history_for_prompt(messages[-6:])
    prompt = GENERAL_ANSWER_PROMPT.format(
        query=query,
        history=history_text,
        current_time=_current_datetime_str(),
        web_context=web_context,
    )

    llm = get_llm("qa", streaming=True)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    answer_text = _get_message_content(response).strip()

    answer_mode = "web_augmented" if web_sources else "general"

    logger.info(
        "generate_general.done",
        answer_length=len(answer_text),
        web_sources=len(web_sources),
    )

    return {
        "answer":      answer_text,
        "sources":     web_sources,
        "answer_mode": answer_mode,
        "messages":    [AIMessage(content=answer_text)],
        "should_summarize": should_trigger_summary(messages),
        "structured_output": {
            "answer":      answer_text,
            "sources":     web_sources,
            "confidence":  1.0,
            "answer_mode": answer_mode,
        },
    }


# ── Node: enqueue_pending - queue low-confidence questions for teachers ──

async def enqueue_pending_node(state: QAState) -> dict:
    """Add the low-confidence question to knowledge_pending_queue for teachers to fill the gap.
    ON CONFLICT DO NOTHING keeps it idempotent. Failures are logged only; the answer is already
    written. Returns {} (no State change)."""
    from backend.dependencies import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    text("""
                        INSERT INTO knowledge_pending_queue
                            (id, tenant_id, question, student_id, confidence, status)
                        VALUES (:id, :tenant_id, :question, :student_id, :confidence, 'pending')
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "id":         str(uuid.uuid4()),
                        "tenant_id":  state["tenant_id"],
                        "question":   state["original_query"],
                        "student_id": state["student_id"],
                        "confidence": state.get("confidence", 0.0),
                    },
                )
        logger.info(
            "enqueue_pending.done",
            question=state["original_query"][:50],
            confidence=state.get("confidence", 0),
        )
    except Exception as e:
        logger.warning("enqueue_pending.failed", error=str(e))

    return {}


# ── Node: save_memory - summary and session record ──

async def save_memory_node(state: QAState) -> dict:
    """Compress the history into the summary when should_summarize is set (every 10 turns), then
    upsert qa_sessions. Both steps only log on failure. Returns {} (no State change)."""
    from backend.dependencies import AsyncSessionLocal

    messages   = state.get("messages", [])
    student_id = state["student_id"]
    session_id = state["session_id"]
    tenant_id  = state["tenant_id"]
    thread_id  = build_thread_id(student_id, session_id)
    summary    = state.get("existing_summary")

    # ── Summary compression ──
    if state.get("should_summarize", False):
        try:
            # Compress only the last 10 turns (<= 20 messages); older content lives in
            # existing_summary. Passing every message would grow with the conversation until it
            # exceeds the model's context window.
            msgs_to_compress = trim_messages_to_window(messages, window_size=10)
            summary = await compress_to_summary(
                messages=msgs_to_compress,
                existing_summary=summary,
            )
            logger.info("save_memory.summary_compressed", thread_id=thread_id)
        except Exception as e:
            logger.warning("save_memory.compress_failed", error=str(e))

    # ── Upsert qa_sessions ──
    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                await session.execute(
                    text("""
                        INSERT INTO qa_sessions
                            (id, tenant_id, student_id, thread_id, summary, summary_version)
                        VALUES (:id, :tenant_id, :student_id, :thread_id, :summary, 1)
                        ON CONFLICT (thread_id) DO UPDATE
                            SET summary         = EXCLUDED.summary,
                                summary_version = qa_sessions.summary_version + 1,
                                updated_at      = NOW()
                    """),
                    {
                        "id":         str(uuid.uuid4()),
                        "tenant_id":  tenant_id,
                        "student_id": student_id,
                        "thread_id":  thread_id,
                        "summary":    summary,
                    },
                )
    except Exception as e:
        logger.warning("save_memory.db_write_failed", error=str(e))

    return {}
