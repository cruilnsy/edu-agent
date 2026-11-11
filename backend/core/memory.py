# backend/core/memory.py
# Conversation memory: per-agent checkpointers, thread ids, window trimming and summary compression.

from typing import Optional
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from backend.core.logger import get_logger

logger = get_logger(__name__)

# One MemorySaver per agent: their State schemas differ, and sharing one saver makes the
# msgpack serialization clash between schemas
_memory_savers: dict[str, MemorySaver] = {}


def get_memory_saver(agent_type: str = "default") -> MemorySaver:
    """MemorySaver singleton for an agent ("qa", "exam", "resume", "interview"), passed to
    StateGraph.compile(checkpointer=...). In-memory: history is lost on restart. Swapping in
    AsyncPostgresSaver makes it persistent without changing the agents."""
    if agent_type not in _memory_savers:
        _memory_savers[agent_type] = MemorySaver()
        logger.info("memory.saver_initialized", agent=agent_type)
    return _memory_savers[agent_type]


def build_thread_id(student_id: str, session_id: str) -> str:
    """Checkpointer thread id, "student_{student_id}_session_{session_id}": each session of a
    student keeps its own history."""
    return f"student_{student_id}_session_{session_id}"


def build_config(student_id: str, session_id: str) -> dict:
    """Config for graph.ainvoke(state, config=...): {"configurable": {"thread_id": ...}}."""
    return {
        "configurable": {
            "thread_id": build_thread_id(student_id, session_id),
        }
    }


def trim_messages_to_window(
    messages: list[BaseMessage],
    window_size: int = 10,
) -> list[BaseMessage]:
    """Keep the last window_size turns (1 turn = 1 human + 1 AI message). System messages are
    always kept, first."""
    system_messages  = [m for m in messages if isinstance(m, SystemMessage)]
    dialogue_messages = [m for m in messages if not isinstance(m, SystemMessage)]

    max_dialogue_messages = window_size * 2

    if len(dialogue_messages) <= max_dialogue_messages:
        return messages   # Within the window

    trimmed = dialogue_messages[-max_dialogue_messages:]

    logger.info(
        "memory.window_trimmed",
        original=len(dialogue_messages),
        kept=len(trimmed),
        window_size=window_size,
    )

    return system_messages + trimmed


def should_trigger_summary(
    messages: list[BaseMessage],
    threshold: int = 10,
) -> bool:
    """True every `threshold` turns (10, 20, 30, ...): time to compress the history."""
    dialogue_count = sum(
        1 for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
    )
    rounds = dialogue_count // 2
    return rounds >= threshold and rounds % threshold == 0


async def compress_to_summary(
    messages: list[BaseMessage],
    existing_summary: Optional[str] = None,
) -> str:
    """Compress the conversation into a learner profile summary. Incremental: the previous
    summary is passed in so its content is not repeated."""
    from langchain_core.messages import HumanMessage as LCHuman
    from backend.core.llm_factory import get_llm

    SUMMARY_PROMPT = """Compress the student conversation below into a structured learner profile.

Keep: topics the student clearly did not understand, weak spots that come up repeatedly, new
details about the student's projects.
Keep briefly: topics the student has mastered, learning progress.
Drop: anything already in the previous summary, small talk, questions resolved and understood.

Previous summary:
{previous_summary}

New conversation:
{new_conversations}

Output only the summary, with no heading or prefix."""

    conversation_text = "\n".join([
        f"{'Student' if isinstance(m, HumanMessage) else 'Assistant'}: {m.text if hasattr(m, 'text') and not callable(m.text) else str(m.content)}"
        for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
    ])

    prompt_text = SUMMARY_PROMPT.format(
        previous_summary=existing_summary or "(none)",
        new_conversations=conversation_text,
    )

    llm = get_llm("summarize")
    response = await llm.ainvoke([LCHuman(content=prompt_text)])
    summary = (
        response.content if isinstance(response.content, str)
        else str(response.content)
    ).strip()

    logger.info(
        "memory.summary_generated",
        input_messages=len(messages),
        summary_length=len(summary),
    )
    return summary
