# backend/agents/interview/graph.py

from langgraph.graph import END, START, StateGraph

from backend.agents.interview.nodes import (
    check_stage_node, evaluate_answer_node, generate_report_node, generate_response_node,
    load_context_node, route_after_check_stage, save_memory_node, save_report_node,
)
from backend.agents.interview.state import InterviewState
from backend.core.memory import get_memory_saver


def build_interview_graph(checkpointer=None):
    """Build and compile the mock interview graph. One run handles one candidate message.

        START -> load_context -> evaluate_answer -> check_stage
                     ├─(finished)-> generate_report -> save_report -> END
                     └─(else)    -> generate_response -> END
                                    save_memory -> END     (in parallel, every 10 rounds)

    evaluate_answer runs before check_stage, so each answer is graded against the question it
    answers even when the stage then changes. The history summary runs next to the reply, not
    after it, so those turns are not slower; the stream only forwards generate_response tokens. The checkpointer keeps the state between
    requests (one thread per session); pass another one to run the graph in isolation.
    """
    builder = StateGraph(InterviewState)

    builder.add_node("load_context",      load_context_node)
    builder.add_node("evaluate_answer",   evaluate_answer_node)
    builder.add_node("check_stage",       check_stage_node)
    builder.add_node("generate_response", generate_response_node)
    builder.add_node("generate_report",   generate_report_node)
    builder.add_node("save_report",       save_report_node)
    builder.add_node("save_memory",       save_memory_node)

    builder.add_edge(START,             "load_context")
    builder.add_edge("load_context",    "evaluate_answer")
    builder.add_edge("evaluate_answer", "check_stage")
    builder.add_conditional_edges("check_stage", route_after_check_stage,
                                  ["generate_report", "generate_response", "save_memory"])

    builder.add_edge("generate_response", END)               # Normal turn
    builder.add_edge("save_memory",       END)               # Every 10 rounds, next to the reply
    builder.add_edge("generate_report",   "save_report")     # Last turn
    builder.add_edge("save_report",       END)

    return builder.compile(checkpointer=checkpointer or get_memory_saver("interview"))
