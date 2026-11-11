# backend/agents/exam/graph.py

from langgraph.graph import StateGraph, START, END

from backend.agents.exam.state import ExamState
from backend.agents.exam.nodes import (
    parse_word_node,
    load_questions_meta_node,
    run_three_tracks_node,
    aggregate_results_node,
    analyze_weak_points_node,
    notify_teacher_node,
    teacher_review_node,
    apply_teacher_decision_node,
    publish_results_node,
)
from backend.core.memory import get_memory_saver


def build_exam_graph():
    """Build and compile the exam grading graph.

    Linear flow:
        parse_word -> load_questions_meta -> run_three_tracks
        -> aggregate_results -> analyze_weak_points
        -> notify_teacher -> teacher_review [interrupt]
        -> apply_teacher_decision -> publish_results -> END
    """
    builder = StateGraph(ExamState)

    # Nodes
    builder.add_node("parse_word",             parse_word_node)
    builder.add_node("load_questions_meta",    load_questions_meta_node)
    builder.add_node("run_three_tracks",       run_three_tracks_node)
    builder.add_node("aggregate_results",      aggregate_results_node)
    builder.add_node("analyze_weak_points",    analyze_weak_points_node)
    builder.add_node("notify_teacher",         notify_teacher_node)
    builder.add_node("teacher_review",         teacher_review_node)
    builder.add_node("apply_teacher_decision", apply_teacher_decision_node)
    builder.add_node("publish_results",        publish_results_node)

    # Edges: one linear chain
    builder.add_edge(START,                    "parse_word")
    builder.add_edge("parse_word",             "load_questions_meta")
    builder.add_edge("load_questions_meta",    "run_three_tracks")
    builder.add_edge("run_three_tracks",       "aggregate_results")
    builder.add_edge("aggregate_results",      "analyze_weak_points")
    builder.add_edge("analyze_weak_points",    "notify_teacher")
    builder.add_edge("notify_teacher",         "teacher_review")
    builder.add_edge("teacher_review",         "apply_teacher_decision")
    builder.add_edge("apply_teacher_decision", "publish_results")
    builder.add_edge("publish_results",        END)

    # The checkpointer keeps the paused state until the teacher resumes it
    checkpointer = get_memory_saver("exam")
    return builder.compile(checkpointer=checkpointer)
