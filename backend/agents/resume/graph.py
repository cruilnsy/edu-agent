# backend/agents/resume/graph.py

from langgraph.graph import END, START, StateGraph

from backend.agents.resume.nodes import (
    diagnose_issues_node,
    download_pdf_node,
    extract_structured_node,
    extract_text_node,
    generate_summary_node,
    run_six_dimensions_node,
    save_results_node,
    upload_to_minio_node,
)
from backend.agents.resume.state import ResumeState


def build_resume_graph(persist: bool = True):
    """Build and compile the resume review graph.

    A straight line: no branches, no interrupt, no checkpointer (one-shot job).
    The six parallel dimension reviews are the slowest step.
    persist=False skips save_results, so evals can run without touching the database.
    """
    builder = StateGraph(ResumeState)

    steps = [
        ("upload_to_minio",    upload_to_minio_node),
        ("download_pdf",       download_pdf_node),
        ("extract_text",       extract_text_node),
        ("extract_structured", extract_structured_node),
        ("run_six_dimensions", run_six_dimensions_node),
        ("diagnose_issues",    diagnose_issues_node),
        ("generate_summary",   generate_summary_node),
    ]
    if persist:
        steps.append(("save_results", save_results_node))

    for name, node in steps:
        builder.add_node(name, node)

    # Chain the nodes: START -> step 1 -> ... -> last step -> END
    names = [name for name, _ in steps]
    for src, dst in zip([START] + names, names + [END]):
        builder.add_edge(src, dst)

    # No checkpointer: a one-shot job never needs to resume
    return builder.compile()
