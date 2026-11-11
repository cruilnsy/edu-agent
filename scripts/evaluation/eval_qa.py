# scripts/evaluation/eval_qa.py
# QA agent eval on datasets/qa/questions.jsonl:
#   Retrieval: Recall@1 and Recall@3 for three configurations, to show what each stage adds:
#     dense (BGE-M3 dense only) -> hybrid (dense + sparse) -> rerank (hybrid + BGE reranker, as in the agent)
#     A hit is a top-k chunk whose source_name contains one of the question's gold_sections.
#   Answers: the full QA graph per question; an LLM judge compares the answer with the reference
#     (correct = 1, partial = 0.5, wrong = 0). Uncovered questions must not take the high-confidence path.
#   MIN_CHUNK_SCORE: how many gold and non-gold chunks each threshold would drop from the reranked top 3.
#   Plus P95 latency, LLM calls and tokens per answer. The graph runs without its DB writes.
#
# Run from the project root:
#     python -m scripts.evaluation.eval_qa
#     python -m scripts.evaluation.eval_qa --skip-answers      # retrieval only, no LLM calls

import argparse
import asyncio
import json
import statistics
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from backend.api.v1.qa import qa_state
from backend.agents.qa import graph as qa_graph
from backend.agents.qa import nodes
from backend.config import get_settings
from backend.core.knowledge_base import COLLECTION_NAME, BGEMEmbedder, KnowledgeBaseClient
from backend.core.llm_factory import get_structured_llm
from backend.core.reranker import BGEReranker
from scripts.evaluation import metrics
from scripts.evaluation.report import check, print_summary, write_report
from scripts.evaluation.tracking import UsageTracker

DATASET = Path(__file__).parent / "datasets" / "qa" / "questions.jsonl"
CONFIGS = ("dense", "hybrid", "rerank")
THRESHOLDS = (0.2, 0.35, 0.5)

# Targets from the evaluation plan; revisit them after the first baseline.
MIN_RECALL_AT_3 = 0.8
MIN_ANSWER_CORRECTNESS = 0.8
MAX_P95_LATENCY_S = 8


class Verdict(BaseModel):
    verdict: str = Field(description='"correct", "partial" or "wrong"')
    reason: str = Field(description="One sentence explaining the verdict")


JUDGE_PROMPT = """You grade a teaching assistant's answer against a reference answer.

Question: {question}

Reference answer: {reference}

Assistant's answer: {answer}

- correct: covers the key facts of the reference and contradicts none of them
- partial: some key facts are right but one is missing or vague
- wrong: misses the main point or states something that contradicts the reference
Extra correct detail beyond the reference is fine. Judge the content, not the style."""

VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def load_questions() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_gold(source_name: str, gold_sections: list[str]) -> bool:
    name = source_name.lower()
    return any(g.lower() in name for g in gold_sections)


# ── Retrieval ──

def retrieve_all_configs(question: str, tenant_id: str) -> dict[str, list[dict]]:
    """Top 3 per configuration; each item has source_name and score."""
    embedder, kb = BGEMEmbedder.get_instance(), KnowledgeBaseClient()
    dense_vec, sparse_vec = embedder.encode_query(question)
    filters = kb._build_filter(tenant_id)

    dense_hits = kb._client.search(
        collection_name=COLLECTION_NAME, data=[dense_vec], anns_field="embedding", limit=3,
        filter=filters, search_params={"metric_type": "COSINE", "params": {"ef": kb.ANN_EF}},
        output_fields=["source_name"],
    )[0]
    hybrid = kb._hybrid_search(dense_vec, sparse_vec, top_k=nodes.RECALL_TOP_K_PRECISE, filters=filters)
    reranked, _ = BGEReranker.get_instance().rerank_with_confidence(question, hybrid, top_k=nodes.RERANK_TOP_K)

    return {
        "dense": [{"source_name": h["entity"]["source_name"], "score": h["distance"]} for h in dense_hits],
        "hybrid": [{"source_name": c["metadata"]["source_name"], "score": c["score"]} for c in hybrid[:3]],
        "rerank": [{"source_name": d.metadata["source_name"], "score": d.score} for d in reranked],
    }


def retrieval_metrics(cases: list[dict]) -> list:
    covered = [c for c in cases if c["gold_sections"]]
    results = []
    for config in CONFIGS:
        for k in (1, 3):
            hits = [any(is_gold(r["source_name"], c["gold_sections"]) for r in c["retrieval"][config][:k])
                    for c in covered]
            results.append(check(f"recall@{k}_{config}", metrics.rate(hits),
                                 min_value=MIN_RECALL_AT_3 if (k, config) == (3, "rerank") else None))
    for t in THRESHOLDS:
        gold_dropped = other_dropped = 0
        for c in covered:
            for r in c["retrieval"]["rerank"]:
                if r["score"] < t:
                    if is_gold(r["source_name"], c["gold_sections"]):
                        gold_dropped += 1
                    else:
                        other_dropped += 1
        current = " (current)" if t == nodes.MIN_CHUNK_SCORE else ""
        results.append(check(f"min_score_{t}_drops", None,
                             note=f"gold chunks dropped {gold_dropped}, other chunks dropped {other_dropped}{current}"))
    return results


# ── Answers ──

async def _no_side_effect(state) -> dict:
    return {}


def build_eval_graph():
    """The QA graph without its DB writes: eval runs must not add sessions or pending questions."""
    qa_graph.enqueue_pending_node = _no_side_effect
    qa_graph.save_memory_node = _no_side_effect
    return qa_graph.build_qa_graph()


async def answer_and_judge(graph, case: dict, tenant_id: str) -> dict:
    session_id = str(uuid.uuid4())
    state = qa_state(case["question"], {"user_id": "eval", "tenant_id": tenant_id}, session_id)
    tracker = UsageTracker()
    with tracker.timed():
        final = await graph.ainvoke(state, config={"configurable": {"thread_id": session_id}, "callbacks": [tracker]})
    run = {"answer": final.get("answer", ""), "answer_mode": final.get("answer_mode"),
           "query_type": final.get("query_type"), "confidence": final.get("confidence", 0.0),
           "high_confidence": bool(final.get("is_high_confidence")),
           "fallback": bool(final.get("fallback_used")), **tracker.snapshot()}

    if case["reference_answer"]:
        judge = get_structured_llm("qa", Verdict)
        verdict: Verdict = await judge.ainvoke([HumanMessage(content=JUDGE_PROMPT.format(
            question=case["question"], reference=case["reference_answer"], answer=run["answer"].split("📚")[0]))])
        run["verdict"], run["judge_reason"] = verdict.verdict.lower(), verdict.reason
    return run


def answer_metrics(cases: list[dict]) -> list:
    runs = [c["run"] for c in cases]
    judged = [c["run"] for c in cases if c["reference_answer"]]
    uncovered = [c["run"] for c in cases if not c["gold_sections"]]
    return [
        check("answer_correctness", statistics.mean(VERDICT_SCORE.get(r["verdict"], 0.0) for r in judged),
              min_value=MIN_ANSWER_CORRECTNESS),
        check("rag_path_rate_covered", metrics.rate([r["answer_mode"] == "rag" for r in judged]),
              note="covered questions answered from the knowledge base"),
        check("uncovered_not_rag_rate", metrics.rate([not r["high_confidence"] for r in uncovered]),
              note="uncovered questions kept off the high-confidence path"),
        check("fallback_rate", metrics.rate([r["fallback"] for r in runs]), max_value=0.05,
              note="a step failed and a default was used"),
        check("latency_p95_s", metrics.percentile([r["latency_s"] for r in runs], 95), max_value=MAX_P95_LATENCY_S),
        check("llm_calls_per_answer", statistics.mean(r["llm_calls"] for r in runs)),
        check("tokens_per_answer", statistics.mean(r["input_tokens"] + r["output_tokens"] for r in runs)),
    ]


async def main_async(skip_answers: bool) -> None:
    tenant_id = get_settings().default_tenant_id
    cases = load_questions()
    for case in cases:
        case["retrieval"] = retrieve_all_configs(case["question"], tenant_id)
    results = retrieval_metrics(cases)

    if not skip_answers:
        graph = build_eval_graph()
        for case in cases:
            case["run"] = await answer_and_judge(graph, case, tenant_id)
        results += answer_metrics(cases)

    meta = {"questions": len(cases), "covered": sum(bool(c["gold_sections"]) for c in cases),
            "min_chunk_score": nodes.MIN_CHUNK_SCORE, "high_confidence_threshold": nodes.HIGH_CONFIDENCE_THRESHOLD}
    _, md_path = write_report("qa", results, cases, meta)
    print_summary(results)

    print("\nRetrieval misses at 3 (rerank):")
    for c in cases:
        if c["gold_sections"] and not any(is_gold(r["source_name"], c["gold_sections"]) for r in c["retrieval"]["rerank"]):
            print(f"  [{c['id']}] {c['question']}\n      got: {[r['source_name'][:70] for r in c['retrieval']['rerank']]}")
    if not skip_answers:
        print("\nAnswers not judged correct, and uncovered questions:")
        for c in cases:
            r = c["run"]
            if r.get("verdict", "correct") != "correct" or not c["gold_sections"]:
                print(f"  [{c['id']}] {r.get('verdict', '-')} mode={r['answer_mode']} conf={r['confidence']:.2f}  "
                      f"{r.get('judge_reason', '')[:120]}")
    print(f"\nReport: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="QA agent eval.")
    parser.add_argument("--skip-answers", action="store_true", help="Retrieval metrics only (no LLM calls)")
    args = parser.parse_args()
    asyncio.run(main_async(args.skip_answers))


if __name__ == "__main__":
    main()
