# scripts/evaluation/eval_orchestrator.py
# Routing eval of the unified assistant on datasets/orchestrator/routing.jsonl: each message
# goes through route_message (rules, then the LLM) and the label is compared with the gold one.
#   Accuracy and macro F1 over all labels; per-label recall; confusion matrix.
#   Rule hits on messages that should reach the LLM (and the reverse).
#   Fallback rate, P95 latency and tokens of the LLM-routed messages.
#
# Run from the project root:
#     python -m scripts.evaluation.eval_orchestrator

import asyncio
import json
import statistics
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from backend.core.orchestrator import ALL_LABELS, RULE_LABELS, route_message
from scripts.evaluation import metrics
from scripts.evaluation.report import check, print_summary, write_report
from scripts.evaluation.tracking import UsageTracker

DATASET = Path(__file__).parent / "datasets" / "orchestrator" / "routing.jsonl"
MIN_ACCURACY = 0.9           # From the evaluation plan: set after the first baseline
MIN_MACRO_F1 = 0.85
MAX_P95_LATENCY_S = 3


def load_cases() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


async def route_case(case: dict) -> dict:
    tracker = UsageTracker()
    with tracker.timed():
        decision = await RunnableLambda(route_message).ainvoke(case["text"], config={"callbacks": [tracker]})
    return {**case, "predicted": decision.label, "source": decision.source, "confidence": decision.confidence,
            "reason": decision.reason, "correct": decision.label == case["label"], **tracker.snapshot()}


async def main_async() -> None:
    cases = [await route_case(c) for c in load_cases()]
    gold, predicted = [c["label"] for c in cases], [c["predicted"] for c in cases]
    llm_cases = [c for c in cases if c["source"] != "rule"]
    labels = [label for label in ALL_LABELS if label in gold]

    results = [
        check("accuracy", metrics.accuracy(predicted, gold), min_value=MIN_ACCURACY, note=f"{len(cases)} messages"),
        check("macro_f1", metrics.macro_f1(predicted, gold), min_value=MIN_MACRO_F1),
    ]
    for label in labels:
        of_label = [c for c in cases if c["label"] == label]
        results.append(check(f"recall_{label}", metrics.rate([c["correct"] for c in of_label]),
                             note=f"{len(of_label)} messages"))
    results += [
        check("rule_false_hits", sum(1 for c in cases if c["source"] == "rule" and c["label"] not in RULE_LABELS),
              max_value=0, note="messages for an agent answered by a fixed reply"),
        check("rule_misses", sum(1 for c in cases if c["source"] != "rule" and c["label"] in RULE_LABELS),
              note="social messages sent to the LLM (costs a call, still answered)"),
        check("fallback_rate", metrics.rate([c["source"] == "fallback" for c in cases])),
        check("latency_p95_llm_s", metrics.percentile([c["latency_s"] for c in llm_cases], 95),
              max_value=MAX_P95_LATENCY_S, note=f"{len(llm_cases)} LLM-routed messages"),
        check("tokens_per_llm_route", statistics.mean(c["input_tokens"] + c["output_tokens"] for c in llm_cases)),
    ]

    matrix = metrics.confusion(predicted, gold, labels)
    meta = {"messages": len(cases), "labels": len(labels)}
    _, md_path = write_report("orchestrator", results, cases, meta)
    print_summary(results)

    width = max(len(label) for label in labels)
    print(f"\nConfusion matrix (rows: gold, columns: predicted)\n{'':<{width}}  "
          + " ".join(f"{label[:5]:>5}" for label in labels))
    for label, row in zip(labels, matrix):
        print(f"{label:<{width}}  " + " ".join(f"{n:>5}" for n in row))

    wrong = [c for c in cases if not c["correct"]]
    print(f"\nMisrouted ({len(wrong)}):")
    for c in wrong:
        print(f"  [{c['id']}] gold {c['label']}, got {c['predicted']} ({c['source']}): {c['text']}  -- {c['reason']}")
    print(f"\nReport: {md_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
