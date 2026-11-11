# scripts/evaluation/eval_classifier.py
# QA query classifier eval: accuracy, macro F1 and confusion matrix with the production decision rule
# (general only when P(general) >= GENERAL_CONFIDENCE_THRESHOLD), on two sets:
#   held_out: a fixed 10% of the generated data (datasets/classifier/held_out.jsonl), excluded from training
#   boundary: hand-written cases near the label boundary (datasets/classifier/boundary.jsonl)
# The held-out set comes from the same generator as the training data, so expect it to score higher.
#
# Run from the project root:
#     python -m scripts.evaluation.eval_classifier

import time
from pathlib import Path

from backend.config import resolve_path
from backend.core.query_classifier import GENERAL_CONFIDENCE_THRESHOLD, HELD_OUT_DATA, QueryClassifier
from scripts.evaluation import metrics
from scripts.evaluation.report import check, print_summary, write_report

BOUNDARY_PATH = Path(__file__).parent / "datasets" / "classifier" / "boundary.jsonl"
LABELS = ["general", "specialized"]


def evaluate(classifier: QueryClassifier, name: str, rows: list[dict]) -> tuple[list, list[dict]]:
    cases = []
    for row in rows:
        start = time.perf_counter()
        label, confidence = classifier.classify(row["text"])
        cases.append({"set": name, "text": row["text"], "gold": row["label"], "predicted": label,
                      "confidence": round(confidence, 4), "latency_ms": (time.perf_counter() - start) * 1000})
    gold = [c["gold"] for c in cases]
    predicted = [c["predicted"] for c in cases]
    specialized = [c for c in cases if c["gold"] == "specialized"]
    results = [
        check(f"{name}_accuracy", metrics.accuracy(predicted, gold)),
        check(f"{name}_macro_f1", metrics.macro_f1(predicted, gold)),
        check(f"{name}_specialized_recall",
              metrics.ratio(sum(c["predicted"] == "specialized" for c in specialized), len(specialized)),
              note="in-topic questions sent to RAG"),
    ]
    matrix = metrics.confusion(predicted, gold, LABELS)
    results.append(check(f"{name}_confusion", None, note=f"rows=gold, cols=predicted {LABELS}: {matrix}"))
    return results, cases


def main() -> None:
    held_out = QueryClassifier.load_jsonl(resolve_path(HELD_OUT_DATA))
    boundary = QueryClassifier.load_jsonl(str(BOUNDARY_PATH))

    classifier = QueryClassifier()
    classifier.classify("warm up")                    # Exclude model loading from latency
    all_metrics, all_cases = [], []
    for name, data in (("held_out", held_out), ("boundary", boundary)):
        results, cases = evaluate(classifier, name, data)
        all_metrics += results
        all_cases += cases
    all_metrics.append(check("latency_p95_ms", metrics.percentile([c["latency_ms"] for c in all_cases], 95)))

    meta = {"held_out": len(held_out), "boundary": len(boundary), "threshold": GENERAL_CONFIDENCE_THRESHOLD}
    json_path, md_path = write_report("classifier", all_metrics, all_cases, meta)
    print_summary(all_metrics)
    print("\nMisclassified:")
    for c in all_cases:
        if c["predicted"] != c["gold"]:
            print(f"  [{c['set']}] gold={c['gold']:<11} conf={c['confidence']:.2f}  {c['text'][:90]}")
    print(f"\nReport: {md_path}")


if __name__ == "__main__":
    main()
