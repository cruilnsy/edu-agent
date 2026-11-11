# tests/unit/test_qa_dataset.py
# The QA eval set must keep its shape: unique ids, 8/6/6 covered questions per topic with a gold
# section and a reference answer, and uncovered questions with neither.

import json
from collections import Counter
from pathlib import Path

DATASET = Path(__file__).resolve().parents[2] / "scripts" / "evaluation" / "datasets" / "qa" / "questions.jsonl"


def test_qa_dataset_shape():
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len({r["id"] for r in rows}) == len(rows)
    assert Counter(r["topic"] for r in rows) == {"backend": 8, "frontend": 6, "ai": 6, "uncovered": 3}
    for r in rows:
        covered = r["topic"] != "uncovered"
        assert bool(r["gold_sections"]) == covered, r["id"]
        assert bool(r["reference_answer"]) == covered, r["id"]
