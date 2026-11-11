# tests/unit/test_demo_data.py
# The recorded demo data: complete, and free of real account ids.

import json
import re

from scripts.demo.seed_demo import DATA_PATH

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def test_has_one_finished_example_of_each_agent():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    (resume,), (exam,), (interview,) = data["resume_reviews"], data["exam_submissions"], data["interview_sessions"]
    assert resume["scores"]["dimension_scores"] and resume["issues"] and resume["summary"]
    assert len(exam["reviews"]) == 7 and any(r["reviewed_by_teacher"] for r in exam["reviews"])
    assert interview["report"]["dimensions"] and interview["resume_review_id"] == resume["id"]


def test_ids_are_only_the_records_own():
    """Every UUID in the file is a demo record or part of the sample exam, never a user. Account
    ids are written as placeholders that seed_demo.py fills in."""
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    exam = data["exam_submissions"][0]
    known = {data["resume_reviews"][0]["id"], exam["id"], exam["exam_id"], data["interview_sessions"][0]["id"],
             data["interview_sessions"][0]["session_id"]}
    known |= {r["id"] for r in exam["reviews"]} | {r["question_id"] for r in exam["reviews"]}
    known |= {p["point_id"] for r in exam["reviews"] for p in (r["ai_raw_result"] or {}).get("point_results", [])}
    assert set(UUID.findall(DATA_PATH.read_text(encoding="utf-8"))) <= known
