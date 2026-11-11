# scripts/evaluation/eval_exam.py
# Exam agent eval on datasets/exam: the AI grades of the LLM-graded questions (short answer
# and code) compared with your labels in labels.json.
#   Normalized MAE: mean |AI - human| / full score.             Target <= 0.15
#   Cohen's kappa:  agreement on low / mid / high levels.        Target >= 0.6
#   Teacher change rate: share of answers a teacher would change (|AI - human| > 10% of full).
#   Review flag recall:  of the answers off by more than 20%, the share the agent sent to the teacher.
#   Grader-instruction detection: answers that ask the grader for marks must be flagged.
#   Plus answer-sheet parse rate, fallback rate, P95 latency and tokens per submission.
# Each answer sheet is rendered to .docx and parsed by the agent's parser, then graded by the
# agent's three tracks directly (no API, no DB writes, no weak-point analysis).
#
# Label first: the script refuses to run while labels.json has empty scores, so you do not see
# AI grades before scoring the answers yourself.
#
# Run from the project root:
#     python -m scripts.evaluation.eval_exam
#     python -m scripts.evaluation.eval_exam --runs 2          # also checks run-to-run consistency

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from backend.agents.exam import nodes
from scripts.evaluation import metrics
from scripts.evaluation.exam_docx import DATASET_DIR, load_exam, load_submissions, render_answer_sheet
from scripts.evaluation.report import check, print_summary, write_report
from scripts.evaluation.tracking import UsageTracker

LLM_GRADED = ("short_answer", "code")

# Targets from the evaluation plan; revisit them after the first baseline.
MAX_NORMALIZED_MAE = 0.15
MIN_KAPPA = 0.6
MAX_P95_LATENCY_S = 60
TEACHER_CHANGE_TOLERANCE = 0.10     # A teacher would change a grade off by more than 10% of the full score
LARGE_ERROR = 0.20                  # Off by more than 20%: the flag for teacher review should catch it


def load_labels() -> dict[str, dict]:
    return {row["id"]: row for row in json.loads((DATASET_DIR / "labels.json").read_text(encoding="utf-8"))}


def exam_questions(exam: dict) -> tuple[list[dict], dict[str, list[dict]]]:
    """exam.json in the shape load_questions_meta reads from the DB."""
    rows = [{"id": q["question_id"], "question_no": q["question_no"], "question_type": q["question_type"],
             "content": q["content"], "correct_answer": q["correct_answer"], "score": q["score"],
             "knowledge_tag": q["knowledge_tag"]} for q in exam["questions"]]
    points = {q["question_id"]: [{"id": f"{q['question_id']}-p{i}", "desc": sp["desc"], "score": sp["score"]}
                                 for i, sp in enumerate(q.get("scoring_points", []))]
              for q in exam["questions"]}
    return rows, points


async def grade_submission(exam: dict, submission: dict, tmp_dir: Path) -> dict:
    """Render, parse and grade one answer sheet. Returns the parse check and the per-question grades."""
    sheet = render_answer_sheet(exam, submission["answers"], tmp_dir / f"{submission['id']}.docx")
    parsed = nodes._sync_parse_word(str(sheet))
    parsed_answers = {str(p["question_no"]): p["student_answer"] for p in parsed}
    parse_ok = all(parsed_answers.get(no, "") == answer.strip() for no, answer in submission["answers"].items())

    rows, points = exam_questions(exam)
    questions = nodes.merge_questions(rows, points, parsed)

    async def run(qs):
        tracks = await nodes.run_three_tracks_node({"parsed_questions": qs})
        return await nodes.aggregate_results_node(tracks)

    tracker = UsageTracker()
    with tracker.timed():
        # RunnableLambda passes the tracker to every LLM call made inside the tracks
        aggregated = await RunnableLambda(run).ainvoke(questions, config={"callbacks": [tracker]})
    return {"parse_ok": parse_ok, "by_question": aggregated["pre_review_summary"]["by_question"],
            "fallback_used": aggregated["fallback_used"], **tracker.snapshot()}


def answer_rows(submissions: list[dict], labels: dict[str, dict], runs: list[list[dict]]) -> list[dict]:
    """One row per LLM-graded answer: human score, AI score of each run, flags."""
    rows = []
    for i, submission in enumerate(submissions):
        for result in runs[i][0]["by_question"]:
            if result["question_type"] not in LLM_GRADED:
                continue
            no = str(result["question_no"])
            ai_scores = [next(r for r in run["by_question"] if r["question_no"] == result["question_no"])["score"]
                         for run in runs[i]]
            rows.append({
                "submission": submission["id"], "question_no": no, "question_type": result["question_type"],
                "full_score": result["full_score"], "human": labels[submission["id"]]["scores"][no],
                "ai": ai_scores[0], "ai_runs": ai_scores, "needs_review": result["needs_review"],
                "confidence": result.get("confidence"), "fallback": bool(result.get("fallback")),
                "grader_instructions_expected": no in submission.get("grader_instructions", []),
                "grader_instructions_found": bool(result.get("grader_instructions_found")),
                "feedback": result.get("ai_feedback", ""),
            })
    return rows


def grading_metrics(rows: list[dict], runs: list[list[dict]], n_runs: int) -> list:
    errors = [abs(r["ai"] - r["human"]) / r["full_score"] for r in rows]
    large = [r for r, e in zip(rows, errors) if e > LARGE_ERROR]
    injected = [r for r in rows if r["grader_instructions_expected"]]
    clean = [r for r in rows if not r["grader_instructions_expected"]]
    first_runs = [run[0] for run in runs]

    results = [
        check("normalized_mae", statistics.mean(errors), max_value=MAX_NORMALIZED_MAE,
              note=f"{len(rows)} answers; mean |AI - human| / full score"),
    ]
    for qtype in LLM_GRADED:
        typed = [r for r in rows if r["question_type"] == qtype]
        results.append(check(f"mae_points_{qtype}", metrics.mae([r["ai"] for r in typed], [r["human"] for r in typed]),
                             note=f"{len(typed)} answers"))
    results += [
        check("cohen_kappa", metrics.cohen_kappa([metrics.to_level(r["ai"], r["full_score"]) for r in rows],
                                                 [metrics.to_level(r["human"], r["full_score"]) for r in rows]),
              min_value=MIN_KAPPA, note="levels: < 60% low, < 85% mid, else high"),
        check("teacher_change_rate", metrics.rate([e > TEACHER_CHANGE_TOLERANCE for e in errors]),
              note=f"|AI - human| > {TEACHER_CHANGE_TOLERANCE:.0%} of full score"),
        check("review_flag_rate", metrics.rate([r["needs_review"] for r in rows]),
              note="answers sent to the teacher"),
        check("review_flag_recall", metrics.rate([r["needs_review"] for r in large]),
              note=f"{len(large)} answers off by > {LARGE_ERROR:.0%}"),
        check("grader_instructions_detected", metrics.rate([r["grader_instructions_found"] for r in injected]),
              note=f"{len(injected)} answers ask the grader for marks"),
        check("grader_instructions_false_alarms", metrics.rate([r["grader_instructions_found"] for r in clean])),
        check("parse_rate", metrics.rate([run["parse_ok"] for run in first_runs]), min_value=1.0,
              note="answer sheets whose every answer was read back exactly"),
        check("fallback_rate", metrics.rate([r["fallback"] for r in rows])),
        check("latency_p95_s", metrics.percentile([run["latency_s"] for subs in runs for run in subs], 95),
              max_value=MAX_P95_LATENCY_S, note="per submission"),
        check("tokens_per_submission", statistics.mean(run["input_tokens"] + run["output_tokens"]
                                                       for subs in runs for run in subs)),
    ]
    if n_runs > 1:
        results.append(check("max_score_spread", max(metrics.spread(r["ai_runs"]) for r in rows),
                             note=f"largest max - min across {n_runs} runs of the same answer"))
    return results


async def main_async(n_runs: int) -> None:
    exam, submissions, labels = load_exam(), load_submissions(), load_labels()
    missing = [f"{s['id']} Q{no}" for s in submissions for no, v in labels[s["id"]]["scores"].items() if v is None]
    if missing:
        sys.exit(f"labels.json still has {len(missing)} empty scores (e.g. {', '.join(missing[:3])}). "
                 "Score every answer before running the eval; see datasets/exam/README.md.")

    runs: list[list[dict]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for submission in submissions:
            runs.append([await grade_submission(exam, submission, Path(tmp)) for _ in range(n_runs)])
            print(f"  graded {submission['id']}")

    rows = answer_rows(submissions, labels, runs)
    results = grading_metrics(rows, runs, n_runs)
    meta = {"submissions": len(submissions), "llm_graded_answers": len(rows), "runs": n_runs,
            "review_confidence": nodes.REVIEW_CONFIDENCE}
    usage_keys = ("latency_s", "llm_calls", "input_tokens", "cached_input_tokens", "output_tokens")
    requests = [{"submission": s["id"], **{k: run[k] for k in usage_keys}, "fallback": run["fallback_used"]}
                for s, subs in zip(submissions, runs) for run in subs]
    _, md_path = write_report("exam", results, rows, meta, requests=requests)
    print_summary(results)

    print("\nAnswers off by more than 10% of the full score:")
    for r in rows:
        if abs(r["ai"] - r["human"]) / r["full_score"] > TEACHER_CHANGE_TOLERANCE:
            flag = "flagged" if r["needs_review"] else "NOT flagged"
            print(f"  [{r['submission']} Q{r['question_no']}] AI {r['ai']} vs you {r['human']} "
                  f"/{r['full_score']} ({flag}, conf {r['confidence']})  {r['feedback'][:100]}")
    print(f"\nReport: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exam agent eval.")
    parser.add_argument("--runs", type=int, default=1, help="Grade each submission this many times")
    args = parser.parse_args()
    asyncio.run(main_async(args.runs))


if __name__ == "__main__":
    main()
