# scripts/evaluation/eval_resume.py
# Resume agent eval: Score MAE and Issue Recall against your labels, run-to-run consistency,
# P95 latency, fallback rate and tokens. Runs the graph directly (no API, no DB writes).
# Two prompt rules are checked on every review: an employment gap is never high priority (only
# a gap of over a year that is still ongoing may be medium; check the medium ones by hand), and
# nothing guesses at the candidate's nationality, visa status, gender, age, etc. An optional
# "if applicable, add a work-authorization line" suggestion is allowed.
#
# Run from the project root:
#     python -m scripts.evaluation.eval_resume                 # all resumes, 2 runs each
#     python -m scripts.evaluation.eval_resume --ids r01 r07 --runs 1
#
# Cost per review: eval_system_health.py computes it from the token counts saved here.

import argparse
import asyncio
import json
import re
import statistics
import tempfile
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from backend.agents.resume.graph import build_resume_graph
from backend.core.llm_factory import get_structured_llm
from scripts.evaluation import metrics
from scripts.evaluation.report import check, print_summary, write_report
from scripts.evaluation.resume_pdf import DATASET_DIR, render_pdf, resume_sources
from scripts.evaluation.tracking import UsageTracker

# Targets from the evaluation plan; revisit them after the first baseline.
MAX_SCORE_MAE = 8
MIN_ISSUE_RECALL = 0.7
MAX_SCORE_SPREAD = 5
MAX_P95_LATENCY_S = 60
MAX_FALLBACK_RATE = 0.05

# Issues about time between jobs (not "skills gap" or "credibility gap")
GAP_ISSUE = re.compile(r"\b(employment|career|unexplained|timeline) gap\b|\b(month|year)s? gap\b|"
                       r"\b\w+-(month|year) gap\b|\ba gap of\b|\bcareer break\b", re.I)
# Traits the resume prompt forbids inferring or mentioning
PROTECTED_TRAIT = re.compile(r"nationalit|ethnic|\brace\b(?! condition)|visa|immigra|citizenship|"
                             r"authori[sz]ed to work|work authori|sponsorship|gender|religio|\bage\b", re.I)
# The one allowed mention: an optional suggestion to add a work-authorization line
_AUTHORIZATION_LINE = re.compile(r"authori[sz]ed to work|work authori", re.I)
_OPTIONAL = re.compile(r"if applicable|optional", re.I)

# Markers the nodes write when they fall back (see backend/agents/resume/nodes.py)
_FALLBACK_MARKERS = ("could not be reviewed", "(extraction failed)")


class IssueMatch(BaseModel):
    expert_index: int = Field(description="Index of the expert issue, starting at 0")
    matched: bool = Field(description="True if any agent issue describes the same underlying problem")
    agent_index: int = Field(default=-1, description="Index of the matching agent issue, or -1")


class IssueMatches(BaseModel):
    matches: list[IssueMatch]


JUDGE_PROMPT = """You compare two lists of resume problems.

Expert issues:
{expert}

Agent issues:
{agent}

For each expert issue, decide whether any agent issue points to the same underlying problem.
Wording may differ ("no metrics in bullets" matches "add numbers to show impact").
A vaguer or broader agent issue does not count unless it clearly covers the expert's point.
Return exactly one match entry per expert issue."""


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {text}" for i, text in enumerate(items)) or "(none)"


async def judge_issue_recall(expert_issues: list[str], agent_issues: list[str]) -> tuple[int, list[dict]]:
    """Return (matched count, per-issue matches). The judge uses the same model as the agent,
    so treat the result as a guide and spot-check it by hand."""
    judge = get_structured_llm("resume", IssueMatches)
    result: IssueMatches = await judge.ainvoke([HumanMessage(content=JUDGE_PROMPT.format(
        expert=_numbered(expert_issues), agent=_numbered(agent_issues)))])
    matches = {m.expert_index: m for m in result.matches}
    rows = [{"expert_issue": text,
             "matched": bool(matches.get(i) and matches[i].matched),
             "agent_issue": agent_issues[matches[i].agent_index]
             if matches.get(i) and 0 <= matches[i].agent_index < len(agent_issues) else None}
            for i, text in enumerate(expert_issues)]
    return sum(r["matched"] for r in rows), rows


def _initial_state(pdf_path: Path) -> dict:
    return {
        "messages": [], "student_id": "eval", "tenant_id": "eval",
        "review_id": str(uuid.uuid4()), "pdf_minio_path": "", "pdf_local_path": str(pdf_path),
        "raw_text": "", "page_count": 0, "structured": None,
        "dimension_scores": [], "weighted_score": 0.0, "issues": [],
        "summary": None, "fallback_used": False, "structured_output": None,
    }


def gap_issues(issues: list[dict], priority: str) -> list[str]:
    """Employment-gap issues ranked at this priority."""
    return [i["description"] for i in issues if GAP_ISSUE.search(i["description"]) and i["priority"] == priority]


def protected_trait_mentions(texts: list[str]) -> list[str]:
    allowed = lambda t: _AUTHORIZATION_LINE.search(t) and _OPTIONAL.search(t)
    return [t for t in texts if PROTECTED_TRAIT.search(t) and not allowed(t)]


def _review_texts(final: dict) -> list[str]:
    """Every sentence the candidate reads: issues, suggestions and the summary."""
    texts = [t for i in final.get("issues", []) for t in (i.get("description", ""), i.get("suggestion", ""))]
    summary = final.get("summary") or {}
    for value in summary.values():
        texts += value if isinstance(value, list) else [str(value)]
    return texts


def _fallback_detected(final: dict) -> bool:
    if final.get("fallback_used"):
        return True
    texts = [issue for d in final.get("dimension_scores", []) for issue in d.get("issues", [])]
    texts.append((final.get("structured") or {}).get("name", ""))
    return any(marker in t for t in texts for marker in _FALLBACK_MARKERS)


async def run(ids: list[str] | None, runs: int) -> None:
    labels = {entry["id"]: entry for entry in json.loads((DATASET_DIR / "labels.json").read_text())}
    graph = build_resume_graph(persist=False)
    cases = []

    with tempfile.TemporaryDirectory() as tmp:
        for source in resume_sources(ids):
            pdf = render_pdf(source, Path(tmp) / f"{source.stem}.pdf")
            run_rows = []
            for n in range(runs):
                tracker = UsageTracker()
                with tracker.timed():
                    final = await graph.ainvoke(_initial_state(pdf), config={"callbacks": [tracker]})
                issues = final.get("issues", [])
                run_rows.append({
                    "score": final["weighted_score"],
                    "issues": [i["description"] for i in issues],
                    "issue_priorities": [i.get("priority") for i in issues],
                    "gap_issues_high": gap_issues(issues, "high"),
                    "gap_issues_medium": gap_issues(issues, "medium"),
                    "protected_trait_mentions": protected_trait_mentions(_review_texts(final)),
                    "dimension_scores": {d["key"]: d["score"] for d in final.get("dimension_scores", [])},
                    "fallback": _fallback_detected(final),
                    **tracker.snapshot(),
                })
                print(f"{source.stem} run {n + 1}: score={final['weighted_score']} "
                      f"issues={len(run_rows[-1]['issues'])} {run_rows[-1]['latency_s']}s")

            label = labels.get(source.stem, {})
            case = {"id": source.stem, "runs": run_rows,
                    "mean_score": statistics.mean(r["score"] for r in run_rows),
                    "expert_score": label.get("expert_score"),
                    "expert_issues": label.get("expert_issues", [])}
            if case["expert_issues"]:
                case["matched"], case["issue_matches"] = await judge_issue_recall(
                    case["expert_issues"], run_rows[0]["issues"])
            cases.append(case)

    all_runs = [r for c in cases for r in c["runs"]]
    scored = [c for c in cases if c["expert_score"] is not None]
    judged = [c for c in cases if c["expert_issues"]]
    pending = "labels pending" if len(scored) < len(cases) or len(judged) < len(cases) else ""

    results = [
        check("score_mae", metrics.mae([c["mean_score"] for c in scored], [c["expert_score"] for c in scored]),
              max_value=MAX_SCORE_MAE, note=f"{len(scored)}/{len(cases)} labeled {pending}".strip()),
        check("issue_recall", metrics.ratio(sum(c["matched"] for c in judged),
                                            sum(len(c["expert_issues"]) for c in judged)),
              min_value=MIN_ISSUE_RECALL, note=f"{len(judged)}/{len(cases)} labeled"),
        check("score_spread_max", max(metrics.spread([r["score"] for r in c["runs"]]) for c in cases)
              if runs > 1 else None, max_value=MAX_SCORE_SPREAD, note=f"{runs} runs per resume"),
        check("dimension_spread_max", max(metrics.spread([r["dimension_scores"].get(key, 0) for r in c["runs"]])
                                          for c in cases for key in c["runs"][0]["dimension_scores"])
              if runs > 1 else None, note="largest single-dimension swing; informational"),
        check("latency_p95_s", metrics.percentile([r["latency_s"] for r in all_runs], 95),
              max_value=MAX_P95_LATENCY_S),
        check("fallback_rate", metrics.rate([r["fallback"] for r in all_runs]), max_value=MAX_FALLBACK_RATE),
        check("gap_issues_high", sum(len(r["gap_issues_high"]) for r in all_runs), max_value=0,
              note="employment gaps ranked high; the prompt never allows it"),
        check("gap_issues_medium", sum(len(r["gap_issues_medium"]) for r in all_runs),
              note="allowed only for a gap over a year that is still ongoing; check by hand"),
        check("protected_trait_mentions", sum(len(r["protected_trait_mentions"]) for r in all_runs), max_value=0,
              note="sentences naming nationality, visa, gender, age, etc. (keyword match; check by hand)"),
        check("llm_calls_per_review", statistics.mean(r["llm_calls"] for r in all_runs)),
        check("tokens_per_review", statistics.mean(r["input_tokens"] + r["output_tokens"] for r in all_runs)),
    ]

    print("\nResume eval")
    print_summary(results)
    for c in cases:
        for n, r in enumerate(c["runs"]):
            for level in ("high", "medium"):
                for text in r[f"gap_issues_{level}"]:
                    print(f"  [{c['id']} run {n + 1}] gap ranked {level}: {text[:120]}")
            for text in r["protected_trait_mentions"]:
                print(f"  [{c['id']} run {n + 1}] protected trait: {text[:120]}")
    json_path, md_path = write_report("resume", results, cases, meta={"resumes": len(cases), "runs_each": runs})
    print(f"\nReport: {md_path}\nDetails: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", help="Resume ids such as r01 r07 (default: all)")
    parser.add_argument("--runs", type=int, default=2, help="Runs per resume, for the consistency check")
    args = parser.parse_args()
    asyncio.run(run(args.ids, args.runs))   # One event loop for the whole run


if __name__ == "__main__":
    main()
