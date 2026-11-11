# scripts/evaluation/summary.py
# One table of the core metrics, from the latest report of every eval in eval_results/.
# Makes no LLM calls. Quality metrics come from each agent's eval; latency, tokens, cost and
# fallback rates come from the system_health report, so each number appears once.
#
#     python -m scripts.evaluation.summary
#
# run_all.py calls this after running the evals. Each source report is listed with its time
# and commit, so a report older than the current code is easy to spot.

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from scripts.evaluation.eval_system_health import latest_report
from scripts.evaluation.report import RESULTS_DIR, Metric, git_commit

# Order of the sections in the summary
AGENT_EVALS = ("orchestrator", "classifier", "qa", "resume", "exam", "interview")

# Metrics without a target that are still worth a line, per eval
EXTRAS = {
    "classifier": ("held_out_accuracy", "held_out_macro_f1", "boundary_accuracy", "boundary_specialized_recall"),
    "qa": ("recall@1_rerank", "rag_path_rate_covered", "uncovered_not_rag_rate"),
    "resume": ("dimension_spread_max",),
    "exam": ("teacher_change_rate", "review_flag_recall", "grader_instructions_detected"),
    "interview": ("interviewer_instructions_caught", "one_question_per_message"),
}

# system_health owns these; skipped in the agent sections to avoid showing a number twice
_HEALTH_WORDS = ("latency", "tokens", "fallback", "llm_calls")


@dataclass
class Section:
    eval_name: str
    source: str                          # Report file and commit, or why there is none
    metrics: list[Metric] = field(default_factory=list)


def _metric(row: dict) -> Metric:
    return Metric(row["name"], row["value"], row.get("target", ""), row.get("passed"), row.get("note", ""))


def core_metrics(eval_name: str, report: dict) -> list[Metric]:
    """Metrics with a target, plus the eval's listed extras."""
    rows = [_metric(r) for r in report["metrics"]]
    if eval_name == "system_health":
        return rows                      # Laid out as one row per request in _health_lines
    extras = EXTRAS.get(eval_name, ())
    return [m for m in rows if not any(w in m.name for w in _HEALTH_WORDS) and (m.target or m.name in extras)]


def _source(path: Path, report: dict, head: str) -> str:
    commit = report["meta"].get("git_commit", "not recorded")
    stale = "" if commit.removesuffix("-dirty") == head.removesuffix("-dirty") else " (older code)"
    return f"{path.stem}, commit {commit}{stale}"


def build_sections(results_dir: Path = RESULTS_DIR, head: str | None = None) -> list[Section]:
    head = head or git_commit()
    sections = []
    for name in (*AGENT_EVALS, "system_health"):
        found = latest_report(name, results_dir)
        if not found:
            sections.append(Section(name, "missing: run the eval first"))
            continue
        path, report = found
        sections.append(Section(name, _source(path, report, head), core_metrics(name, report)))
    return sections


def _value(m: Metric | None) -> str:
    return m.display_value() if m else "n/a"


def _health_lines(metrics: list[Metric]) -> list[str]:
    """system_health metrics are named <request>.<metric>; one table row per request kind."""
    status = {True: "pass", False: "**FAIL**", None: "-"}
    by_request: dict[str, dict[str, Metric]] = {}
    for m in metrics:
        request, _, metric = m.name.partition(".")
        by_request.setdefault(request, {})[metric] = m
    lines = ["| Request | P50 (s) | P95 (s) | Target | Result | LLM calls | Tokens in / out | Cache hit "
             "| Cost USD off-peak / peak |", "|---|---|---|---|---|---|---|---|---|"]
    fallbacks = []
    for request, ms in by_request.items():
        if "fallback_rate" in ms:
            fallbacks.append((request, ms["fallback_rate"]))
            continue
        p95 = ms.get("latency_p95_s")
        tokens = (f"{_value(ms.get('input_tokens_per_request'))} / {_value(ms.get('output_tokens_per_request'))}"
                  if "input_tokens_per_request" in ms else _value(ms.get("tokens_per_request")))
        cost = ms.get("cost_usd_per_request")
        cost_text = "n/a" if cost is None or cost.value is None else (
            f"{cost.value:.5f} / {ms['cost_usd_per_request_peak'].value:.5f}" + (" (upper bound)" if cost.note else ""))
        lines.append(f"| {request} | {_value(ms.get('latency_p50_s'))} | {_value(p95)} | {p95.target if p95 else ''} "
                     f"| {status[p95.passed if p95 else None]} | {_value(ms.get('llm_calls_per_request'))} "
                     f"| {tokens} | {_value(ms.get('cache_hit_rate'))} | {cost_text} |")
    lines += ["", "| Agent | Fallback rate | Target | Result | Note |", "|---|---|---|---|---|"]
    lines += [f"| {a} | {m.display_value()} | {m.target} | {status[m.passed]} | {m.note} |" for a, m in fallbacks]
    return lines


def to_markdown(sections: list[Section], head: str, runs: list[dict] | None = None) -> str:
    status = {True: "pass", False: "**FAIL**", None: "-"}
    all_metrics = [(s.eval_name, m) for s in sections for m in s.metrics]
    failed = [f"{e}.{m.name}" if e != "system_health" else m.name for e, m in all_metrics if m.passed is False]
    passed = sum(m.passed is True for _, m in all_metrics)

    lines = ["# Eval summary", "",
             f"- **generated**: {datetime.now():%Y-%m-%d %H:%M}",
             f"- **code**: commit {head}",
             f"- **targets**: {passed} pass, {len(failed)} fail"]
    if failed:
        lines.append(f"- **failing**: {', '.join(failed)}")
    if runs:
        lines += ["", "## This run", "", "| Eval | Result | Time (s) |", "|---|---|---|"]
        lines += [f"| {r['eval']} | {r['result']} | {r['seconds']:.0f} |" for r in runs]
    for s in sections:
        lines += ["", f"## {s.eval_name}", "", f"Source: {s.source}", ""]
        if s.metrics and s.eval_name == "system_health":
            lines += _health_lines(s.metrics)
        elif s.metrics:
            lines += ["| Metric | Value | Target | Result | Note |", "|---|---|---|---|---|"]
            lines += [f"| {m.name} | {m.display_value()} | {m.target} | {status[m.passed]} | {m.note} |"
                      for m in s.metrics]
    return "\n".join(lines) + "\n"


def write_summary(runs: list[dict] | None = None, results_dir: Path = RESULTS_DIR) -> Path:
    head = git_commit()
    sections = build_sections(results_dir, head)
    out_dir = results_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"{stamp}.md"
    md_path.write_text(to_markdown(sections, head, runs))
    data = {"git_commit": head, "runs": runs or [],
            "sections": [{"eval": s.eval_name, "source": s.source,
                          "metrics": [m.__dict__ for m in s.metrics]} for s in sections]}
    (out_dir / f"{stamp}.json").write_text(json.dumps(data, indent=2, default=str))
    return md_path


def main() -> None:
    md_path = write_summary()
    print(md_path.read_text())
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
