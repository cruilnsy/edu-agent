# scripts/evaluation/eval_system_health.py
# System health across the agents, from the latest report of every other eval in eval_results/.
# Makes no LLM calls: it re-reads the per-request data those evals saved and computes, per
# kind of request, P50/P95 latency, LLM calls and tokens per request, the share of input tokens
# served from the provider's cache, an estimated cost (off-peak and peak), and each agent's
# fallback rate. Latency targets come from the evaluation plan.
#
# Cost uses DeepSeek's list prices by default (see pricing.py); override them in USD per 1M tokens:
#     python -m scripts.evaluation.eval_system_health --price-cache-hit 0.003 --price-cache-miss 0.15 --price-out 0.6
# Reports saved before 11/10 have no cache data, so their cost counts every input token as a miss.
#
# Run the other evals first; an agent without a report is listed as missing.

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scripts.evaluation import metrics
from scripts.evaluation.pricing import Prices, load_prices, request_cost
from scripts.evaluation.report import RESULTS_DIR, Metric, check, print_summary, write_report

MAX_FALLBACK_RATE = 0.05


@dataclass
class Series:
    """One kind of request, e.g. one interview turn."""
    name: str
    unit: str                        # What one request is, for the report
    latencies_s: list[float]
    usage: list[dict]                # Per request: llm_calls, input/cached_input/output tokens ([] if no LLM)
    max_p95_s: float | None = None   # Target from the evaluation plan


@dataclass
class AgentHealth:
    series: list[Series]
    fallback_flags: list[bool] | None   # None when the agent has no fallback path to measure
    fallback_unit: str = ""
    legacy: list[Metric] | None = None  # Numbers copied from an old report without raw requests


# ── Reading the reports ───────────────────────────────────────

def latest_report(eval_name: str, results_dir: Path = RESULTS_DIR) -> tuple[Path, dict] | None:
    files = sorted((results_dir / eval_name).glob("*.json"))
    if not files:
        return None
    return files[-1], json.loads(files[-1].read_text())


def _usage(request: dict) -> dict:
    usage = {k: request.get(k, 0) for k in ("llm_calls", "input_tokens", "output_tokens")}
    usage["cached_input_tokens"] = request.get("cached_input_tokens")   # None in reports from before 11/10
    return usage


def qa_health(report: dict) -> AgentHealth:
    runs = [c["run"] for c in report["cases"]]
    has_flag = all("fallback" in r for r in runs)    # Reports from before 11/10 have no flag
    return AgentHealth([Series("qa_answer", "answer", [r["latency_s"] for r in runs], [_usage(r) for r in runs], 8)],
                       fallback_flags=[r["fallback"] for r in runs] if has_flag else None, fallback_unit="answers")


def resume_health(report: dict) -> AgentHealth:
    runs = [run for c in report["cases"] for run in c["runs"]]
    return AgentHealth([Series("resume_review", "review", [r["latency_s"] for r in runs], [_usage(r) for r in runs], 60)],
                       fallback_flags=[bool(r.get("fallback")) for r in runs], fallback_unit="reviews")


def interview_health(report: dict) -> AgentHealth:
    starts = [c["start"] for c in report["cases"]]
    turns = [t for c in report["cases"] for t in c["turns"]]
    normal = [t for t in turns if t.get("stage_after") != "finished"]
    final = [t for t in turns if t.get("stage_after") == "finished"]
    return AgentHealth([
        Series("interview_start", "start (question generation)", [s["latency_s"] for s in starts], [_usage(s) for s in starts]),
        Series("interview_turn", "turn", [t["latency_s"] for t in normal], [_usage(t) for t in normal], 5),
        Series("interview_report_turn", "last turn, with the report", [t["latency_s"] for t in final], [_usage(t) for t in final]),
    ], fallback_flags=[bool(c.get("fallback_used")) for c in report["cases"]], fallback_unit="interviews")


def exam_health(report: dict) -> AgentHealth:
    requests = report.get("requests")
    if requests is None:          # Reports from before 11/10 only kept per-answer rows
        values = {m["name"]: m["value"] for m in report["metrics"]}
        note = "from the exam report; rerun eval_exam for raw data"
        return AgentHealth([], None, legacy=[
            check("exam_submission.latency_p95_s", values.get("latency_p95_s"), max_value=60, note=note),
            check("exam_submission.tokens_per_request", values.get("tokens_per_submission"), note=note),
            check("exam.fallback_rate", values.get("fallback_rate"), max_value=MAX_FALLBACK_RATE,
                  note="share of graded answers; " + note),
        ])
    return AgentHealth([Series("exam_submission", "submission", [r["latency_s"] for r in requests],
                               [_usage(r) for r in requests], 60)],
                       fallback_flags=[bool(r.get("fallback")) for r in requests], fallback_unit="submissions")


def orchestrator_health(report: dict) -> AgentHealth:
    routed = [c for c in report["cases"] if c["source"] != "rule"]   # Rule replies take no time
    return AgentHealth([Series("route_llm", "LLM-routed message", [c["latency_s"] for c in routed],
                               [_usage(c) for c in routed], 3)],
                       fallback_flags=[c["source"] == "fallback" for c in routed], fallback_unit="LLM routes")


def classifier_health(report: dict) -> AgentHealth:
    return AgentHealth([Series("classify", "query (local model)", [c["latency_ms"] / 1000 for c in report["cases"]], [])],
                       fallback_flags=None)


AGENTS: dict[str, Callable[[dict], AgentHealth]] = {
    "qa": qa_health, "resume": resume_health, "interview": interview_health, "exam": exam_health,
    "orchestrator": orchestrator_health, "classifier": classifier_health,
}


# ── Metrics ───────────────────────────────────────────────────

def cache_hit_rate(usage: list[dict]) -> float | None:
    """Share of input tokens served from the cache; None when the report has no cache data."""
    if any(u["cached_input_tokens"] is None for u in usage):
        return None
    total = sum(u["input_tokens"] for u in usage)
    return sum(u["cached_input_tokens"] for u in usage) / total if total else None


def cost_per_request(usage: list[dict], prices: Prices, peak: bool = False) -> float:
    return statistics.mean(request_cost(u, prices, peak) for u in usage)


def series_metrics(s: Series, prices: Prices) -> list[Metric]:
    n = f"{len(s.latencies_s)} × {s.unit}"
    results = [
        check(f"{s.name}.latency_p50_s", metrics.percentile(s.latencies_s, 50), note=n),
        check(f"{s.name}.latency_p95_s", metrics.percentile(s.latencies_s, 95), max_value=s.max_p95_s, note=n),
    ]
    if s.usage:
        hit_rate = cache_hit_rate(s.usage)
        cost_note = "" if hit_rate is not None else "no cache data: every input token priced as a miss (upper bound)"
        results += [
            check(f"{s.name}.llm_calls_per_request", statistics.mean(u["llm_calls"] for u in s.usage)),
            check(f"{s.name}.input_tokens_per_request", statistics.mean(u["input_tokens"] for u in s.usage)),
            check(f"{s.name}.output_tokens_per_request", statistics.mean(u["output_tokens"] for u in s.usage)),
            check(f"{s.name}.cache_hit_rate", hit_rate, note="share of input tokens"),
            check(f"{s.name}.cost_usd_per_request", cost_per_request(s.usage, prices), note=cost_note),
            check(f"{s.name}.cost_usd_per_request_peak", cost_per_request(s.usage, prices, peak=True), note=cost_note),
        ]
    return results


def agent_metrics(agent: str, health: AgentHealth, prices: Prices) -> list[Metric]:
    results = list(health.legacy or [])
    for s in health.series:
        if s.latencies_s:
            results += series_metrics(s, prices)
    if health.fallback_flags:
        results.append(check(f"{agent}.fallback_rate", metrics.rate(health.fallback_flags), max_value=MAX_FALLBACK_RATE,
                             note=f"{len(health.fallback_flags)} {health.fallback_unit}"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="System health from the latest eval reports.")
    parser.add_argument("--price-cache-hit", type=float, help="Off-peak USD per 1M cached input tokens")
    parser.add_argument("--price-cache-miss", type=float, help="Off-peak USD per 1M other input tokens")
    parser.add_argument("--price-out", type=float, help="Off-peak USD per 1M output tokens")
    args = parser.parse_args()
    prices, prices_source = load_prices(cache_hit=args.price_cache_hit, cache_miss=args.price_cache_miss,
                                        output=args.price_out)

    results: list[Metric] = []
    sources: dict[str, str] = {}
    for agent, read in AGENTS.items():
        found = latest_report(agent)
        if not found:
            sources[agent] = "missing: run the eval first"
            continue
        path, report = found
        sources[agent] = f"{path.name} (commit {report['meta'].get('git_commit', 'not recorded')})"
        results += agent_metrics(agent, read(report), prices)

    meta = {"reports": "; ".join(f"{a}: {s}" for a, s in sources.items()),
            "prices_usd_per_1m": (f"off-peak cache hit {prices.cache_hit}, cache miss {prices.cache_miss}, "
                                  f"output {prices.output}; peak x{prices.peak_multiplier} ({prices_source})")}
    _, md_path = write_report("system_health", results, cases=[], meta=meta)
    print_summary(results)
    print("\nSources:")
    for agent, source in sources.items():
        print(f"  {agent:<13} {source}")
    print(f"\nReport: {md_path}")


if __name__ == "__main__":
    main()
