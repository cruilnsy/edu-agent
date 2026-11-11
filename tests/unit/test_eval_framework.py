# tests/unit/test_eval_framework.py
# Metric math, usage tracking and report writing used by scripts/evaluation.

import json
import math

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from scripts.evaluation import metrics as m
from scripts.evaluation import report
from scripts.evaluation import pricing
from scripts.evaluation.pricing import Prices, load_prices, request_cost
from scripts.evaluation.tracking import UsageTracker


def test_recall_at_k_counts_questions_with_a_gold_hit():
    cases = [
        (["c3", "c15", "c21"], {"c15"}),   # Hit at rank 2
        (["c1", "c2", "c9"], {"c9"}),      # Hit at rank 3
        (["c4", "c5", "c6"], {"c7"}),      # Miss
    ]
    assert m.recall_at_k(cases, k=3) == pytest.approx(2 / 3)
    assert m.recall_at_k(cases, k=1) == 0
    assert math.isnan(m.recall_at_k([], k=3))


def test_classification_metrics():
    gold = ["qa", "qa", "exam", "resume"]
    pred = ["qa", "exam", "exam", "resume"]
    assert m.accuracy(pred, gold) == 0.75
    assert 0 < m.macro_f1(pred, gold) < 1
    assert m.confusion(pred, gold, ["qa", "exam", "resume"]) == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ValueError):
        m.accuracy(["qa"], gold)


def test_mae_matches_the_course_example():
    # Expert 80/85/78 vs agent 84/82/80 -> errors 4/3/2 -> MAE 3
    assert m.mae([84, 82, 80], [80, 85, 78]) == pytest.approx(3.0)


def test_levels_and_kappa():
    assert [m.to_level(s, 10) for s in (5, 7, 9)] == ["low", "mid", "high"]
    assert m.cohen_kappa(["low", "mid", "high"], ["low", "mid", "high"]) == pytest.approx(1.0)
    assert math.isnan(m.cohen_kappa(["mid", "mid"], ["mid", "mid"]))


def test_health_metrics():
    assert m.percentile(list(range(1, 101)), 95) == pytest.approx(95.05)
    assert m.rate([True, False, False, False]) == 0.25
    assert m.spread([71.5, 74.0, 72.0]) == pytest.approx(2.5)
    assert m.ratio(4, 5) == 0.8 and math.isnan(m.ratio(1, 0))


def test_usage_tracker_sums_tokens_and_cache_hits():
    tracker = UsageTracker()
    cold = AIMessage(content="ok", usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                                                   "input_token_details": {"cache_read": 0}})
    warm = AIMessage(content="ok", usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                                                   "input_token_details": {"cache_read": 64}})
    for message in (cold, warm):
        tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    snap = tracker.snapshot()
    assert (snap["llm_calls"], snap["input_tokens"], snap["cached_input_tokens"], snap["output_tokens"]) == (2, 200, 64, 40)


def test_usage_tracker_reads_deepseek_cache_field_without_usage_metadata():
    tracker = UsageTracker()
    usage = {"prompt_tokens": 50, "completion_tokens": 5, "prompt_cache_hit_tokens": 32}
    tracker.on_llm_end(LLMResult(generations=[[]], llm_output={"token_usage": usage}))
    assert tracker.snapshot()["cached_input_tokens"] == 32


def test_report_writes_json_and_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    metrics = [
        report.check("score_mae", 3.8, max_value=8),
        report.check("issue_recall", math.nan, min_value=0.7, note="labels pending"),
    ]
    assert [x.passed for x in metrics] == [True, None]

    json_path, md_path = report.write_report("demo", metrics, cases=[{"id": "r01"}], meta={"cases": 1})
    data = json.loads(json_path.read_text())
    assert data["metrics"][0]["value"] == 3.8 and data["metrics"][1]["value"] is None
    assert "| score_mae | 3.800 | <= 8 | pass |" in md_path.read_text()


def test_report_records_the_commit_and_optional_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(report, "git_commit", lambda: "abc1234-dirty")
    json_path, _ = report.write_report("demo", [], cases=[], meta={}, requests=[{"latency_s": 1.0}])
    data = json.loads(json_path.read_text())
    assert data["meta"]["git_commit"] == "abc1234-dirty" and data["requests"] == [{"latency_s": 1.0}]


# ── System health ─────────────────────────────────────────────

def _turn(stage_after, latency):
    return {"stage_after": stage_after, "latency_s": latency, "llm_calls": 1, "input_tokens": 100, "output_tokens": 10}


def test_health_splits_interview_turns_and_reads_fallbacks():
    from scripts.evaluation import eval_system_health as health
    case = {"start": _turn(None, 4.0), "fallback_used": False,
            "turns": [_turn("tech_base", 2.0), _turn("tech_base", 3.0), _turn("finished", 9.0)]}
    result = health.interview_health({"cases": [case, {**case, "fallback_used": True}]})
    by_name = {s.name: s for s in result.series}
    assert by_name["interview_turn"].latencies_s == [2.0, 3.0, 2.0, 3.0]
    assert by_name["interview_report_turn"].latencies_s == [9.0, 9.0]
    assert result.fallback_flags == [False, True]


def test_health_uses_old_exam_metrics_when_raw_requests_are_missing():
    from scripts.evaluation import eval_system_health as health
    old = {"metrics": [{"name": "latency_p95_s", "value": 5.5}, {"name": "tokens_per_submission", "value": 5000},
                       {"name": "fallback_rate", "value": 0.0}]}
    names = [m.name for m in health.agent_metrics("exam", health.exam_health(old), Prices())]
    assert names == ["exam_submission.latency_p95_s", "exam_submission.tokens_per_request", "exam.fallback_rate"]
    new = {"requests": [{"latency_s": 4.0, "llm_calls": 3, "input_tokens": 10, "output_tokens": 5, "fallback": True}]}
    found = {m.name: m for m in health.agent_metrics("exam", health.exam_health(new), Prices())}
    assert found["exam.fallback_rate"].value == 1.0 and found["exam.fallback_rate"].passed is False


def test_cost_prices_cache_hits_separately_and_doubles_at_peak():
    prices = Prices(cache_hit=0.01, cache_miss=1.0, output=2.0)
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 600_000, "output_tokens": 500_000}
    assert request_cost(usage, prices) == pytest.approx(0.006 + 0.4 + 1.0)
    assert request_cost(usage, prices, peak=True) == pytest.approx(2 * 1.406)
    old = {"input_tokens": 1_000_000, "cached_input_tokens": None, "output_tokens": 0}   # Report without cache data
    assert request_cost(old, prices) == pytest.approx(1.0)


def test_prices_come_from_cli_then_env_then_defaults(monkeypatch):
    for name in pricing._ENV.values():
        monkeypatch.delenv(name, raising=False)
    assert load_prices() == (Prices(), pricing.PRICES_SOURCE)
    monkeypatch.setenv("EVAL_PRICE_OUTPUT_PER_1M", "9")
    prices, source = load_prices(cache_miss=0.5)
    assert (prices.cache_hit, prices.cache_miss, prices.output, source) == (0.003, 0.5, 9.0, "custom prices")


def test_health_cache_hit_rate_and_upper_bound_note():
    from scripts.evaluation import eval_system_health as health
    usage = [{"llm_calls": 1, "input_tokens": 100, "cached_input_tokens": 25, "output_tokens": 10}]
    found = {m.name: m for m in health.series_metrics(health.Series("x", "req", [1.0], usage), Prices())}
    assert found["x.cache_hit_rate"].value == 0.25 and found["x.cost_usd_per_request"].note == ""
    old = [{**usage[0], "cached_input_tokens": None}]
    found = {m.name: m for m in health.series_metrics(health.Series("x", "req", [1.0], old), Prices())}
    assert found["x.cache_hit_rate"].value is None and "upper bound" in found["x.cost_usd_per_request"].note


def test_health_picks_the_latest_report(tmp_path):
    from scripts.evaluation import eval_system_health as health
    (tmp_path / "qa").mkdir()
    for stamp in ("20261101-000000", "20261102-000000"):
        (tmp_path / "qa" / f"{stamp}.json").write_text(json.dumps({"stamp": stamp}))
    path, data = health.latest_report("qa", tmp_path)
    assert data["stamp"] == "20261102-000000"
    assert health.latest_report("resume", tmp_path) is None


def _report(commit, metrics):
    return json.dumps({"meta": {"git_commit": commit}, "metrics": metrics})


def test_summary_keeps_targets_and_extras_and_leaves_latency_to_health():
    from scripts.evaluation import summary
    rows = [{"name": "accuracy", "value": 1.0, "target": ">= 0.9", "passed": True},
            {"name": "latency_p95_llm_s", "value": 1.4, "target": "<= 3", "passed": True},
            {"name": "recall_qa", "value": 1.0, "target": "", "passed": None}]
    assert [m.name for m in summary.core_metrics("orchestrator", {"metrics": rows})] == ["accuracy"]
    rows = [{"name": "held_out_accuracy", "value": 0.9}, {"name": "held_out_confusion", "value": None}]
    assert [m.name for m in summary.core_metrics("classifier", {"metrics": rows})] == ["held_out_accuracy"]


def test_summary_lays_out_health_as_one_row_per_request():
    from scripts.evaluation import summary
    rows = [{"name": "qa_answer.latency_p50_s", "value": 4.0},
            {"name": "qa_answer.latency_p95_s", "value": 7.0, "target": "<= 8", "passed": True},
            {"name": "qa_answer.input_tokens_per_request", "value": 800.0},
            {"name": "qa_answer.output_tokens_per_request", "value": 300.0},
            {"name": "qa.fallback_rate", "value": 0.0, "target": "<= 0.05", "passed": True, "note": "23 answers"}]
    lines = summary._health_lines(summary.core_metrics("system_health", {"metrics": rows}))
    assert "| qa_answer | 4.000 | 7.000 | <= 8 | pass | n/a | 800.000 / 300.000 | n/a | n/a |" in lines
    assert "| qa | 0.000 | <= 0.05 | pass | 23 answers |" in lines


def test_summary_marks_missing_and_older_reports(tmp_path):
    from scripts.evaluation import summary
    (tmp_path / "orchestrator").mkdir()
    (tmp_path / "orchestrator" / "20261110-000000.json").write_text(
        _report("abc1234-dirty", [{"name": "accuracy", "value": 0.5, "target": ">= 0.9", "passed": False}]))
    (tmp_path / "classifier").mkdir()
    (tmp_path / "classifier" / "20261110-000000.json").write_text(_report("old0000", []))
    sections = {s.eval_name: s for s in summary.build_sections(tmp_path, head="abc1234")}
    assert "older code" not in sections["orchestrator"].source
    assert "older code" in sections["classifier"].source
    assert sections["qa"].source.startswith("missing")
    md = summary.to_markdown(list(sections.values()), "abc1234", [{"eval": "qa", "result": "ok", "seconds": 3.0}])
    assert "1 fail" in md and "**failing**: orchestrator.accuracy" in md and "| qa | ok | 3 |" in md


def test_resume_gap_check_finds_employment_gaps_by_priority():
    from scripts.evaluation.eval_resume import gap_issues
    issues = [{"priority": "medium", "description": "There is an 11-month gap between the internship and the job."},
              {"priority": "low", "description": "A two-year unexplained gap follows the last role."},
              {"priority": "high", "description": "No repo link: the biggest credibility gap for a new grad."},
              {"priority": "high", "description": "A skills gap between the summary and the experience."}]
    assert gap_issues(issues, "medium") == [issues[0]["description"]]
    assert gap_issues(issues, "low") == [issues[1]["description"]]
    assert gap_issues(issues, "high") == []


def test_resume_fairness_check_finds_protected_traits():
    from scripts.evaluation.eval_resume import protected_trait_mentions
    texts = ["Clarify your visa status for US roles.", "Add metrics to the Average-latency bullet.",
             "Mention your page-load improvements.", "State your age range.",
             "Used Lua scripts to avoid race conditions.", "The intern role is 3+ years old.",
             'Add "Authorized to work in the U.S."',
             'If applicable, add "Authorized to work in the U.S.; no sponsorship required."']
    assert protected_trait_mentions(texts) == [texts[0], texts[3], texts[6]]   # The optional line is allowed
