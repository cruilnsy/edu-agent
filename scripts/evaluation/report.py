# scripts/evaluation/report.py
# Writes one eval run as JSON (for comparing runs) and Markdown (for reading).
# Output goes to eval_results/<eval name>/, which is git-ignored.

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "eval_results"


@dataclass
class Metric:
    name: str
    value: float | None          # None or NaN means "not computed", e.g. labels still pending
    target: str = ""             # Human-readable goal, e.g. "<= 8"
    passed: bool | None = None   # None when there is no target or no value
    note: str = ""

    def display_value(self) -> str:
        if self.value is None or (isinstance(self.value, float) and math.isnan(self.value)):
            return "n/a"
        return f"{self.value:.3f}" if isinstance(self.value, float) else str(self.value)


def check(name: str, value: float | None, *, max_value: float | None = None,
          min_value: float | None = None, note: str = "") -> Metric:
    """Build a Metric and decide pass/fail against an upper or lower bound."""
    missing = value is None or (isinstance(value, float) and math.isnan(value))
    if max_value is not None:
        target, passed = f"<= {max_value}", None if missing else value <= max_value
    elif min_value is not None:
        target, passed = f">= {min_value}", None if missing else value >= min_value
    else:
        target, passed = "", None
    return Metric(name, None if missing else value, target, passed, note)


def git_commit() -> str:
    """Short commit of the code being evaluated, with "-dirty" if there are uncommitted changes."""
    try:
        def git(*args: str) -> str:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                                  check=True, timeout=10).stdout.strip()
        commit = git("rev-parse", "--short", "HEAD")
        return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_report(eval_name: str, metrics: list[Metric], cases: list[dict], meta: dict,
                 requests: list[dict] | None = None) -> tuple[Path, Path]:
    """Save the run and return (json_path, markdown_path).

    requests: optional per-request usage (latency_s, llm_calls, tokens, fallback) for evals whose
    cases are not one request each; eval_system_health.py reads it."""
    meta = {**meta, "git_commit": git_commit()}
    out_dir = RESULTS_DIR / eval_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = out_dir / f"{stamp}.json"
    data = {"eval": eval_name, "meta": meta, "metrics": [asdict(m) for m in metrics], "cases": cases}
    if requests is not None:
        data["requests"] = requests
    json_path.write_text(json.dumps(data, indent=2, default=str))

    md_path = out_dir / f"{stamp}.md"
    md_path.write_text(_markdown(eval_name, metrics, meta))
    return json_path, md_path


def _markdown(eval_name: str, metrics: list[Metric], meta: dict) -> str:
    status = {True: "pass", False: "FAIL", None: "-"}
    lines = [f"# {eval_name} eval", ""]
    lines += [f"- **{k}**: {v}" for k, v in meta.items()]
    lines += ["", "| Metric | Value | Target | Result | Note |", "|---|---|---|---|---|"]
    lines += [f"| {m.name} | {m.display_value()} | {m.target} | {status[m.passed]} | {m.note} |" for m in metrics]
    return "\n".join(lines) + "\n"


def print_summary(metrics: list[Metric]) -> None:
    width = max(len(m.name) for m in metrics)
    for m in metrics:
        result = {True: "pass", False: "FAIL", None: ""}[m.passed]
        print(f"  {m.name:<{width}}  {m.display_value():>8}  {m.target:<8} {result}  {m.note}")
