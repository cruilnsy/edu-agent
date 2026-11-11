# scripts/evaluation/run_all.py
# Runs every eval, then the system health eval, then writes the summary table.
# Each eval runs in its own process, so one failure (e.g. Milvus down, labels missing) is
# recorded and the rest still run.
#
# Run from the project root (needs the DeepSeek key, Milvus for QA, the labels for exam):
#     python -m scripts.evaluation.run_all
#     python -m scripts.evaluation.run_all --only classifier orchestrator
#     python -m scripts.evaluation.run_all --skip interview --price-cache-miss 0.15
#
# Without arguments this takes a while and makes many LLM calls: resume runs each resume twice,
# interview simulates five full interviews.

import argparse
import subprocess
import sys
import time

from scripts.evaluation.report import PROJECT_ROOT
from scripts.evaluation.summary import write_summary

# Cheapest first, so a broken setup shows up before the long runs
EVALS = ("classifier", "orchestrator", "qa", "exam", "resume", "interview")


def run_eval(name: str, extra_args: list[str]) -> dict:
    print(f"\n{'=' * 20} {name} {'=' * 20}", flush=True)
    start = time.perf_counter()
    try:
        code = subprocess.run([sys.executable, "-m", f"scripts.evaluation.eval_{name}", *extra_args],
                              cwd=PROJECT_ROOT).returncode
    except KeyboardInterrupt:
        code = "interrupted"
    seconds = time.perf_counter() - start
    result = "ok" if code == 0 else f"failed (exit {code})"
    print(f"-- {name}: {result} in {seconds:.0f}s", flush=True)
    return {"eval": name, "result": result, "seconds": seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every eval and write the summary.")
    parser.add_argument("--only", nargs="+", choices=EVALS, help="Run just these evals")
    parser.add_argument("--skip", nargs="+", choices=EVALS, default=[], help="Do not run these evals")
    for price in ("cache-hit", "cache-miss", "out"):
        parser.add_argument(f"--price-{price}", type=float,
                            help="Off-peak USD per 1M tokens, overrides the default in pricing.py")
    args = parser.parse_args()

    selected = [e for e in EVALS if (not args.only or e in args.only) and e not in args.skip]
    runs = [run_eval(name, []) for name in selected]

    price_args = []
    for price in ("cache_hit", "cache_miss", "out"):
        value = getattr(args, f"price_{price}")
        if value is not None:
            price_args += [f"--price-{price.replace('_', '-')}", str(value)]
    runs.append(run_eval("system_health", price_args))

    md_path = write_summary(runs)
    print(f"\n{md_path.read_text()}\nSummary: {md_path}")
    sys.exit(0 if all(r["result"] == "ok" for r in runs) else 1)


if __name__ == "__main__":
    main()
