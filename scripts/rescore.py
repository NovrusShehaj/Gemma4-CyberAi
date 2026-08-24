#!/usr/bin/env python3
"""Re-score an existing results.json against (possibly improved) scorers.

Does NOT call the model — reuses stored deterministic responses.

Usage:
    python scripts/rescore.py experiments/baseline_gemma3-4b/results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.evaluation.harness import rescore_results  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Re-score stored results.")
    p.add_argument("results", help="Path to results.json")
    p.add_argument("--benchmark", default="data/evaluation/benchmark_v1.jsonl")
    args = p.parse_args()

    report = rescore_results(args.results, args.benchmark)
    o = report["overall"]
    print(f"Re-scored -> {Path(args.results).parent}/")
    print(f"Overall: pass_rate={o['pass_rate']}  mean_score={o['mean_score']}  (n={o['count']})")
    for cat, a in report["by_category"].items():
        print(f"  {cat:24s} pass={a['pass_rate']:<6} mean={a['mean_score']:<6} (n={a['count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
