#!/usr/bin/env python3
"""Run the frozen benchmark against a model via Ollama and write a scorecard.

Usage:
    python scripts/run_baseline.py                       # base gemma3:4b
    python scripts/run_baseline.py --model gemma3-cyber:v0.1 --out experiments/exp-001

This is the first coding milestone's entrypoint (PROJECT_PLAN.md §23). It
establishes the mandatory baseline BEFORE any fine-tuning (§18).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly (python scripts/run_baseline.py) without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.clients import OllamaClient  # noqa: E402
from gemma_cyber.evaluation.harness import run_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the cyber benchmark via Ollama.")
    parser.add_argument("--model", default="gemma3:4b", help="Ollama model tag")
    parser.add_argument(
        "--benchmark", default="data/evaluation/benchmark_v1.jsonl",
        help="Path to benchmark JSONL",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output experiment dir (default: experiments/baseline_<model>)",
    )
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-predict", type=int, default=512)
    args = parser.parse_args()

    out = args.out or f"experiments/baseline_{args.model.replace(':', '-').replace('/', '-')}"

    client = OllamaClient(model=args.model, host=args.host)
    if not client.is_available():
        print(f"ERROR: Ollama not reachable at {args.host}. Is `ollama serve` running?",
              file=sys.stderr)
        return 2
    if not client.has_model(args.model):
        print(f"ERROR: model '{args.model}' not found locally. Try: ollama pull {args.model}",
              file=sys.stderr)
        return 3

    print(f"Running benchmark '{args.benchmark}' against '{args.model}' ...")
    report = run_benchmark(
        client, args.benchmark, out, seed=args.seed, num_predict=args.num_predict,
    )
    o = report["overall"]
    print(f"\nDone in {report['duration_seconds']}s -> {out}/")
    print(f"Overall: pass_rate={o['pass_rate']}  mean_score={o['mean_score']}  (n={o['count']})")
    print("By category:")
    for cat, a in report["by_category"].items():
        print(f"  {cat:24s} pass={a['pass_rate']:<6} mean={a['mean_score']:<6} (n={a['count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
