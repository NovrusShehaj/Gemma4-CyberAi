#!/usr/bin/env python3
"""Run the benchmark harness against a local MLX model (base or fused LoRA).

Mirror of scripts/run_baseline.py but with an MLX client instead of Ollama, so a
fused adapter can be evaluated on Apple Silicon without a GGUF conversion step.
Same harness, same deterministic scorers, same output layout.

    python scripts/run_baseline_mlx.py --model mlx-community/gemma-3-4b-it-4bit \
        --benchmark data/evaluation/benchmark_v2.jsonl --split test \
        --out experiments/exp-002r-gemma3-cyber-v0.2/base-v2-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.evaluation.harness import run_benchmark  # noqa: E402
from gemma_cyber.evaluation.mlx_client import MlxClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="MLX model path or HF repo id")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--split", choices=["dev", "test"], default=None)
    args = ap.parse_args()

    print(f"Loading MLX model {args.model} ...", flush=True)
    client = MlxClient(args.model)
    print(f"Running '{args.benchmark}' [split={args.split}] ...", flush=True)
    report = run_benchmark(
        client, args.benchmark, args.out, seed=args.seed,
        num_predict=args.num_predict, split=args.split,
    )
    o = report["overall"]
    print(f"\nDone in {report['duration_seconds']}s -> {args.out}/")
    print(f"Overall: pass_rate={o['pass_rate']}  mean_score={o['mean_score']}  (n={o['count']})")
    for cat, a in report["by_category"].items():
        print(f"  {cat:24s} pass={a['pass_rate']:<6} mean={a['mean_score']:<6} (n={a['count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
