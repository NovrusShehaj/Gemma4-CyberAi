#!/usr/bin/env python3
"""Inspect a benchmark JSONL: print distributions and check quality requirements.

Programmatic (not visual) quality gate for Benchmark v2 (PROJECT_PLAN.md §17 /
TODO Benchmark v2). Prints counts by category, domain, scorer, difficulty, and
split; checks duplicate ids and near-duplicate questions; and verifies the v2
acceptance requirements. Exits non-zero if any requirement is violated, so it can
run in CI.

Usage:
    python scripts/inspect_benchmark.py data/evaluation/benchmark_v2.jsonl
    python scripts/inspect_benchmark.py data/evaluation/benchmark_v1.jsonl --no-require
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.data.contamination import jaccard_similarity, record_text  # noqa: E402
from gemma_cyber.evaluation.schema import load_benchmark  # noqa: E402

# v2 acceptance requirements (see configs/eval_success_criteria.md and the roadmap).
REQ_MIN_ITEMS = 100
REQ_MIN_PER_CATEGORY = 6
REQ_MIN_HALLUCINATION = 6
REQ_MIN_INSUFFICIENT = 6
REQ_TEST_RATIO_RANGE = (0.30, 0.50)  # ~40% held out
NEAR_DUP_THRESHOLD = 0.6


def _bar(counter: Counter) -> str:
    return "\n".join(f"    {k:26s} {v}" for k, v in sorted(counter.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect and validate a benchmark JSONL.")
    ap.add_argument("benchmark", help="Path to benchmark JSONL")
    ap.add_argument("--no-require", action="store_true",
                    help="Print stats only; do not enforce v2 acceptance requirements.")
    ap.add_argument("--near-dup-threshold", type=float, default=NEAR_DUP_THRESHOLD)
    args = ap.parse_args()

    items = load_benchmark(args.benchmark)  # also enforces unique ids + schema
    n = len(items)
    cats = Counter(i.category for i in items)
    domains = Counter(i.domain for i in items)
    scorers = Counter(i.scorer for i in items)
    diffs = Counter(i.difficulty for i in items)
    splits = Counter(i.split for i in items)

    print(f"Benchmark: {args.benchmark}")
    print(f"Total items: {n}")
    print(f"\nBy category ({len(cats)}):\n{_bar(cats)}")
    print(f"\nBy domain:\n{_bar(domains)}")
    print(f"\nBy scorer:\n{_bar(scorers)}")
    print(f"\nBy difficulty:\n{_bar(diffs)}")
    print(f"\nBy split:\n{_bar(splits)}")

    test_ratio = splits.get("test", 0) / n if n else 0.0
    print(f"\nTest ratio: {test_ratio:.3f}  (target ~0.40)")
    print("Per-category test/total:")
    for c in sorted(cats):
        ci = [i for i in items if i.category == c]
        t = sum(1 for i in ci if i.split == "test")
        print(f"    {c:26s} {t}/{len(ci)}")

    # Near-duplicate question scan (O(n^2) is fine at this size).
    records = [i.model_dump() for i in items]
    near = []
    for x in range(len(records)):
        for y in range(x + 1, len(records)):
            s = jaccard_similarity(record_text(records[x]), record_text(records[y]), 3)
            if s >= args.near_dup_threshold:
                near.append((records[x]["id"], records[y]["id"], round(s, 3)))
    print(f"\nNear-duplicate pairs (>= {args.near_dup_threshold}): "
          f"{near if near else 'none'}")

    missing_meta = [i.id for i in items if not (i.license and i.source and i.provenance)]
    print(f"Items missing license/source/provenance: {missing_meta or 'none'}")

    if args.no_require:
        return 0

    # Enforce requirements.
    problems: list[str] = []
    if n < REQ_MIN_ITEMS:
        problems.append(f"only {n} items (need >= {REQ_MIN_ITEMS})")
    low = {c: v for c, v in cats.items() if v < REQ_MIN_PER_CATEGORY}
    if low:
        problems.append(f"categories below {REQ_MIN_PER_CATEGORY}: {low}")
    if cats.get("hallucination", 0) < REQ_MIN_HALLUCINATION:
        problems.append(f"hallucination items {cats.get('hallucination', 0)} < {REQ_MIN_HALLUCINATION}")
    if cats.get("insufficient_evidence", 0) < REQ_MIN_INSUFFICIENT:
        problems.append(f"insufficient_evidence items {cats.get('insufficient_evidence', 0)} < {REQ_MIN_INSUFFICIENT}")
    lo, hi = REQ_TEST_RATIO_RANGE
    if not (lo <= test_ratio <= hi):
        problems.append(f"test ratio {test_ratio:.3f} outside {REQ_TEST_RATIO_RANGE}")
    if near:
        problems.append(f"near-duplicate question pairs present: {near}")
    if missing_meta:
        problems.append(f"items missing license/source/provenance: {missing_meta}")

    if problems:
        print("\nFAIL — requirement violations:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK — all v2 acceptance requirements satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
