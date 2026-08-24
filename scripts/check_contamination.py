#!/usr/bin/env python3
"""Check two JSONL datasets for contamination (overlap) and fail on any hit.

Primary use: gate a candidate TRAINING set against the frozen evaluation benchmark
BEFORE training, so leaked/paraphrased benchmark questions cannot inflate results
(PROJECT_PLAN.md §16.4). Also usable to self-check a benchmark against a previous
version (e.g. v2 vs v1) or to look for internal near-duplicates.

Usage:
    # Candidate training data vs the held-out benchmark:
    python scripts/check_contamination.py data/training/train_v0.1.jsonl \\
        data/evaluation/benchmark_v2.jsonl

    # Make sure benchmark v2 did not accidentally copy v1:
    python scripts/check_contamination.py data/evaluation/benchmark_v2.jsonl \\
        data/evaluation/benchmark_v1.jsonl

Exit status:
    0  clean          (no exact matches, no fuzzy matches >= threshold)
    1  contamination  (at least one match) -- suitable for CI / pre-train gating
    2  usage / IO error

Determinism: standard library only; identical inputs give identical output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly (python scripts/check_contamination.py) without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.data.contamination import (  # noqa: E402
    DEFAULT_FIELDS,
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_NGRAM,
    check_contamination,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} invalid JSON: {exc}") from exc
    return rows


def main() -> int:
    p = argparse.ArgumentParser(
        description="Detect contamination between two JSONL datasets (exact + fuzzy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Exit 1 on any match (exact or fuzzy >= --threshold); use for pre-train gating.",
    )
    p.add_argument("dataset_a", help="First dataset (e.g. candidate TRAINING data).")
    p.add_argument("dataset_b", help="Second dataset (e.g. the frozen EVALUATION benchmark).")
    p.add_argument(
        "--fields", default=",".join(DEFAULT_FIELDS),
        help="Comma-separated record fields to compare (missing fields are skipped).",
    )
    p.add_argument("--id-field", default="id", help="Record field used as the identifier.")
    p.add_argument("--ngram", type=int, default=DEFAULT_NGRAM, help="Word n-gram size for fuzzy match.")
    p.add_argument(
        "--threshold", type=float, default=DEFAULT_FUZZY_THRESHOLD,
        help="Fuzzy Jaccard similarity at/above which a pair is flagged (0-1).",
    )
    p.add_argument(
        "--max-fuzzy-print", type=int, default=25,
        help="Cap on how many fuzzy matches to print (all still count toward exit status).",
    )
    args = p.parse_args()

    path_a, path_b = Path(args.dataset_a), Path(args.dataset_b)
    for path in (path_a, path_b):
        if not path.is_file():
            print(f"ERROR: not a file: {path}", file=sys.stderr)
            return 2

    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip())
    try:
        a = _load_jsonl(path_a)
        b = _load_jsonl(path_b)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = check_contamination(
        a, b, id_field=args.id_field, fields=fields,
        ngram=args.ngram, threshold=args.threshold,
    )

    print(f"A: {path_a} ({report.a_count} records)")
    print(f"B: {path_b} ({report.b_count} records)")
    print(report.summary())

    if report.exact:
        print(f"\nExact matches ({len(report.exact)}):")
        for em in report.exact:
            print(f"  A[{em.a_id}] == B[{em.b_id}]")

    if report.fuzzy:
        print(f"\nFuzzy matches >= {args.threshold} ({len(report.fuzzy)}):")
        for fm in report.fuzzy[: args.max_fuzzy_print]:
            print(f"  A[{fm.a_id}] ~ B[{fm.b_id}]  similarity={fm.similarity}")
        if len(report.fuzzy) > args.max_fuzzy_print:
            print(f"  ... and {len(report.fuzzy) - args.max_fuzzy_print} more")

    if report.is_clean:
        print("\nOK: no contamination detected.")
        return 0
    print("\nFAIL: contamination detected (see matches above).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
