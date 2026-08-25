#!/usr/bin/env python3
"""Validate a training dataset JSONL against the Pydantic TrainingItem schema.

Checks:
  * Schema conformity and valid JSON on every line.
  * No duplicate item IDs.
  * Required metadata: task_type, domain, difficulty, provenance, license.
  * Message structure: system/user/assistant sequence ending in assistant.
  * Optional contamination check against evaluation benchmark splits.

Usage:
    python scripts/validate_dataset.py --dataset data/training/sft_v0.1.jsonl
    python scripts/validate_dataset.py --dataset data/training/sft_v0.1.jsonl \
        --check-contamination data/evaluation/benchmark_v2.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.data.contamination import check_contamination, jaccard_similarity  # noqa: E402
from gemma_cyber.data.schema import TrainingItem, load_training_dataset  # noqa: E402
from gemma_cyber.evaluation.schema import load_benchmark  # noqa: E402


def assistant_uniqueness(items: list[TrainingItem]) -> dict:
    """Measure answer-level diversity, the metric v0.1 failed (91 unique of 360).

    Returns exact-unique-answer count and a list of near-duplicate answer pairs
    (word-trigram Jaccard >= 0.9) that indicate superficial copy-paste variation.
    """
    answers = [
        next((m.content for m in it.messages if m.role == "assistant"), "")
        for it in items
    ]
    ids = [it.id for it in items]
    exact_unique = len({" ".join(a.split()).lower() for a in answers})
    near_dups: list[tuple[str, str, float]] = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            sim = jaccard_similarity(answers[i], answers[j], n=3)
            if sim >= 0.9:
                near_dups.append((ids[i], ids[j], round(sim, 3)))
    return {
        "total": len(items),
        "exact_unique_answers": exact_unique,
        "unique_ratio": round(exact_unique / len(items), 3) if items else 0.0,
        "near_duplicate_pairs": near_dups,
    }


def summarize_dataset(items: list[TrainingItem]) -> dict:
    """Compute dataset composition statistics."""
    task_counts = Counter(it.metadata.task_type for it in items)
    domain_counts = Counter(it.metadata.domain for it in items)
    diff_counts = Counter(it.metadata.difficulty for it in items)
    fab_count = sum(1 for it in items if it.metadata.fabricated_premise)
    req_evi_count = sum(1 for it in items if it.metadata.requires_evidence)
    licenses = Counter(it.metadata.license for it in items)

    return {
        "total": len(items),
        "by_task": dict(task_counts.most_common()),
        "by_domain": dict(domain_counts.most_common()),
        "by_difficulty": dict(diff_counts.most_common()),
        "fabricated_premise_count": fab_count,
        "requires_evidence_count": req_evi_count,
        "licenses": dict(licenses.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate training dataset JSONL.")
    parser.add_argument("--dataset", "-d", required=True, help="Path to training dataset JSONL")
    parser.add_argument(
        "--check-contamination",
        "-c",
        nargs="*",
        default=[],
        help="One or more evaluation benchmark JSONL paths to check contamination against",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Contamination fuzzy threshold (default: 0.50)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset file '{dataset_path}' does not exist.", file=sys.stderr)
        return 2

    print(f"Validating dataset: {dataset_path}")
    try:
        items = load_training_dataset(dataset_path)
    except ValueError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    stats = summarize_dataset(items)
    print("\nDataset Summary:")
    print(f"  Total items: {stats['total']}")
    print("  By Domain:")
    for dom, count in stats["by_domain"].items():
        pct = (count / stats["total"]) * 100
        print(f"    - {dom}: {count} ({pct:.1f}%)")
    print("  By Task Type:")
    for task, count in stats["by_task"].items():
        pct = (count / stats["total"]) * 100
        print(f"    - {task}: {count} ({pct:.1f}%)")
    print("  By Difficulty:")
    for diff, count in stats["by_difficulty"].items():
        pct = (count / stats["total"]) * 100
        print(f"    - {diff}: {count} ({pct:.1f}%)")
    print(f"  Fabricated Premise (Traps): {stats['fabricated_premise_count']}")
    print(f"  Requires Evidence: {stats['requires_evidence_count']}")
    print(f"  Licenses: {stats['licenses']}")

    # Answer-level diversity (the metric v0.1 failed: 91 unique answers of 360).
    uniq = assistant_uniqueness(items)
    print("\nDiversity (assistant answers):")
    print(
        f"  Exact-unique answers: {uniq['exact_unique_answers']}/{uniq['total']} "
        f"(ratio {uniq['unique_ratio']})"
    )
    if uniq["near_duplicate_pairs"]:
        print(f"  WARNING: {len(uniq['near_duplicate_pairs'])} near-duplicate answer "
              f"pair(s) (Jaccard >= 0.9):")
        for a_id, b_id, sim in uniq["near_duplicate_pairs"][:10]:
            print(f"    {a_id} ~ {b_id} ({sim})")
    else:
        print("  No near-duplicate answer pairs (Jaccard >= 0.9).")

    # Contamination check if requested
    if args.check_contamination:
        print("\nRunning Contamination Checks:")
        # Convert training items to record dicts for contamination checker
        train_records = [
            {
                "id": it.id,
                "question": next(
                    (m.content for m in it.messages if m.role == "user"), ""
                ),
                "context": next(
                    (m.content for m in it.messages if m.role == "system"), ""
                ),
                "evidence": "",
            }
            for it in items
        ]

        has_contamination = False
        for eval_path_str in args.check_contamination:
            eval_path = Path(eval_path_str)
            if not eval_path.exists():
                print(f"WARNING: eval file '{eval_path}' not found, skipping.", file=sys.stderr)
                continue

            eval_items = load_benchmark(eval_path)
            eval_records = [
                {
                    "id": it.id,
                    "question": it.question,
                    "context": it.context or "",
                    "evidence": it.evidence or "",
                    "choices": it.choices,
                }
                for it in eval_items
            ]

            report = check_contamination(
                train_records, eval_records, threshold=args.threshold
            )
            print(f"  vs {eval_path.name} ({len(eval_items)} items): {report.summary()}")

            if not report.is_clean:
                has_contamination = True
                if report.exact:
                    print(f"    EXACT MATCHES ({len(report.exact)}):")
                    for em in report.exact[:5]:
                        print(f"      Train[{em.a_id}] == Eval[{em.b_id}]")
                if report.fuzzy:
                    print(f"    FUZZY MATCHES >= {args.threshold} ({len(report.fuzzy)}):")
                    for fm in report.fuzzy[:5]:
                        print(f"      Train[{fm.a_id}] ~ Eval[{fm.b_id}] (similarity: {fm.similarity:.3f})")

        if has_contamination:
            print("\nFAIL: Contamination detected against evaluation benchmarks!", file=sys.stderr)
            return 1
        print("\nContamination Check: ALL CLEAN (0 exact, 0 fuzzy >= threshold)")

    print("\nSUCCESS: Dataset schema validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
