#!/usr/bin/env python3
"""Seed data/models/registry.json with the project's honest, current model state.

Idempotent: re-running overwrites the two known entries but keeps the file valid.
The registry reflects REALITY, not aspiration:

  * ``gemma3:4b`` — the frozen base/reference model, stage ``evaluated`` (it has a
    recorded baseline scorecard). It is the anchor we try to beat, not a promoted
    product, so ``passed_eval=False`` and it is not in ``production``.
  * ``gemma3-cyber:v0.2`` — the exp-002 candidate, stage ``experimental``. Per the
    project record its improvement is UNPROVEN (trained once, artifacts lost, no
    eval). It stays experimental until a scorecard clears exp-002's success
    criteria; only then does ``mark_evaluated(passed=True)`` gate a promotion.

Run:  python scripts/seed_registry.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.inference.config import DEFAULT_REGISTRY_PATH  # noqa: E402
from gemma_cyber.inference.registry import ModelRecord, ModelRegistry  # noqa: E402


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def main() -> int:
    reg = ModelRegistry(DEFAULT_REGISTRY_PATH)
    commit = _git_commit()

    reg.register(
        ModelRecord(
            version="gemma3:4b",
            stage="evaluated",
            ollama_tag="gemma3:4b",
            base_model="gemma3:4b",
            dataset_version=None,
            passed_eval=False,
            eval_ref="experiments/baseline_gemma3-4b_v2/scorecard.md",
            notes="Frozen base/reference (the anchor to beat). Baseline pass_rate 0.84.",
        ),
        overwrite=True,
    )
    reg.register(
        ModelRecord(
            version="gemma3-cyber:v0.2",
            stage="experimental",
            ollama_tag="gemma3-cyber:v0.2",
            base_model="gemma3:4b",
            dataset_version="sft_v0.2",
            git_commit=commit,
            experiment="exp-002-gemma3-cyber-v0.2",
            passed_eval=False,
            eval_ref=None,
            notes=(
                "exp-002 candidate. Improvement UNPROVEN (trained once on Colab, "
                "artifacts lost, no evaluation run). Stays experimental until a "
                "scorecard clears docs/experiments/exp-002.md success criteria."
            ),
        ),
        overwrite=True,
    )

    print(f"Seeded registry at {reg.path}")
    for rec in reg.list():
        print(f"  {rec.version:24s} [{rec.stage}]  passed_eval={rec.passed_eval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
