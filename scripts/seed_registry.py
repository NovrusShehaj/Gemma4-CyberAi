#!/usr/bin/env python3
"""Seed data/models/registry.json with the project's honest, current model state.

By default this script **refuses to overwrite** a version that already has an
``eval_ref`` or ``history`` (measured experiments). Pass ``--force`` only when
you intend to replace those records. ``passed_eval`` stays false; this script
never promotes.

The registry reflects REALITY, not aspiration:

  * ``gemma3:4b`` — frozen base/reference, stage ``evaluated`` (baseline
    scorecard exists). ``passed_eval=False``; not in ``production``.
  * ``gemma3-cyber:v0.2`` — exp-002r candidate of record (ep3 fused SHA).
    Stage ``experimental``. MEASURED fail: v3 attack_mapping 0.000 and the
    Kerberoasting T1060 trap still fails. Stays experimental.

Run:  python scripts/seed_registry.py
      python scripts/seed_registry.py --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.inference.config import DEFAULT_REGISTRY_PATH  # noqa: E402
from gemma_cyber.inference.errors import RegistryError  # noqa: E402
from gemma_cyber.inference.registry import ModelRecord, ModelRegistry  # noqa: E402

# Candidate of record = exp-002r end-of-epoch-3 fused MLX 4-bit weights.
EXP002R_EP3_FUSED_SHA256 = (
    "1ea70da8a68526b1abbffb1eff6738319f961550348ee90ecea0bedd497ef702"
)

_SEED_VERSIONS = ("gemma3:4b", "gemma3-cyber:v0.2")

_V02_NOTES = (
    "exp-002r (2026-08-27) candidate of record = end-of-epoch-3 fused MLX 4-bit "
    "(SHA 1ea70da8…). MEASURED: DOES NOT PASS. v2 test overall 0.956 "
    "(bar ≥ 0.913) PASS; v2 hallucination n=3 0.333 (Δ 0 vs MLX base); "
    "v3 attack_mapping 0.000 vs base 0.250 FAIL; Kerberoasting trap still "
    "emits T1060 (correct is T1558.003 / TA0006). Stays experimental. "
    "Do not promote. Next: exp-003 / sft_v0.3."
)

_BASE_NOTES = (
    "Frozen base/reference (the APP-GATE serve default and the anchor to beat). "
    "Ollama gemma3:4b benchmark_v2 test pass_rate 0.933; hallucination traps "
    "0.000 (n=3). Not a specialized cyber model; not promoted to production. "
    "See docs/model-card.md."
)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def _protected_versions(reg: ModelRegistry) -> list[str]:
    blocked: list[str] = []
    for version in _SEED_VERSIONS:
        try:
            rec = reg.get(version)
        except RegistryError:
            continue
        if rec.eval_ref or rec.history:
            blocked.append(version)
    return blocked


def seed_registry(
    path: Path,
    *,
    force: bool = False,
    commit: str | None = None,
) -> int:
    """Write the two known records. Return 0 on success, 1 if refused."""
    path = Path(path)
    if path.exists() and not force:
        existing = ModelRegistry(path)
        blocked = _protected_versions(existing)
        if blocked:
            print(
                "Refusing to overwrite measured registry records "
                f"{blocked} at {path}. Re-run with --force if you intend to "
                "replace eval_ref/history.",
                file=sys.stderr,
            )
            return 1

    reg = ModelRegistry(path)
    if commit is None:
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
            notes=_BASE_NOTES,
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
            fused_model_sha256=EXP002R_EP3_FUSED_SHA256,
            experiment="exp-002r-gemma3-cyber-v0.2",
            passed_eval=False,
            eval_ref="experiments/exp-002r-gemma3-cyber-v0.2/RESULTS.md",
            notes=_V02_NOTES,
        ),
        overwrite=True,
    )

    print(f"Seeded registry at {reg.path}")
    for rec in reg.list():
        print(f"  {rec.version:24s} [{rec.stage}]  passed_eval={rec.passed_eval}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Seed the model registry with gemma3:4b and gemma3-cyber:v0.2. "
            "Does not clobber records that already have eval_ref or history "
            "unless --force is passed. Never sets passed_eval=true."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing records that have eval_ref or history.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Registry JSON path (default: data/models/registry.json).",
    )
    args = parser.parse_args(argv)
    return seed_registry(args.path, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
