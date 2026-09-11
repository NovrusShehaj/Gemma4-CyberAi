"""seed_registry.py refuses to clobber measured records unless --force."""

from __future__ import annotations

import sys
from pathlib import Path

from gemma_cyber.inference.registry import ModelRecord, ModelRegistry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from seed_registry import seed_registry  # noqa: E402


def test_seed_writes_empty_path(tmp_path):
    path = tmp_path / "registry.json"
    assert seed_registry(path, force=False, commit="abc") == 0
    reg = ModelRegistry(path)
    base = reg.get("gemma3:4b")
    v02 = reg.get("gemma3-cyber:v0.2")
    assert base.passed_eval is False and base.stage == "evaluated"
    assert v02.passed_eval is False and v02.stage == "experimental"
    assert v02.fused_model_sha256 is not None


def test_seed_refuses_to_clobber_eval_ref(tmp_path):
    path = tmp_path / "registry.json"
    reg = ModelRegistry(path)
    reg.register(ModelRecord(
        version="gemma3:4b",
        eval_ref="experiments/keep-me.md",
        notes="measured",
    ))
    assert seed_registry(path, force=False, commit="abc") == 1
    assert ModelRegistry(path).get("gemma3:4b").eval_ref == "experiments/keep-me.md"


def test_seed_force_overwrites(tmp_path):
    path = tmp_path / "registry.json"
    reg = ModelRegistry(path)
    reg.register(ModelRecord(
        version="gemma3:4b",
        eval_ref="old.md",
        notes="old",
    ))
    assert seed_registry(path, force=True, commit="abc") == 0
    rec = ModelRegistry(path).get("gemma3:4b")
    assert rec.eval_ref == "experiments/baseline_gemma3-4b_v2/scorecard.md"
    assert rec.passed_eval is False
