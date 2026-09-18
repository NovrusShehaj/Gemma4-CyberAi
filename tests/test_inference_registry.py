"""Model registry: provenance persistence + gated promotion lifecycle."""

from __future__ import annotations

import pytest

from gemma_cyber.inference.errors import RegistryError
from gemma_cyber.inference.registry import (
    ModelRecord,
    ModelRegistry,
    PromotionError,
)


def _reg(tmp_path):
    return ModelRegistry(tmp_path / "registry.json")


def test_register_and_roundtrip(tmp_path):
    reg = _reg(tmp_path)
    reg.register(ModelRecord(version="gemma3-cyber:v0.2", dataset_version="sft_v0.2"))
    # Reload from disk -> same data.
    reg2 = ModelRegistry(tmp_path / "registry.json")
    rec = reg2.get("gemma3-cyber:v0.2")
    assert rec.stage == "experimental"
    assert rec.dataset_version == "sft_v0.2"
    assert rec.ollama_tag == "gemma3-cyber:v0.2"  # defaults to version


def test_duplicate_register_rejected(tmp_path):
    reg = _reg(tmp_path)
    reg.register(ModelRecord(version="m"))
    with pytest.raises(RegistryError):
        reg.register(ModelRecord(version="m"))
    reg.register(ModelRecord(version="m", notes="v2"), overwrite=True)
    assert reg.get("m").notes == "v2"


def test_promotion_requires_passing_eval(tmp_path):
    reg = _reg(tmp_path)
    reg.register(ModelRecord(version="m"))
    # experimental -> evaluated is legal only via mark_evaluated / after eval.
    with pytest.raises(PromotionError):
        reg.promote("m", "candidate")  # skips a step AND ungated


def test_full_lifecycle(tmp_path):
    reg = _reg(tmp_path)
    reg.register(ModelRecord(version="m"))
    reg.mark_evaluated("m", passed=True, eval_ref="experiments/x/scorecard.md")
    assert reg.get("m").stage == "evaluated"
    reg.promote("m", "candidate")
    assert reg.get("m").stage == "candidate"
    reg.promote("m", "production")
    assert reg.get("m").stage == "production"
    # History records creation (from==to) then each promotion hop, with subject.
    hist = reg.get("m").history
    assert [h["to"] for h in hist] == [
        "experimental", "evaluated", "candidate", "production",
    ]
    assert all("subject" in h for h in hist)


def test_failed_eval_does_not_advance(tmp_path):
    reg = _reg(tmp_path)
    reg.register(ModelRecord(version="m"))
    reg.mark_evaluated("m", passed=False)
    assert reg.get("m").stage == "experimental"
    assert reg.get("m").passed_eval is False


def test_illegal_transition_rejected(tmp_path):
    reg = _reg(tmp_path)
    reg.register(ModelRecord(version="m"))
    reg.mark_evaluated("m", passed=True)
    # evaluated -> production skips candidate.
    with pytest.raises(PromotionError):
        reg.promote("m", "production")


def test_single_production_invariant(tmp_path):
    reg = _reg(tmp_path)
    for v in ("a", "b"):
        reg.register(ModelRecord(version=v))
        reg.mark_evaluated(v, passed=True)
        reg.promote(v, "candidate")
    reg.promote("a", "production")
    reg.promote("b", "production")  # should demote a
    assert reg.get("a").stage == "candidate"
    assert reg.get("b").stage == "production"
    assert reg.production().version == "b"


def test_resolve_stage_and_raw(tmp_path):
    reg = _reg(tmp_path)
    reg.register(ModelRecord(version="gemma3-cyber:v0.2", ollama_tag="gc:v0.2"))
    reg.mark_evaluated("gemma3-cyber:v0.2", passed=True)
    reg.promote("gemma3-cyber:v0.2", "candidate")
    reg.promote("gemma3-cyber:v0.2", "production")
    assert reg.resolve("production") == "gc:v0.2"
    assert reg.resolve("gemma3-cyber:v0.2") == "gc:v0.2"
    assert reg.resolve("some-raw-tag:latest") == "some-raw-tag:latest"


def test_rollback_from_production(tmp_path):
    reg = _reg(tmp_path)
    reg.register(ModelRecord(version="m"))
    reg.mark_evaluated("m", passed=True)
    reg.promote("m", "candidate")
    reg.promote("m", "production")
    reg.promote("m", "candidate", reason="incident rollback")
    assert reg.get("m").stage == "candidate"


def test_fused_sha_roundtrips(tmp_path):
    sha = "1ea70da8a68526b1abbffb1eff6738319f961550348ee90ecea0bedd497ef702"
    reg = _reg(tmp_path)
    reg.register(ModelRecord(version="m", fused_model_sha256=sha))
    rec = ModelRegistry(tmp_path / "registry.json").get("m")
    assert rec.fused_model_sha256 == sha


def test_committed_registry_honesty():
    from gemma_cyber.inference.config import DEFAULT_REGISTRY_PATH

    reg = ModelRegistry(DEFAULT_REGISTRY_PATH, read_only=True)
    base = reg.get("gemma3:4b")
    v02 = reg.get("gemma3-cyber:v0.2")
    assert base.stage == "evaluated" and base.passed_eval is False
    assert v02.stage == "experimental" and v02.passed_eval is False
    assert v02.fused_model_sha256 == (
        "1ea70da8a68526b1abbffb1eff6738319f961550348ee90ecea0bedd497ef702"
    )
    assert "0.956" in (v02.notes or "")
    assert "DOES NOT PASS" in (v02.notes or "")
    assert "T1060" in (v02.notes or "")
    assert reg.production() is None
