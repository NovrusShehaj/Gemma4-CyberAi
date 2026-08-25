"""Tests for the structured cybersecurity fact registry."""

from gemma_cyber.knowledge import load_fact_registry


def test_registry_loads_and_is_indexed():
    reg = load_fact_registry()
    assert "kerberoasting" in reg.attack_techniques
    assert reg.technique("kerberoasting").key == "kerberoasting"
    assert len(reg.technique_keys()) >= 8


def test_kerberoasting_canonical_facts():
    """Lock the exact facts the v0.1 base model got wrong."""
    k = load_fact_registry().technique("kerberoasting")
    assert k.id == "T1558.003"
    assert k.tactic_id == "TA0006"
    assert "Credential Access" in k.tactic
    # The two IDs the model confused it with are recorded as forbidden.
    assert "T1060" in k.forbidden_ids
    assert "T1068" in k.forbidden_ids


def test_forbidden_ids_are_not_the_correct_id():
    """A technique must never list its own correct ID as forbidden."""
    reg = load_fact_registry()
    for key, t in reg.attack_techniques.items():
        assert t.id not in t.forbidden_ids, key


def test_obsolete_t1060_recorded():
    reg = load_fact_registry()
    assert "T1060" in reg.obsolete_ids
