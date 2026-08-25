"""Regression tests for the benchmark_v3 instrument and its scorers."""

from pathlib import Path

from gemma_cyber.evaluation.schema import load_benchmark
from gemma_cyber.evaluation.scorers import score_item

V3 = Path("data/evaluation/benchmark_v3.jsonl")


def _load():
    return load_benchmark(V3)


def test_v3_loads_and_has_explicit_splits():
    items = _load()
    assert len(items) >= 40
    splits = {it.split for it in items}
    assert splits == {"dev", "test"}


def test_v3_has_attack_precision_factual_items():
    items = _load()
    factual = [it for it in items if it.scorer == "factual"]
    assert factual, "v3 must contain factual-scored ATT&CK precision items"
    # Every factual item must declare its constraints.
    for it in factual:
        assert it.required_all or it.required_any


def test_v3_kerberoasting_item_catches_t1060():
    """The flagship item must fail a T1060 answer and pass a correct T1558.003 answer."""
    items = {it.id: it for it in _load()}
    item = items["v3-attack-kerberoasting-t1060-trap"]
    bad = "Kerberoasting is T1060, a privilege escalation technique; the ticket has the DA hash."
    good = ("Kerberoasting is T1558.003 under Credential Access (TA0006); the ticket is "
            "encrypted with the service account key, cracked offline.")
    assert not score_item(item, bad).passed
    assert score_item(item, good).passed


def test_v3_mcq_answers_are_valid_keys():
    for it in _load():
        if it.scorer == "mcq":
            assert it.choices and it.answer in it.choices
