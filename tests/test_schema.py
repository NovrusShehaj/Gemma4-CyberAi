"""Validate the frozen benchmarks load, are well-formed, and are trap-covered."""

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from gemma_cyber.evaluation.schema import BenchmarkItem, load_benchmark

DATA = Path(__file__).resolve().parent.parent / "data" / "evaluation"
BENCHMARK = DATA / "benchmark_v1.jsonl"
BENCHMARK_V2 = DATA / "benchmark_v2.jsonl"


def test_benchmark_loads_and_is_nonempty():
    items = load_benchmark(BENCHMARK)
    assert len(items) >= 20


def test_ids_are_unique():
    items = load_benchmark(BENCHMARK)
    ids = [it.id for it in items]
    assert len(ids) == len(set(ids))


def test_every_item_has_license_and_provenance():
    # Licensing/provenance is mandatory (PROJECT_PLAN.md §16).
    for it in load_benchmark(BENCHMARK):
        assert it.license
        assert it.provenance


def test_trap_categories_present():
    cats = {it.category for it in load_benchmark(BENCHMARK)}
    assert "insufficient_evidence" in cats
    assert "hallucination" in cats


def test_mcq_requires_choices_and_valid_answer():
    with pytest.raises(ValueError):
        BenchmarkItem(id="x", category="c", domain="general", scorer="mcq", question="q")


def test_keyword_requires_keywords():
    with pytest.raises(ValueError):
        BenchmarkItem(id="x", category="c", domain="general", scorer="keyword", question="q")


def test_render_prompt_includes_question_and_choices():
    item = BenchmarkItem(
        id="x", category="c", domain="general", scorer="mcq", question="Pick one",
        choices={"A": "foo", "B": "bar"}, answer="A",
    )
    prompt = item.render_prompt()
    assert "Pick one" in prompt
    assert "A) foo" in prompt and "B) bar" in prompt


# -- split field (Benchmark v2+) ---------------------------------------------

def test_split_defaults_to_dev():
    item = BenchmarkItem(id="x", category="c", domain="general", scorer="mcq",
                         question="q", choices={"A": "a", "B": "b"}, answer="A")
    assert item.split == "dev"


def test_split_accepts_dev_and_test():
    for s in ("dev", "test"):
        item = BenchmarkItem(id="x", category="c", domain="general", scorer="mcq",
                             question="q", choices={"A": "a", "B": "b"}, answer="A", split=s)
        assert item.split == s


def test_split_rejects_invalid_value():
    with pytest.raises(ValidationError):
        BenchmarkItem(id="x", category="c", domain="general", scorer="mcq", question="q",
                      choices={"A": "a", "B": "b"}, answer="A", split="train")  # type: ignore[arg-type]


def test_v1_still_loads_and_is_frozen_without_split():
    # benchmark_v1 pre-dates the split field; it must still load (defaulting to dev).
    items = load_benchmark(BENCHMARK)
    assert len(items) == 25
    assert all(it.split == "dev" for it in items)


# -- benchmark_v2 acceptance requirements ------------------------------------

def test_v2_loads_with_at_least_100_items():
    assert len(load_benchmark(BENCHMARK_V2)) >= 100


def test_v2_at_least_six_items_per_category():
    cats = Counter(it.category for it in load_benchmark(BENCHMARK_V2))
    low = {c: n for c, n in cats.items() if n < 6}
    assert not low, f"categories below 6 items: {low}"


def test_v2_has_enough_discriminating_traps():
    cats = Counter(it.category for it in load_benchmark(BENCHMARK_V2))
    assert cats["hallucination"] >= 6
    assert cats["insufficient_evidence"] >= 6


def test_v2_covers_all_three_domains():
    domains = {it.domain for it in load_benchmark(BENCHMARK_V2)}
    assert {"blue_team", "offensive_ctf", "general"} <= domains


def test_v2_split_is_roughly_60_40():
    items = load_benchmark(BENCHMARK_V2)
    test_ratio = sum(it.split == "test" for it in items) / len(items)
    assert 0.30 <= test_ratio <= 0.50, f"test ratio {test_ratio:.3f} outside 30-50%"


def test_v2_every_item_has_license_source_and_provenance():
    for it in load_benchmark(BENCHMARK_V2):
        assert it.license and it.source and it.provenance, it.id


def test_v2_ids_are_unique_and_disjoint_from_v1():
    v1 = {it.id for it in load_benchmark(BENCHMARK)}
    v2 = [it.id for it in load_benchmark(BENCHMARK_V2)]
    assert len(v2) == len(set(v2))
    assert not (v1 & set(v2)), "v2 ids must not collide with v1 ids"
