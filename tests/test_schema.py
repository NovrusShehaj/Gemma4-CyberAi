"""Validate the frozen benchmark loads, is well-formed, and is trap-covered."""

from pathlib import Path

import pytest

from gemma_cyber.evaluation.schema import BenchmarkItem, load_benchmark

BENCHMARK = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "benchmark_v1.jsonl"


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
