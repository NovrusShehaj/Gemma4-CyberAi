"""Unit tests for the deterministic scorers."""

from gemma_cyber.evaluation.schema import BenchmarkItem
from gemma_cyber.evaluation.scorers import score_item


def _mcq(answer="B"):
    return BenchmarkItem(
        id="m", category="c", domain="general", scorer="mcq", question="q",
        choices={"A": "a", "B": "b", "C": "c", "D": "d"}, answer=answer,
    )


def test_mcq_correct_various_formats():
    item = _mcq("B")
    for resp in ["Answer: B", "The answer is (B).", "B) because ...", "I choose B."]:
        assert score_item(item, resp).passed, resp


def test_mcq_incorrect():
    item = _mcq("B")
    assert not score_item(item, "Answer: C").passed


def test_keyword_threshold():
    item = BenchmarkItem(
        id="k", category="c", domain="general", scorer="keyword", question="q",
        expected_keywords=["brute", "block", "203.0.113.9"], keyword_threshold=0.5,
    )
    good = score_item(item, "This is a brute force; block 203.0.113.9 now.")
    assert good.passed and good.score == 1.0
    partial = score_item(item, "Looks like a brute force attempt.")
    assert not partial.passed and 0.0 < partial.score < 0.5


def test_insufficient_evidence_scorer():
    item = BenchmarkItem(
        id="i", category="c", domain="general", scorer="insufficient_evidence", question="q",
    )
    assert score_item(item, "There is insufficient evidence to determine that.").passed
    assert not score_item(item, "It is definitely the Emotet malware family.").passed


def test_hallucination_scorer():
    item = BenchmarkItem(
        id="h", category="c", domain="general", scorer="hallucination", question="q",
    )
    ok = score_item(item, "I have no record of that CVE; it does not appear to be real.")
    assert ok.passed
    bad = score_item(item, "CVE-2029-88888 is a buffer overflow in Apache 2.9, patched in 2.9.1.")
    assert not bad.passed
