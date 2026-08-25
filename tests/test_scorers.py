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
    for resp in ["Answer: B", "The answer is (B).", "B) because ...", "I choose B.",
                 "**B**  **Explanation:** ...", "B\n\nExplanation: ..."]:
        assert score_item(item, resp).passed, resp


def test_mcq_incorrect():
    item = _mcq("B")
    assert not score_item(item, "Answer: C").passed


def test_mcq_ignores_lowercase_prose_letters():
    """Regression: 'A CVSS score...' must not be read as choice A (article trap)."""
    item = _mcq("B")
    resp = "**B**  **Explanation:**  A CVSS v3.1 score of 9.8 is critical, a very high risk."
    res = score_item(item, resp)
    assert res.passed, res.detail


def test_mcq_leading_letter_beats_prose():
    """Regression: leading 'C' answer must win over later prose letters."""
    item = _mcq("C")
    resp = "C  **Explanation:** Confidentiality protects data from unauthorized access."
    assert score_item(item, resp).passed


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
    # Model's actual phrasing on the real benchmark must also count as correct.
    assert score_item(item, "It is impossible to determine which malware family.").passed
    assert not score_item(item, "It is definitely the Emotet malware family.").passed


def test_hallucination_scorer():
    item = BenchmarkItem(
        id="h", category="c", domain="general", scorer="hallucination", question="q",
    )
    ok = score_item(item, "I have no record of that CVE; it does not appear to be real.")
    assert ok.passed
    bad = score_item(item, "CVE-2029-88888 is a buffer overflow in Apache 2.9, patched in 2.9.1.")
    assert not bad.passed


# --- factual scorer (Benchmark v3) -----------------------------------------------------

def _factual(required_all=None, required_any=None, forbidden=None):
    return BenchmarkItem(
        id="f", category="attack_mapping", domain="blue_team", scorer="factual",
        question="What is the ATT&CK ID and tactic for Kerberoasting?",
        required_all=required_all, required_any=required_any, forbidden=forbidden,
    )


def test_factual_pass_when_required_present_no_forbidden():
    item = _factual(required_all=["T1558.003", "Credential Access"], forbidden=["T1060", "T1068"])
    resp = "Kerberoasting is T1558.003 under the Credential Access tactic (TA0006)."
    res = score_item(item, resp)
    assert res.passed and res.score == 1.0


def test_factual_forbidden_hard_fails_even_with_correct_keywords():
    """The core hardening: a wrong ATT&CK ID cannot be masked by correct keywords.

    This is the exact v0.1 failure mode (Kerberoasting -> T1060 while still mentioning
    'ticket' and 'service account'). It MUST score 0.0 and fail.
    """
    item = _factual(required_all=["T1558.003"], forbidden=["T1060", "T1068"])
    resp = ("Kerberoasting is T1060. The attacker requests a service ticket for a "
            "service account and cracks it offline. It also maps to T1558.003.")
    res = score_item(item, resp)
    assert not res.passed and res.score == 0.0
    assert "FORBIDDEN" in res.detail


def test_factual_partial_when_required_missing():
    item = _factual(required_all=["T1558.003", "Credential Access"], forbidden=["T1060"])
    resp = "Kerberoasting is T1558.003."  # missing the tactic
    res = score_item(item, resp)
    assert not res.passed and 0.0 < res.score < 1.0


def test_factual_id_boundary_precision():
    """T1558 (parent) must NOT satisfy a requirement for the sub-technique T1558.003,
    and a forbidden T1060 must not be found inside an unrelated token like T10600."""
    item = _factual(required_all=["T1558.003"], forbidden=["T1060"])
    # Parent-only mention does not count as the sub-technique.
    assert not score_item(item, "This is Kerberos technique T1558 generally.").passed
    # Forbidden ID embedded in a longer number must not fire a false positive.
    ok = score_item(item, "See doc ref T10600 and the technique is T1558.003.")
    assert ok.passed, ok.detail


def test_factual_requires_constraints_at_schema_level():
    import pytest
    with pytest.raises(ValueError):
        BenchmarkItem(id="bad", category="c", domain="general", scorer="factual",
                      question="q")
