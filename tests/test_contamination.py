"""Unit tests for the contamination detector.

Uses tiny synthetic fixtures (no dependency on the real benchmark) and stays
deterministic and fast.
"""

from gemma_cyber.data.contamination import (
    check_contamination,
    jaccard_similarity,
    normalize_text,
    record_text,
    word_ngrams,
)

# -- normalization -----------------------------------------------------------

def test_normalize_folds_case_whitespace_and_punctuation():
    a = normalize_text("What   is  SQL Injection?")
    b = normalize_text("what is sql injection")
    assert a == b == "what is sql injection"


def test_normalize_folds_unicode_punctuation():
    # Smart quotes / em dash vs. ascii equivalents must normalize the same.
    assert normalize_text("the attacker’s “tool”—fast") == \
        normalize_text("the attacker's \"tool\"-fast")


def test_normalize_does_not_collapse_distinct_text():
    assert normalize_text("SYN scan") != normalize_text("ACK scan")


# -- exact matching ----------------------------------------------------------

def test_identical_records_detected_as_exact():
    a = [{"id": "t1", "question": "Explain a SYN flood attack."}]
    b = [{"id": "e1", "question": "Explain a SYN flood attack."}]
    rep = check_contamination(a, b)
    assert len(rep.exact) == 1
    assert rep.exact[0].a_id == "t1" and rep.exact[0].b_id == "e1"
    assert not rep.is_clean


def test_normalized_equivalent_records_detected_as_exact():
    # Same content, different case/whitespace/punctuation -> still exact.
    a = [{"id": "t1", "question": "What is   CROSS-site scripting?!"}]
    b = [{"id": "e1", "question": "what is cross-site scripting"}]
    rep = check_contamination(a, b)
    assert len(rep.exact) == 1
    assert not rep.fuzzy  # exact pairs are not double-counted as fuzzy


def test_mcq_choices_included_so_same_stem_differing_options_differ():
    a = [{"id": "t1", "question": "Pick one", "choices": {"A": "tcp", "B": "udp"}}]
    b = [{"id": "e1", "question": "Pick one", "choices": {"A": "http", "B": "dns"}}]
    rep = check_contamination(a, b, threshold=0.9)
    assert not rep.exact


# -- fuzzy matching ----------------------------------------------------------

def test_paraphrase_detected_as_fuzzy():
    base = "An attacker performs a brute force attack against the ssh service on the server"
    edit = "An attacker performs a brute force attack against the ssh service on the host"
    sim = jaccard_similarity(base, edit, n=3)
    assert sim >= 0.7
    a = [{"id": "t1", "question": base}]
    b = [{"id": "e1", "question": edit}]
    rep = check_contamination(a, b, threshold=0.7)
    assert not rep.exact
    assert len(rep.fuzzy) == 1
    assert rep.fuzzy[0].similarity >= 0.7


def test_unrelated_records_not_flagged():
    a = [{"id": "t1", "question": "How does DNS tunneling exfiltrate data?"}]
    b = [{"id": "e1", "question": "What port does RDP use by default?"}]
    rep = check_contamination(a, b, threshold=0.7)
    assert rep.is_clean
    assert not rep.exact and not rep.fuzzy


def test_shared_vocabulary_but_different_meaning_scores_low():
    # Same topic words, different phrasing/order -> trigram Jaccard stays low.
    a = "the firewall blocked the inbound connection from the attacker"
    b = "the attacker blocked the firewall using an inbound connection flood"
    assert jaccard_similarity(a, b, n=3) < 0.5


def test_threshold_controls_sensitivity():
    a = [{"id": "t1", "question": "detect lateral movement using windows event logs today"}]
    b = [{"id": "e1", "question": "detect lateral movement using windows event logs now"}]
    strict = check_contamination(a, b, threshold=0.95)
    loose = check_contamination(a, b, threshold=0.5)
    assert strict.is_clean
    assert loose.fuzzy


# -- helpers -----------------------------------------------------------------

def test_word_ngrams_short_text_falls_back_to_unigrams():
    grams = word_ngrams("only two", n=3)
    assert grams == {("only",), ("two",)}


def test_record_text_skips_missing_fields():
    text = record_text({"id": "x", "question": "q only"}, fields=("context", "evidence", "question"))
    assert text.strip() == "q only"


def test_empty_texts_are_clean():
    rep = check_contamination([{"id": "t1"}], [{"id": "e1"}])
    assert rep.is_clean


def test_report_is_deterministic():
    a = [{"id": "t1", "question": "brute force ssh attack on the server host"},
         {"id": "t2", "question": "cross site scripting in the search box field"}]
    b = [{"id": "e1", "question": "brute force ssh attack on the server node"}]
    r1 = check_contamination(a, b, threshold=0.5)
    r2 = check_contamination(a, b, threshold=0.5)
    assert [(m.a_id, m.b_id) for m in r1.fuzzy] == [(m.a_id, m.b_id) for m in r2.fuzzy]
