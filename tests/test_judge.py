"""Unit tests for the LLM-judge scorer.

The provider is mocked at the client boundary (a fake `.generate`), so these tests are
deterministic, fast, and require NO live LLM / Ollama / credentials.
"""

import pytest

from gemma_cyber.clients.ollama_client import GenerationResult
from gemma_cyber.evaluation.judge import (
    JudgeScorer,
    JudgeVerdict,
    build_judge_prompt,
    build_reference,
    parse_judge_output,
)
from gemma_cyber.evaluation.schema import BenchmarkItem


class FakeClient:
    """Fake generation client. Returns a fixed string, or raises if `raise_exc` set."""

    model = "fake-judge"

    def __init__(self, response="", raise_exc=None):
        self._response = response
        self._raise = raise_exc
        self.calls = []

    def generate(self, prompt, system=None, temperature=0.0, seed=0, num_predict=None,
                 think=None):
        self.calls.append({"prompt": prompt, "system": system,
                           "temperature": temperature, "seed": seed, "think": think})
        if self._raise is not None:
            raise self._raise
        return GenerationResult(text=self._response, model=self.model, prompt=prompt,
                                system=system, options={})


def _kw_item():
    return BenchmarkItem(
        id="k1", category="log_analysis", domain="blue_team", scorer="keyword",
        question="What is happening and what do you do?",
        expected_keywords=["brute force", "block"], keyword_threshold=0.5,
    )


def _mcq_item():
    return BenchmarkItem(
        id="m1", category="fundamentals", domain="general", scorer="mcq",
        question="Pick one", choices={"A": "x", "B": "y"}, answer="B",
    )


# -- output parsing ----------------------------------------------------------

def test_parse_valid_pass():
    passed, score, reason, err = parse_judge_output('{"verdict": "PASS", "score": 0.9, "reason": "good"}')
    assert passed and score == 0.9 and reason == "good" and err is None


def test_parse_valid_fail():
    passed, score, reason, err = parse_judge_output('{"verdict":"FAIL","score":0.0,"reason":"wrong"}')
    assert passed is False and err is None


def test_parse_extracts_json_from_surrounding_prose():
    passed, score, reason, err = parse_judge_output('Sure!\n{"verdict":"PASS","score":1.0,"reason":"ok"}\nThanks')
    assert passed is True and err is None


def test_parse_empty_is_error_not_pass():
    passed, score, reason, err = parse_judge_output("")
    assert passed is False and score == 0.0 and err is not None


def test_parse_no_json_is_error():
    passed, score, reason, err = parse_judge_output("I think it is fine, PASS.")
    assert passed is False and err is not None


def test_parse_malformed_json_is_error():
    passed, score, reason, err = parse_judge_output('{"verdict": "PASS", "score": }')
    assert passed is False and err is not None


def test_parse_invalid_verdict_value_is_error():
    passed, score, reason, err = parse_judge_output('{"verdict": "MAYBE", "score": 0.5}')
    assert passed is False and err is not None and "verdict" in err


def test_parse_out_of_range_score_is_error():
    passed, score, reason, err = parse_judge_output('{"verdict": "PASS", "score": 1.7}')
    assert passed is False and err is not None


def test_parse_score_defaults_from_verdict_when_missing():
    passed, score, reason, err = parse_judge_output('{"verdict": "PASS", "reason": "ok"}')
    assert passed is True and score == 1.0 and err is None


# -- JudgeScorer.evaluate / score -------------------------------------------

def test_evaluate_pass_decision():
    judge = JudgeScorer(client=FakeClient('{"verdict":"PASS","score":0.8,"reason":"correct"}'))
    v = judge.evaluate(_kw_item(), "This is a brute force attack; block the IP.")
    assert isinstance(v, JudgeVerdict)
    assert v.passed and v.score == 0.8 and v.error is None
    assert v.model == "fake-judge" and v.prompt_version


def test_evaluate_fail_decision():
    judge = JudgeScorer(client=FakeClient('{"verdict":"FAIL","score":0.1,"reason":"off-topic"}'))
    v = judge.evaluate(_kw_item(), "The weather is nice today.")
    assert not v.passed and v.error is None


def test_evaluate_ambiguous_output_fails_safely():
    # An ambiguous/unparseable judge answer must NOT become a pass.
    judge = JudgeScorer(client=FakeClient("Hard to say, maybe partially correct."))
    v = judge.evaluate(_kw_item(), "some answer")
    assert v.passed is False and v.error is not None


def test_evaluate_provider_failure_is_error_not_pass():
    judge = JudgeScorer(client=FakeClient(raise_exc=RuntimeError("boom")))
    v = judge.evaluate(_kw_item(), "answer")
    assert v.passed is False and v.score == 0.0 and v.error and "boom" in v.error


def test_score_returns_scoreresult_interface():
    judge = JudgeScorer(client=FakeClient('{"verdict":"PASS","score":1.0,"reason":"ok"}'))
    res = judge.score(_mcq_item(), "B")
    assert res.item_id == "m1" and res.scorer == "judge" and res.passed and res.score == 1.0


def test_score_error_detail_is_flagged():
    judge = JudgeScorer(client=FakeClient(raise_exc=TimeoutError("timed out")))
    res = judge.score(_kw_item(), "answer")
    assert not res.passed and res.detail.startswith("JUDGE_ERROR")


def test_judge_is_deterministic_call():
    client = FakeClient('{"verdict":"PASS","score":1.0,"reason":"ok"}')
    judge = JudgeScorer(client=client, temperature=0.0, seed=0)
    judge.evaluate(_kw_item(), "answer")
    assert client.calls[0]["temperature"] == 0.0 and client.calls[0]["seed"] == 0


def test_judge_disables_thinking_by_default():
    # Gemma 4 thinking models return empty output unless thinking is disabled.
    client = FakeClient('{"verdict":"PASS","score":1.0,"reason":"ok"}')
    JudgeScorer(client=client).evaluate(_kw_item(), "answer")
    assert client.calls[0]["think"] is False


# -- prompt construction -----------------------------------------------------

def test_reference_describes_task_type():
    assert "multiple-choice" in build_reference(_mcq_item()).lower()
    assert "concept" in build_reference(_kw_item()).lower()


def test_hallucination_reference_mentions_fabricated():
    item = BenchmarkItem(id="h", category="hallucination", domain="general",
                         scorer="hallucination", question="Describe CVE-2029-99999.")
    assert "fabricat" in build_reference(item).lower()


def test_judge_prompt_includes_question_and_answer():
    prompt = build_judge_prompt(_kw_item(), "my candidate answer")
    assert "my candidate answer" in prompt
    assert "What is happening" in prompt
