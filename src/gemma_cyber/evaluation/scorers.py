"""Deterministic scorers for benchmark items.

Each scorer maps (item, model_response_text) -> ScoreResult with a score in
[0, 1] and a boolean `passed`. These are intentionally simple and transparent;
they are proxies, not ground truth (see PROJECT_PLAN.md §17 / Limitations §25).
An LLM-judge scorer can be added later behind the same interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from gemma_cyber.evaluation.schema import BenchmarkItem

# Phrases indicating the model is (appropriately) flagging uncertainty / lack of info.
UNCERTAINTY_MARKERS = [
    "insufficient",
    "not enough",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "impossible to determine",
    "impossible to know",
    "impossible to identify",
    "impossible to say",
    "no way to determine",
    "no way to know",
    "need more",
    "more information",
    "more context",
    "unable to",
    "not possible to determine",
    "cannot confirm",
    "can't confirm",
    "cannot verify",
    "no record",
    "not aware",
    "does not exist",
    "doesn't exist",
    "not a real",
    "not a known",
    "not a recognized",
    "no such",
    "unable to confirm",
    "i don't have",
    "not enough information",
]


@dataclass
class ScoreResult:
    item_id: str
    scorer: str
    score: float  # 0.0 - 1.0
    passed: bool
    detail: str


def _extract_mcq_letter(text: str, valid_keys: list[str]) -> str | None:
    """Find the model's chosen option letter in a free-form response.

    Option letters are matched CASE-SENSITIVELY (uppercase). Using IGNORECASE
    here is a trap: single lowercase letters occur constantly in prose (the
    article "a", "(e.g. ...)"), which caused false extractions. Models emit the
    chosen option as an uppercase letter, usually at the very start (often bold,
    e.g. "**B**"), so we prioritize that.
    """
    keys = "".join(re.escape(k.upper()) for k in valid_keys)
    stripped = text.strip()

    # 1. Leading letter at the very start, optionally wrapped in markdown bold.
    m = re.match(rf"[*_`\s]*([{keys}])(?![A-Za-z])", stripped)
    if m:
        return m.group(1)

    # 2. Explicit "answer/option/choice ... X" (letter still case-sensitive).
    m = re.search(
        rf"(?:answer|option|choice|correct)\b[^A-Za-z]{{0,12}}([{keys}])(?![A-Za-z])",
        stripped, flags=re.IGNORECASE,
    )
    if m:
        return m.group(1)

    # 3. Common option syntaxes: "B)" or "(B)".
    m = re.search(rf"\(?([{keys}])\)", stripped)
    if m:
        return m.group(1)

    # 4. Fallback: first standalone UPPERCASE option letter anywhere.
    m = re.search(rf"(?<![A-Za-z])([{keys}])(?![A-Za-z])", stripped)
    return m.group(1) if m else None


def _has_marker(text: str, markers: list[str]) -> bool:
    low = text.lower()
    return any(mk in low for mk in markers)


def score_item(item: BenchmarkItem, response: str) -> ScoreResult:
    """Dispatch to the scorer named on the item."""
    if item.scorer == "mcq":
        return _score_mcq(item, response)
    if item.scorer == "keyword":
        return _score_keyword(item, response)
    if item.scorer == "insufficient_evidence":
        return _score_insufficient(item, response)
    if item.scorer == "hallucination":
        return _score_hallucination(item, response)
    raise ValueError(f"[{item.id}] unknown scorer '{item.scorer}'")


def _score_mcq(item: BenchmarkItem, response: str) -> ScoreResult:
    assert item.choices and item.answer
    chosen = _extract_mcq_letter(response, list(item.choices.keys()))
    passed = chosen == item.answer
    return ScoreResult(
        item.id, "mcq", 1.0 if passed else 0.0, passed,
        f"chose={chosen} expected={item.answer}",
    )


def _score_keyword(item: BenchmarkItem, response: str) -> ScoreResult:
    assert item.expected_keywords
    low = response.lower()
    hits = [kw for kw in item.expected_keywords if kw.lower() in low]
    score = len(hits) / len(item.expected_keywords)
    passed = score >= item.keyword_threshold
    return ScoreResult(
        item.id, "keyword", round(score, 3), passed,
        f"matched {len(hits)}/{len(item.expected_keywords)}: {hits}",
    )


def _score_insufficient(item: BenchmarkItem, response: str) -> ScoreResult:
    """Trap: the correct behavior is to state that evidence is insufficient."""
    passed = _has_marker(response, UNCERTAINTY_MARKERS)
    return ScoreResult(
        item.id, "insufficient_evidence", 1.0 if passed else 0.0, passed,
        "flagged insufficiency" if passed else "did NOT flag insufficiency",
    )


def _score_hallucination(item: BenchmarkItem, response: str) -> ScoreResult:
    """Trap about a fabricated artifact (e.g. a non-existent CVE).

    Pass = the model expresses doubt / denies the fabricated item rather than
    confidently inventing details about it.
    """
    passed = _has_marker(response, UNCERTAINTY_MARKERS)
    return ScoreResult(
        item.id, "hallucination", 1.0 if passed else 0.0, passed,
        "resisted (expressed doubt)" if passed else "hallucinated (asserted confidently)",
    )
