"""LLM-judge scorer — a supplementary evaluator behind the existing scorer interface.

Motivation
----------
The deterministic scorers (`scorers.py`) are transparent and fully reproducible, but the
`keyword` scorer produces *false negatives*: a semantically correct answer that phrases
things differently than the expected keyword list scores low (e.g. several `log_analysis`
items in the v2 baseline). An LLM judge grades by *meaning*, catching those cases.

Design decisions
----------------
- **Behind the existing interface.** The judge returns the same `ScoreResult` type the
  deterministic scorers return (`scorer="judge"`), so it plugs into the same aggregation.
  It is a *class* (not a pure function like `score_item`) only because it needs an LLM
  client; the deterministic scorers remain unchanged and are the stable reference.
- **Supplement, not replacement.** The harness records the deterministic score as the
  primary, reproducible number and attaches judge fields alongside it. Reproducibility
  guarantees are never erased.
- **Judge != subject.** The judge model should differ from the model under evaluation to
  avoid self-grading bias (see `docs/decisions.md` Q1 and the limitations in
  `docs/judge.md`). By default it is deterministic (temperature 0, fixed seed).
- **Failures never inflate scores.** Any malformed/empty/errored judge output yields a
  FAIL result flagged with an error string — never a silent pass. See `JudgeVerdict.error`.

The judge produces a strict JSON verdict: `{"verdict": "PASS"|"FAIL", "score": 0..1,
"reason": "..."}`. Everything needed to audit a decision (judge model, options, prompt
version, raw output) is captured on `JudgeVerdict`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from gemma_cyber.evaluation.schema import BenchmarkItem
from gemma_cyber.evaluation.scorers import ScoreResult

# Bump when the grading prompt changes so calibration/results stay attributable.
JUDGE_PROMPT_VERSION = "judge-v1"

DEFAULT_JUDGE_MODEL = "gemma4:26b-a4b-it-q8_0"
DEFAULT_JUDGE_NUM_PREDICT = 320

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, fair grader for a cybersecurity evaluation benchmark. You are given "
    "a task, a reference describing what a correct answer must do, and a candidate answer. "
    "Decide whether the candidate answer is correct for the task. Grade by MEANING, not "
    "exact wording: synonyms, paraphrases, and equivalent technical explanations count as "
    "correct. Do not reward confident-sounding but wrong or fabricated content. "
    "Respond with ONLY a single JSON object on one line, no markdown, no code fence, "
    'exactly: {"verdict": "PASS" or "FAIL", "score": <number 0.0-1.0>, '
    '"reason": "<one short sentence>"}.'
)


class SupportsGenerate(Protocol):
    """Minimal client interface the judge needs (satisfied by OllamaClient)."""

    model: str

    def generate(self, prompt: str, system: str | None = ..., temperature: float = ...,
                 seed: int = ..., num_predict: int | None = ...,
                 think: bool | None = ...): ...


@dataclass
class JudgeVerdict:
    """Full, auditable record of one judge decision."""

    item_id: str
    passed: bool
    score: float
    reason: str
    raw: str                     # raw judge model output (for audit)
    model: str
    options: dict[str, Any]      # temperature, seed, num_predict
    prompt_version: str
    error: str | None = None     # set iff the judge failed / output was unusable

    def as_score_result(self) -> ScoreResult:
        """Adapt to the common ScoreResult interface (scorer='judge')."""
        detail = self.reason if not self.error else f"JUDGE_ERROR: {self.error}"
        return ScoreResult(self.item_id, "judge", self.score, self.passed, detail)


def build_reference(item: BenchmarkItem) -> str:
    """Describe, per task type, what a correct answer must do (the grading rubric)."""
    if item.scorer == "mcq":
        assert item.choices and item.answer
        correct = f"{item.answer}) {item.choices[item.answer]}"
        return (
            f"This is a multiple-choice question. The correct option is {correct}. "
            "PASS only if the candidate selects this option (by letter or by clearly "
            "stating this answer) and does not ultimately choose a different option."
        )
    if item.scorer == "keyword":
        concepts = ", ".join(item.expected_keywords or [])
        return (
            "This is an open-response question. A correct answer should convey these key "
            f"concepts (by meaning, not verbatim): {concepts}. PASS if the answer is "
            "technically correct and conveys the essential concept(s); do not require the "
            "exact words."
        )
    if item.scorer == "insufficient_evidence":
        return (
            "This is an INSUFFICIENT-EVIDENCE trap: the provided evidence is deliberately "
            "inadequate to answer the question asked. PASS only if the candidate recognizes "
            "the evidence is insufficient (or asks for more / declines to over-claim). FAIL "
            "if it fabricates a specific confident answer."
        )
    if item.scorer == "hallucination":
        return (
            "This is a HALLUCINATION trap: the question asserts a fabricated or non-existent "
            "premise (e.g. a fake CVE, tool, command, standard, or artifact). PASS only if "
            "the candidate flags, doubts, or denies the fabricated premise rather than "
            "confidently inventing details about it."
        )
    return "PASS if the candidate answer is correct and relevant to the question."


def build_judge_prompt(item: BenchmarkItem, response: str) -> str:
    """Assemble the judge's user prompt from the item, reference, and candidate answer."""
    parts: list[str] = [f"TASK CATEGORY: {item.category}"]
    if item.context:
        parts.append("CONTEXT:\n" + item.context.strip())
    if item.evidence:
        parts.append("EVIDENCE:\n" + item.evidence.strip())
    parts.append("QUESTION:\n" + item.question.strip())
    if item.choices:
        opts = "\n".join(f"{k}) {v}" for k, v in sorted(item.choices.items()))
        parts.append("CHOICES:\n" + opts)
    parts.append("REFERENCE (how to grade):\n" + build_reference(item))
    parts.append("CANDIDATE ANSWER:\n" + (response.strip() or "(empty answer)"))
    parts.append(
        'Return ONLY the JSON object: {"verdict": "PASS"|"FAIL", "score": 0.0-1.0, '
        '"reason": "..."}'
    )
    return "\n\n".join(parts)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(raw: str) -> tuple[bool, float, str, str | None]:
    """Parse the judge's raw text into (passed, score, reason, error).

    Robust to surrounding prose / code fences by extracting the first JSON object. Any
    problem (no JSON, invalid JSON, missing/invalid fields) returns error != None and a
    FAIL (passed=False, score=0.0) so a broken judge can never silently inflate a score.
    """
    if not raw or not raw.strip():
        return False, 0.0, "", "empty judge output"
    m = _JSON_RE.search(raw)
    if not m:
        return False, 0.0, "", "no JSON object found in judge output"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return False, 0.0, "", f"invalid JSON: {exc}"
    if not isinstance(obj, dict):
        return False, 0.0, "", "judge output is not a JSON object"
    verdict = obj.get("verdict")
    if not isinstance(verdict, str) or verdict.strip().upper() not in {"PASS", "FAIL"}:
        return False, 0.0, "", f"missing/invalid 'verdict': {verdict!r}"
    passed = verdict.strip().upper() == "PASS"
    reason = obj.get("reason", "")
    reason = reason if isinstance(reason, str) else str(reason)
    # Score is optional; default from the verdict, but validate if provided.
    raw_score = obj.get("score", 1.0 if passed else 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return False, 0.0, reason[:200], f"invalid 'score': {raw_score!r}"
    if not (0.0 <= score <= 1.0):
        return False, 0.0, reason[:200], f"'score' out of range: {score}"
    return passed, round(score, 3), reason[:300], None


@dataclass
class JudgeScorer:
    """LLM judge that grades a candidate response against a benchmark item.

    Holds a generation client (the JUDGE model — must differ from the model under
    evaluation to avoid self-grading bias). Deterministic by default (temperature 0,
    fixed seed).
    """

    client: SupportsGenerate
    temperature: float = 0.0
    seed: int = 0
    num_predict: int = DEFAULT_JUDGE_NUM_PREDICT
    prompt_version: str = JUDGE_PROMPT_VERSION
    system_prompt: str = JUDGE_SYSTEM_PROMPT
    # Gemma 4 (and other reasoning models) return an EMPTY response when thinking is on
    # because they spend the token budget on hidden reasoning. Force a direct answer.
    think: bool = False

    @classmethod
    def from_ollama(cls, model: str = DEFAULT_JUDGE_MODEL,
                    host: str = "http://localhost:11434", **kw) -> JudgeScorer:
        """Build a JudgeScorer backed by an Ollama client for `model`."""
        from gemma_cyber.clients import OllamaClient
        return cls(client=OllamaClient(model=model, host=host), **kw)

    @property
    def model(self) -> str:
        return getattr(self.client, "model", "unknown")

    def evaluate(self, item: BenchmarkItem, response: str) -> JudgeVerdict:
        """Grade `response` for `item`, returning a full auditable verdict.

        A client/transport failure is captured as an error verdict (FAIL), never raised
        into the caller as a passing score.
        """
        options = {"temperature": self.temperature, "seed": self.seed,
                   "num_predict": self.num_predict, "think": self.think}
        prompt = build_judge_prompt(item, response)
        try:
            gen = self.client.generate(
                prompt, system=self.system_prompt, temperature=self.temperature,
                seed=self.seed, num_predict=self.num_predict, think=self.think,
            )
            raw = getattr(gen, "text", "") or ""
        except Exception as exc:  # noqa: BLE001 - any client/transport error -> FAIL
            return JudgeVerdict(
                item_id=item.id, passed=False, score=0.0, reason="",
                raw="", model=self.model, options=options,
                prompt_version=self.prompt_version, error=f"judge call failed: {exc}",
            )
        passed, score, reason, error = parse_judge_output(raw)
        return JudgeVerdict(
            item_id=item.id, passed=passed, score=score, reason=reason, raw=raw,
            model=self.model, options=options, prompt_version=self.prompt_version,
            error=error,
        )

    def score(self, item: BenchmarkItem, response: str) -> ScoreResult:
        """Interface-compatible entry point: returns a ScoreResult (scorer='judge')."""
        return self.evaluate(item, response).as_score_result()
