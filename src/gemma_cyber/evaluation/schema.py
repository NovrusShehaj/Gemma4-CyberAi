"""Benchmark item schema and loader.

A benchmark item is one evaluation case. Fields split into two groups:
  * PROMPT fields (context, evidence, question, choices) -> shown to the model.
  * SCORING + METADATA fields -> used by scorers / provenance, never shown.

Scoring is intentionally DETERMINISTIC in milestone 1 (no LLM judge yet) so the
baseline is fully reproducible. An LLM-judge scorer can be added later without
changing this schema (see PROJECT_PLAN.md §17).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Scorer = Literal["mcq", "keyword", "insufficient_evidence", "hallucination", "factual"]
Split = Literal["dev", "test"]


class BenchmarkItem(BaseModel):
    """One evaluation case.

    Original content only. No proprietary HTB/THM material (PROJECT_PLAN.md §16).
    """

    id: str
    schema_version: str = "1.0"

    # taxonomy
    category: str  # e.g. "fundamentals", "log_analysis", "web_security"
    domain: str  # "blue_team" | "offensive_ctf" | "general"
    difficulty: Literal["intro", "intermediate", "advanced"] = "intermediate"

    # dev/test split (Benchmark v2+). `test` is HELD OUT: it must never be used
    # to iterate on prompts, data, or config. See data/evaluation/README.md.
    # Defaults to "dev" so that pre-split artifacts (e.g. benchmark_v1) still load.
    split: Split = "dev"

    # prompt fields
    question: str
    context: str | None = None
    evidence: str | None = None
    choices: dict[str, str] | None = None  # for mcq: {"A": "...", "B": "..."}

    # scoring
    scorer: Scorer
    answer: str | None = None  # mcq: correct choice key, e.g. "B"
    expected_keywords: list[str] | None = None  # keyword scorer
    keyword_threshold: float = 0.5  # fraction of keywords required to pass

    # `factual` scorer (Benchmark v3+): layered fact/trap checking. A response that
    # contains ANY `forbidden` term hard-fails (score 0) regardless of other content,
    # so a correct-sounding answer that also asserts a wrong ATT&CK ID cannot pass.
    # `required_all` terms must all be present; `required_any` (if given) needs >=1.
    # Terms are matched with word boundaries when they look like ATT&CK IDs, else as
    # case-insensitive substrings. See scorers._score_factual.
    required_all: list[str] | None = None
    required_any: list[str] | None = None
    forbidden: list[str] | None = None

    # provenance / licensing (PROJECT_PLAN.md §16)
    source: str = "original"
    license: str = "CC-BY-4.0"
    provenance: str = "authored for gemma-cyber benchmark_v1"
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_scorer_requirements(self) -> BenchmarkItem:
        if self.scorer == "mcq":
            if not self.choices or not self.answer:
                raise ValueError(f"[{self.id}] mcq items require 'choices' and 'answer'")
            if self.answer not in self.choices:
                raise ValueError(f"[{self.id}] answer '{self.answer}' not in choices")
        if self.scorer == "keyword" and not self.expected_keywords:
            raise ValueError(f"[{self.id}] keyword items require 'expected_keywords'")
        if self.scorer == "factual" and not (self.required_all or self.required_any):
            raise ValueError(
                f"[{self.id}] factual items require 'required_all' and/or 'required_any'"
            )
        return self

    def render_prompt(self) -> str:
        """Assemble the user-facing prompt from context/evidence/question/choices."""
        parts: list[str] = []
        if self.context:
            parts.append(self.context.strip())
        if self.evidence:
            parts.append("Evidence:\n" + self.evidence.strip())
        parts.append(self.question.strip())
        if self.choices:
            opts = "\n".join(f"{k}) {v}" for k, v in sorted(self.choices.items()))
            parts.append(opts)
            parts.append("Answer with the single letter of the best option, then explain briefly.")
        return "\n\n".join(parts)


def load_benchmark(path: str | Path) -> list[BenchmarkItem]:
    """Load and validate a JSONL benchmark file. Raises on duplicate ids."""
    path = Path(path)
    items: list[BenchmarkItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} invalid JSON: {exc}") from exc
            item = BenchmarkItem(**raw)
            if item.id in seen:
                raise ValueError(f"{path}:{lineno} duplicate id '{item.id}'")
            seen.add(item.id)
            items.append(item)
    if not items:
        raise ValueError(f"{path} contains no benchmark items")
    return items
