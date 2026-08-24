"""Contamination detection between two text datasets (e.g. train vs. eval).

Why this exists
---------------
Before any training run, every candidate training row must be checked against the
frozen evaluation benchmark (PROJECT_PLAN.md §16.4). If a benchmark question (or a
lightly reworded version of it) leaks into training data, the post-training score is
inflated and the whole experiment becomes uninterpretable. This module is the
reusable, deterministic core of that gate; `scripts/check_contamination.py` is a thin
CLI over it.

It is intentionally dependency-free (standard library only) and fully deterministic:
the same inputs always produce the same report. No LLM, no network, no randomness.

Two detection methods
---------------------
1. **Normalized exact match** — two records collide if their *normalized* text is
   identical. Normalization (see `normalize_text`) folds away superficial differences:
   case, surrounding/repeated whitespace, common Unicode punctuation variants (smart
   quotes, en/em dashes), and punctuation. It deliberately does NOT stem, drop stop
   words, or reorder tokens, so genuinely different examples stay different.

2. **Fuzzy word n-gram Jaccard** — for near-duplicates / paraphrases. Each text is
   turned into the set of its word n-grams (default n=3). Similarity is the Jaccard
   index |A∩B| / |A∪B| of those sets. Trigrams were chosen because they capture local
   word-order (so unrelated texts that merely share vocabulary score low) while still
   tolerating small edits. Texts shorter than n words fall back to their unigram set so
   short items are still comparable.

Threshold guidance
------------------
`DEFAULT_FUZZY_THRESHOLD = 0.7`. Word-trigram Jaccard ≥ 0.7 means the two texts share
the large majority of their local phrasing — in practice a paraphrase or a copy with
minor edits, not two independently authored questions on the same topic (which
typically score well below 0.3). Lower the threshold to cast a wider (noisier) net;
raise it to flag only near-verbatim copies. The exact-match check is threshold-free.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

DEFAULT_FIELDS: tuple[str, ...] = ("question", "context", "evidence")
DEFAULT_NGRAM: int = 3
DEFAULT_FUZZY_THRESHOLD: float = 0.7

# Unicode punctuation that commonly differs between otherwise-identical text.
_UNICODE_FOLD = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
}
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Return a canonical form for comparison.

    Steps: Unicode NFKC → fold common punctuation variants → lowercase → strip
    remaining punctuation → collapse all whitespace to single spaces → trim.
    Deterministic and idempotent. Kept deliberately conservative so that distinct
    examples are not normalized into false collisions.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _UNICODE_FOLD.items():
        text = text.replace(src, dst)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def record_text(record: dict, fields: tuple[str, ...] | list[str] = DEFAULT_FIELDS) -> str:
    """Join the given fields of a record into one string (missing/None fields skipped).

    For MCQ items the answer choices are included when present so that two items with
    the same stem but different options are not treated as identical.
    """
    parts: list[str] = []
    for f in fields:
        val = record.get(f)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    choices = record.get("choices")
    if isinstance(choices, dict):
        parts.extend(str(v) for v in choices.values())
    return "\n".join(parts)


def word_ngrams(text: str, n: int = DEFAULT_NGRAM) -> set[tuple[str, ...]]:
    """Set of word n-grams of `text` after normalization.

    If the text has fewer than `n` tokens, fall back to unigrams so short texts remain
    comparable (an empty text yields the empty set).
    """
    tokens = normalize_text(text).split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {(t,) for t in tokens}
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(a: str, b: str, n: int = DEFAULT_NGRAM) -> float:
    """Word-n-gram Jaccard similarity in [0, 1]. Two empty texts are defined as 0.0."""
    sa, sb = word_ngrams(a, n), word_ngrams(b, n)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


@dataclass(frozen=True)
class ExactMatch:
    a_id: str
    b_id: str
    normalized: str


@dataclass(frozen=True)
class FuzzyMatch:
    a_id: str
    b_id: str
    similarity: float


@dataclass
class ContaminationReport:
    """Result of comparing dataset A (e.g. train) against dataset B (e.g. eval)."""

    a_count: int
    b_count: int
    ngram: int
    threshold: float
    exact: list[ExactMatch] = field(default_factory=list)
    fuzzy: list[FuzzyMatch] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True iff no exact matches and no fuzzy matches at/above the threshold."""
        return not self.exact and not self.fuzzy

    def summary(self) -> str:
        status = "CLEAN" if self.is_clean else "CONTAMINATION DETECTED"
        return (
            f"{status}: {len(self.exact)} exact, {len(self.fuzzy)} fuzzy "
            f"(>= {self.threshold}) across {self.a_count}x{self.b_count} records "
            f"(word {self.ngram}-grams)"
        )


def _ids(records: list[dict], id_field: str, prefix: str) -> list[str]:
    return [str(r.get(id_field, f"{prefix}{i}")) for i, r in enumerate(records)]


def check_contamination(
    a_records: list[dict],
    b_records: list[dict],
    id_field: str = "id",
    fields: tuple[str, ...] | list[str] = DEFAULT_FIELDS,
    ngram: int = DEFAULT_NGRAM,
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> ContaminationReport:
    """Compare every record in A against every record in B.

    Records are dicts (e.g. rows loaded from JSONL). Comparable text is extracted with
    `record_text(record, fields)`. A pair is reported as an exact match when their
    normalized text is identical, otherwise as a fuzzy match when their word-n-gram
    Jaccard similarity is >= `threshold`. Results are sorted deterministically (exact by
    id pair; fuzzy by descending similarity then id pair).

    This is an O(len(A) * len(B)) comparison, which is more than adequate for the
    dataset sizes in this project (hundreds–thousands of rows). Exact matching is done
    first via a hash map, so only non-exact pairs pay the n-gram cost.
    """
    a_ids = _ids(a_records, id_field, "a#")
    b_ids = _ids(b_records, id_field, "b#")
    a_texts = [record_text(r, fields) for r in a_records]
    b_texts = [record_text(r, fields) for r in b_records]
    a_norm = [normalize_text(t) for t in a_texts]
    b_norm = [normalize_text(t) for t in b_texts]

    # Index B by normalized text for exact matching.
    b_by_norm: dict[str, list[int]] = {}
    for j, nb in enumerate(b_norm):
        if nb:
            b_by_norm.setdefault(nb, []).append(j)

    exact: list[ExactMatch] = []
    exact_pairs: set[tuple[int, int]] = set()
    for i, na in enumerate(a_norm):
        if not na:
            continue
        for j in b_by_norm.get(na, []):
            exact.append(ExactMatch(a_ids[i], b_ids[j], na))
            exact_pairs.add((i, j))

    # Precompute n-gram sets once per side, then fuzzy-compare non-exact pairs.
    a_grams = [word_ngrams(t, ngram) for t in a_texts]
    b_grams = [word_ngrams(t, ngram) for t in b_texts]
    fuzzy: list[FuzzyMatch] = []
    for i in range(len(a_records)):
        sa = a_grams[i]
        if not sa:
            continue
        for j in range(len(b_records)):
            if (i, j) in exact_pairs:
                continue
            sb = b_grams[j]
            if not sb:
                continue
            union = len(sa | sb)
            if not union:
                continue
            sim = len(sa & sb) / union
            if sim >= threshold:
                fuzzy.append(FuzzyMatch(a_ids[i], b_ids[j], round(sim, 4)))

    exact.sort(key=lambda m: (m.a_id, m.b_id))
    fuzzy.sort(key=lambda m: (-m.similarity, m.a_id, m.b_id))
    return ContaminationReport(
        a_count=len(a_records), b_count=len(b_records),
        ngram=ngram, threshold=threshold, exact=exact, fuzzy=fuzzy,
    )
