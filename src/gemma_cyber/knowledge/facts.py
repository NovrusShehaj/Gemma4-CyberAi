"""Loader + typed view over `data/knowledge/security_facts.json`.

The JSON is the source of truth; this module gives it a validated, typed surface so
scorers/judge/builder consume the same facts. Intentionally dependency-free beyond
pydantic (already a project dependency) and the standard library.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class AttackTechnique(BaseModel):
    """One MITRE ATT&CK technique fact, verified at authoring time."""

    key: str
    id: str
    name: str
    parent_id: str | None = None
    tactic: str
    tactic_id: str
    aliases: list[str] = Field(default_factory=list)
    # Technique IDs commonly (and wrongly) asserted for THIS technique.
    forbidden_ids: list[str] = Field(default_factory=list)
    confused_with: dict[str, str] = Field(default_factory=dict)
    key_facts: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)


class FactRegistry(BaseModel):
    """All facts, indexed by key. Immutable in spirit — treat as read-only."""

    schema_version: str = "1.0"
    provenance: str = ""
    license: str = "CC-BY-4.0"
    attack_techniques: dict[str, AttackTechnique]
    obsolete_ids: dict[str, str] = Field(default_factory=dict)

    def technique(self, key: str) -> AttackTechnique:
        """Look up a technique by registry key. Raises KeyError if absent."""
        return self.attack_techniques[key]

    def technique_keys(self) -> list[str]:
        return sorted(self.attack_techniques)


def default_facts_path() -> Path:
    """Resolve `data/knowledge/security_facts.json` from the repo root.

    Walks up from this file until it finds the data file, so it works whether the
    package is imported from the repo or installed editable.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "knowledge" / "security_facts.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "could not locate data/knowledge/security_facts.json above " + str(here)
    )


@lru_cache(maxsize=8)
def load_fact_registry(path: str | Path | None = None) -> FactRegistry:
    """Load and validate the fact registry (cached by path).

    Each technique's own JSON key is injected as its `key` field so callers can
    round-trip. Raises on invalid/duplicate structure via pydantic.
    """
    p = Path(path) if path is not None else default_facts_path()
    raw = json.loads(p.read_text(encoding="utf-8"))
    techs = raw.get("attack_techniques", {})
    for k, v in techs.items():
        v.setdefault("key", k)
    return FactRegistry(**raw)
