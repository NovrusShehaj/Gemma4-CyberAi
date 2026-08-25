"""Structured, verified cybersecurity fact registry.

A single source of truth for high-risk facts (ATT&CK technique IDs, tactic mappings,
Kerberos mechanics) so a fact is defined ONCE and reused by benchmark scorers, the LLM
judge rubric, and dataset authoring instead of being maintained independently (and
drifting) in each layer. See `docs/decisions.md`.
"""

from __future__ import annotations

from gemma_cyber.knowledge.facts import (
    AttackTechnique,
    FactRegistry,
    default_facts_path,
    load_fact_registry,
)

__all__ = [
    "AttackTechnique",
    "FactRegistry",
    "default_facts_path",
    "load_fact_registry",
]
