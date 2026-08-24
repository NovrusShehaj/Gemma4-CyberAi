"""Training dataset schema and loader.

Defines the structure for fine-tuning / SFT training examples:
  * Chat format with system, user, and assistant messages.
  * Explicit metadata capturing task type, evidence requirements, fabrication flags,
    and licensing/provenance.
  * Validation rules ensuring license completeness, non-empty provenance,
    and valid message sequences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

TaskType = Literal[
    "log_analysis",
    "incident_response",
    "detection_engineering",
    "attack_mapping",
    "hallucination_refusal",
    "insufficient_evidence",
    "ctf_methodology",
    "fundamentals",
    "active_directory",
    "windows_security",
    "web_security",
    "network",
    "privilege_escalation",
    "vulnerability_analysis",
    "cryptography",
]

Domain = Literal["blue_team", "offensive_ctf", "general"]
Difficulty = Literal["intro", "intermediate", "advanced"]
Role = Literal["system", "user", "assistant"]


class TrainingMessage(BaseModel):
    """One message in a training conversation."""

    role: Role
    content: str

    @model_validator(mode="after")
    def _validate_content_non_empty(self) -> TrainingMessage:
        if not self.content or not self.content.strip():
            raise ValueError("message content must not be empty")
        return self


class TrainingMetadata(BaseModel):
    """Provenance, taxonomy, and behavior metadata for a training item."""

    task_type: TaskType
    domain: Domain
    difficulty: Difficulty = "intermediate"
    requires_evidence: bool = True
    fabricated_premise: bool = False
    source: str = "original"
    license: str = "CC-BY-4.0"
    provenance: str
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_provenance_and_license(self) -> TrainingMetadata:
        if not self.license or not self.license.strip():
            raise ValueError("license must be non-empty")
        if not self.provenance or not self.provenance.strip():
            raise ValueError("provenance must be non-empty")
        return self


class TrainingItem(BaseModel):
    """One training record (SFT / chat example)."""

    id: str
    schema_version: str = "1.0"
    messages: list[TrainingMessage]
    metadata: TrainingMetadata

    @model_validator(mode="after")
    def _validate_messages_structure(self) -> TrainingItem:
        if len(self.messages) < 2:
            raise ValueError(f"[{self.id}] training item must have at least 2 messages")
        if self.messages[-1].role != "assistant":
            raise ValueError(f"[{self.id}] last message must be from role 'assistant'")
        # Ensure at least one user message exists before assistant
        if not any(m.role == "user" for m in self.messages):
            raise ValueError(f"[{self.id}] training item must include at least one 'user' message")
        return self

    def to_chat_dict(self) -> list[dict[str, str]]:
        """Export messages as list of dicts for HuggingFace / ChatML format."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def render_text(self) -> str:
        """Extract all text for indexing or contamination checking."""
        return "\n".join(m.content for m in self.messages)


def load_training_dataset(path: str | Path) -> list[TrainingItem]:
    """Load and validate a JSONL training dataset file. Raises on duplicate ids or invalid schema."""
    path = Path(path)
    items: list[TrainingItem] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} invalid JSON: {exc}") from exc

            item = TrainingItem(**raw)
            if item.id in seen_ids:
                raise ValueError(f"{path}:{lineno} duplicate id '{item.id}'")
            seen_ids.add(item.id)
            items.append(item)

    if not items:
        raise ValueError(f"{path} contains no training items")
    return items
