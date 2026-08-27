"""Model registry + promotion lifecycle.

A model *version* (e.g. ``gemma3-cyber:v0.2``) is more than an Ollama tag: it has
a provenance (base model, dataset version, git commit, GGUF checksum) and a
promotion **stage**. This registry is the traceable link between "a benchmark
scorecard" and "the thing serving traffic", so a version promoted to production is
identifiable and reproducible (Phase 5 requirement).

Promotion lifecycle (one-directional, gated):

    experimental -> evaluated -> candidate -> production

Rules enforced here:
  * Transitions may only move forward one step, EXCEPT you may always
    ``archive`` (retire) a version, and rollback from ``production`` to
    ``candidate`` is allowed (an incident action).
  * Promotion to ``candidate`` or ``production`` requires ``passed_eval=True`` on
    the record — you cannot ship a version that has not cleared its gate. This is
    the "only models satisfying the evaluation gates are eligible for promotion"
    rule made mechanical.
  * At most one version may be ``production`` at a time; promoting a new one
    demotes the incumbent to ``candidate`` (recorded, not silently dropped).

Storage is a single JSON file (boring, diff-able, no database). Concurrent
writers are out of scope for a solo/small-team tool.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from gemma_cyber.inference.errors import RegistryError

Stage = Literal["experimental", "evaluated", "candidate", "production", "archived"]

STAGES: tuple[Stage, ...] = (
    "experimental",
    "evaluated",
    "candidate",
    "production",
    "archived",
)

# Allowed forward transitions. `archived` is reachable from anywhere; production
# may roll back to candidate.
_ALLOWED: dict[Stage, set[Stage]] = {
    "experimental": {"evaluated", "archived"},
    "evaluated": {"candidate", "archived"},
    "candidate": {"production", "evaluated", "archived"},
    "production": {"candidate", "archived"},
    "archived": {"experimental"},  # un-retire to re-evaluate
}

# Stages that require a passing evaluation on record before entry.
_REQUIRES_EVAL: set[Stage] = {"candidate", "production"}


class PromotionError(RegistryError):
    """A requested stage transition is not allowed (bad transition or ungated)."""


class RegistryReadOnlyError(RegistryError):
    """A write was attempted against a registry opened read-only (GitOps mode)."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ModelRecord:
    """One registered model version and its provenance + lifecycle stage."""

    version: str  # canonical id, e.g. "gemma3-cyber:v0.2" (also the Ollama tag by default)
    stage: Stage = "experimental"
    ollama_tag: str = ""  # concrete tag to serve; defaults to `version`
    base_model: str = "gemma3:4b"
    dataset_version: str | None = None  # e.g. "sft_v0.2"
    git_commit: str | None = None
    gguf_sha256: str | None = None
    experiment: str | None = None  # e.g. "exp-002-gemma3-cyber-v0.2"
    passed_eval: bool = False  # gate flag; set when a scorecard clears the criteria
    eval_ref: str | None = None  # path to the scorecard/results proving the gate
    notes: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    history: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.ollama_tag:
            self.ollama_tag = self.version
        if self.stage not in STAGES:
            raise RegistryError(f"unknown stage {self.stage!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ModelRecord:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


class ModelRegistry:
    """JSON-backed registry of model versions and their promotion stages."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._records: dict[str, ModelRecord] = {}
        if self.path.exists():
            self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise RegistryError(f"registry at {self.path} is unreadable: {exc}") from exc
        records = raw.get("models", raw) if isinstance(raw, dict) else raw
        for item in records:
            rec = ModelRecord.from_dict(item)
            self._records[rec.version] = rec

    def save(self) -> None:
        """Persist atomically: write a sibling temp file, fsync, then ``os.replace``.

        ``os.replace`` is atomic on the same filesystem, so a crash mid-write can
        never leave a truncated/corrupt registry — a reader sees either the old or
        the new file, never a partial one. The file is created with owner-only
        permissions (0600) so a host-mounted registry isn't world-readable.
        Refuses to write when opened ``read_only`` (GitOps/hosted mode).
        """
        if self.read_only:
            raise RegistryReadOnlyError(
                f"registry at {self.path} is read-only; manage it via reviewed "
                "source-controlled changes (GitOps), not runtime mutation"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "gemma-cyber/model-registry@1",
            "updated_at": _now(),
            "models": [r.to_dict() for r in self._records.values()],
        }
        data = json.dumps(payload, indent=2) + "\n"
        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".registry.", suffix=".tmp"
        )
        try:
            os.chmod(tmp, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            # Leave the existing registry untouched; clean up the temp file.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- reads --------------------------------------------------------------

    def list(self, stage: Stage | None = None) -> list[ModelRecord]:
        recs = list(self._records.values())
        if stage is not None:
            recs = [r for r in recs if r.stage == stage]
        return sorted(recs, key=lambda r: r.version)

    def get(self, version: str) -> ModelRecord:
        try:
            return self._records[version]
        except KeyError:
            raise RegistryError(f"no registered model version {version!r}") from None

    def production(self) -> ModelRecord | None:
        for r in self._records.values():
            if r.stage == "production":
                return r
        return None

    def resolve(self, ref: str) -> str:
        """Resolve a serving reference to a concrete Ollama tag.

        ``ref`` may be a stage alias (``"production"``/``"candidate"``), a
        registered version, or an unregistered raw tag (returned as-is so ad-hoc
        models still work). Stage aliases require exactly one matching record.
        """
        if ref in ("production", "candidate", "evaluated", "experimental"):
            matches = [r for r in self._records.values() if r.stage == ref]
            if not matches:
                raise RegistryError(f"no model in stage {ref!r}")
            if len(matches) > 1 and ref != "production":
                raise RegistryError(
                    f"{len(matches)} models in stage {ref!r}; specify a version"
                )
            return matches[0].ollama_tag
        if ref in self._records:
            return self._records[ref].ollama_tag
        return ref  # raw tag passthrough

    # -- writes -------------------------------------------------------------

    def register(
        self, record: ModelRecord, *, overwrite: bool = False, subject: str | None = None
    ) -> ModelRecord:
        if record.version in self._records and not overwrite:
            raise RegistryError(
                f"version {record.version!r} already registered (use overwrite=True)"
            )
        record.updated_at = _now()
        # A brand-new record starts its audit trail with a uniform history entry so
        # the creating subject is attributable; ``from``==``to`` marks a creation.
        if not record.history:
            record.history.append(
                {
                    "from": record.stage,
                    "to": record.stage,
                    "at": _now(),
                    "reason": "registered",
                    "subject": subject or "unknown",
                }
            )
        self._records[record.version] = record
        self.save()
        return record

    def mark_evaluated(
        self,
        version: str,
        *,
        passed: bool,
        eval_ref: str | None = None,
        subject: str | None = None,
    ) -> ModelRecord:
        """Record an evaluation outcome and, on a pass, advance to ``evaluated``.

        This is the gate input: ``passed`` becomes the ``passed_eval`` flag that
        promotion to candidate/production later checks. A failing eval is recorded
        (kept honest) but does not advance the stage.
        """
        rec = self.get(version)
        rec.passed_eval = passed
        rec.eval_ref = eval_ref
        if passed and rec.stage == "experimental":
            self._transition(
                rec, "evaluated", reason="passed evaluation gate", subject=subject
            )
        else:
            rec.updated_at = _now()
        self.save()
        return rec

    def promote(
        self,
        version: str,
        to: Stage,
        *,
        reason: str | None = None,
        subject: str | None = None,
    ) -> ModelRecord:
        rec = self.get(version)
        if to not in STAGES:
            raise PromotionError(f"unknown stage {to!r}")
        if to not in _ALLOWED.get(rec.stage, set()):
            raise PromotionError(
                f"illegal transition {rec.stage!r} -> {to!r} for {version!r}; "
                f"allowed: {sorted(_ALLOWED.get(rec.stage, set()))}"
            )
        if to in _REQUIRES_EVAL and not rec.passed_eval:
            raise PromotionError(
                f"cannot promote {version!r} to {to!r}: no passing evaluation on record "
                f"(mark_evaluated(passed=True) first)"
            )
        # Single-production invariant: demote the incumbent.
        if to == "production":
            incumbent = self.production()
            if incumbent is not None and incumbent.version != version:
                self._transition(
                    incumbent, "candidate", reason=f"demoted for {version}", subject=subject
                )
        self._transition(rec, to, reason=reason or "manual promotion", subject=subject)
        self.save()
        return rec

    def _transition(
        self, rec: ModelRecord, to: Stage, *, reason: str, subject: str | None = None
    ) -> None:
        rec.history.append(
            {
                "from": rec.stage,
                "to": to,
                "at": _now(),
                "reason": reason,
                "subject": subject or "unknown",
            }
        )
        rec.stage = to
        rec.updated_at = _now()
