"""Local audit trail for authorisation decisions (plan §22).

What goes in: who asked (session), what tool, redacted arguments, the guard's
decision and the rule that produced it, how long it took, and the outcome.

What never goes in: prompts, completions, file bodies, environment dumps, API
keys, or anything a secret scanner found. That exclusion is structural — this
module only ever receives the fields listed in :class:`AuditRecord`, and every
string value passes through :func:`redact_value` on the way out.

Files are JSONL, one per UTC day, mode 0600, local only. No cloud sync.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma_cyber.agent.sanitize import redact_value

__all__ = ["AuditLog", "AuditRecord", "NullAuditLog", "agent_logger"]

#: Extras this package's debug logger is allowed to emit. Same defensive shape as
#: `api/logging_setup.py`: an unknown key is dropped, not serialised.
ALLOWED_LOG_EXTRAS = frozenset(
    {"session_id", "tool", "decision", "rule", "latency_ms", "model", "provider",
     "iteration", "mode", "error_code", "exit_code", "event"}
)


def agent_logger() -> logging.Logger:
    return logging.getLogger("gemma_cyber.agent")


@dataclass(slots=True)
class AuditRecord:
    """One authorisation decision. Every field is safe to persist."""

    session_id: str
    tool: str
    decision: str                      # "allow" | "deny" | "approved" | "rejected"
    rule: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    mode: str = ""
    ok: bool | None = None
    error_code: str | None = None
    duration_ms: int = 0
    exit_code: int | None = None
    truncated: bool = False
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        payload = {
            "ts": datetime.fromtimestamp(self.ts, UTC).isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "tool": self.tool,
            "decision": self.decision,
            "rule": self.rule,
            "mode": self.mode,
            # Arguments are redacted AND length-bounded: a `new_string` on
            # fs.edit would otherwise persist a file body into the audit log.
            "args": redact_value(self.args, max_chars=120),
            "ok": self.ok,
            "error_code": self.error_code,
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
            "truncated": self.truncated,
        }
        return json.dumps({k: v for k, v in payload.items() if v is not None}, default=str)


class AuditLog:
    """Append-only JSONL sink. Failures to write are logged, never fatal."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self) -> Path:
        return self.directory / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"

    def write(self, record: AuditRecord) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = self._path()
            existed = path.exists()
            with path.open("a", encoding="utf-8") as handle:
                handle.write(record.to_json() + "\n")
            if not existed:
                try:
                    path.chmod(0o600)
                except OSError:  # pragma: no cover - platform dependent
                    pass
        except OSError as exc:
            # An unwritable audit directory must not break the agent, but it must
            # be visible: a silent audit gap is worse than a noisy one.
            agent_logger().warning(
                "audit write failed: %s", exc.strerror, extra={"tool": record.tool}
            )

    def read_today(self) -> list[dict[str, Any]]:
        """Read back today's records (used by tests and `gemma4 doctor`)."""
        path = self._path()
        if not path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # pragma: no cover - defensive
                    continue
        return out


class NullAuditLog(AuditLog):
    """No-op sink for `gemma4 ask` and unit tests that assert nothing is written."""

    def __init__(self) -> None:
        super().__init__(Path(os.devnull))
        self.records: list[AuditRecord] = []

    def write(self, record: AuditRecord) -> None:
        self.records.append(record)

    def read_today(self) -> list[dict[str, Any]]:
        return [json.loads(r.to_json()) for r in self.records]
