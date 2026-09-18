"""Session and undo persistence (plan §18).

JSONL plus a `meta.json` sidecar, not SQLite: a session is a thing you can
`cat`, `grep`, and delete with `rm`, and there is no migration story to own in
v1. Listing is a directory scan; if that ever becomes slow, that is the trigger
to revisit — not a guess made up front.

Privacy properties that are enforced, not documented:

* files are created 0600 and directories 0700;
* messages are appended incrementally, so an interrupted or crashed turn leaves
  a resumable session rather than nothing;
* nothing is synchronised anywhere.

Undo snapshots live beside sessions and expire after 24h. `/undo` restores from
these, never via `git reset` — the working tree contains the user's own changes
and the agent has no business discarding them.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gemma_cyber.agent.errors import SessionError
from gemma_cyber.agent.types import Message, PermissionMode, Usage, new_id

__all__ = ["Session", "SessionMeta", "SessionStore", "UndoStore"]

SESSION_SCHEMA_VERSION = 1
UNDO_TTL_SECONDS = 24 * 60 * 60

_META_NAME = "meta.json"
_MESSAGES_NAME = "messages.jsonl"


def _secure_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)


def _secure_write(path: Path, data: str) -> None:
    """Atomic, 0600, same-directory temp file (plan §65).

    Same-directory matters: `os.replace` is only atomic within a filesystem, and
    a temp file in `/tmp` may be on a different one.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise SessionError(f"could not write {path.name}: {exc.strerror}") from exc


@dataclass(slots=True)
class SessionMeta:
    """The sidecar. Everything except the conversation itself."""

    id: str
    workspace: str
    provider: str = "ollama"
    model: str = ""
    mode: str = PermissionMode.READ_ONLY.value
    profile: str = "general"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    schema_version: int = SESSION_SCHEMA_VERSION
    usage: Usage = field(default_factory=Usage)
    files_touched: list[str] = field(default_factory=list)
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "workspace": self.workspace,
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "profile": self.profile,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage": self.usage.to_dict(),
            "files_touched": self.files_touched,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMeta:
        """Unknown keys are ignored, per the plan's forward-compatibility rule."""
        usage = data.get("usage") or {}
        return cls(
            id=str(data.get("id", "")),
            workspace=str(data.get("workspace", "")),
            provider=str(data.get("provider", "ollama")),
            model=str(data.get("model", "")),
            mode=str(data.get("mode", PermissionMode.READ_ONLY.value)),
            profile=str(data.get("profile", "general")),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            updated_at=float(data.get("updated_at", 0.0) or 0.0),
            schema_version=int(data.get("schema_version", SESSION_SCHEMA_VERSION) or 1),
            usage=Usage(
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
                cached_tokens=int(usage.get("cached_tokens", 0) or 0),
            ),
            files_touched=list(data.get("files_touched") or []),
            title=str(data.get("title", "")),
        )


@dataclass(slots=True)
class Session:
    meta: SessionMeta
    messages: list[Message] = field(default_factory=list)


class SessionStore:
    """Directory-per-session store under ``<data>/sessions/``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- paths --------------------------------------------------------------

    def session_dir(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
        if not safe or safe != session_id:
            raise SessionError(f"invalid session id: {session_id!r}")
        return self.root / safe

    # -- lifecycle ----------------------------------------------------------

    def create(self, meta: SessionMeta | None = None, **kwargs: Any) -> Session:
        meta = meta or SessionMeta(id=new_id(), **kwargs)
        if not meta.id:
            meta.id = new_id()
        directory = self.session_dir(meta.id)
        _secure_mkdir(directory)
        self._write_meta(meta)
        (directory / _MESSAGES_NAME).touch(mode=0o600, exist_ok=True)
        return Session(meta=meta)

    def append(self, session_id: str, message: Message) -> None:
        """Append one message. Called after *every* turn element, not at the end.

        Incremental append is the difference between "the process died and the
        session is recoverable" and "the process died and an hour is gone".
        """
        path = self.session_dir(session_id) / _MESSAGES_NAME
        try:
            _secure_mkdir(path.parent)
            existed = path.exists()
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(message.to_dict(), default=str) + "\n")
            if not existed:
                path.chmod(0o600)
        except OSError as exc:
            raise SessionError(f"could not append to session {session_id}: {exc.strerror}") from exc

    def load(self, session_id: str) -> Session:
        directory = self.session_dir(session_id)
        meta_path = directory / _META_NAME
        if not meta_path.is_file():
            raise SessionError(f"no such session: {session_id}")
        try:
            meta = SessionMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionError(f"session {session_id} metadata is unreadable") from exc

        messages: list[Message] = []
        messages_path = directory / _MESSAGES_NAME
        if messages_path.is_file():
            for line_no, line in enumerate(
                messages_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    messages.append(Message.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    # One corrupt line must not lose the whole conversation.
                    messages.append(Message(
                        role="tool", name="session",
                        content=f"[unreadable record at line {line_no}; skipped]",
                    ))
        return Session(meta=meta, messages=messages)

    def update_meta(self, meta: SessionMeta) -> None:
        meta.updated_at = time.time()
        self._write_meta(meta)

    def _write_meta(self, meta: SessionMeta) -> None:
        directory = self.session_dir(meta.id)
        _secure_mkdir(directory)
        _secure_write(directory / _META_NAME, json.dumps(meta.to_dict(), indent=2, default=str))

    def delete(self, session_id: str) -> None:
        directory = self.session_dir(session_id)
        if not directory.is_dir():
            raise SessionError(f"no such session: {session_id}")
        for child in sorted(directory.rglob("*"), reverse=True):
            try:
                child.unlink() if child.is_file() else child.rmdir()
            except OSError:  # pragma: no cover - best effort
                pass
        try:
            directory.rmdir()
        except OSError as exc:  # pragma: no cover - best effort
            raise SessionError(f"could not delete session {session_id}: {exc.strerror}") from exc

    # -- listing ------------------------------------------------------------

    def list(self, *, workspace: Path | str | None = None, limit: int = 50) -> list[SessionMeta]:
        """Sessions newest-first, optionally filtered to one workspace."""
        if not self.root.is_dir():
            return []
        target = str(Path(workspace).resolve(strict=False)) if workspace else None
        found: list[SessionMeta] = []
        for directory in self.root.iterdir():
            meta_path = directory / _META_NAME
            if not meta_path.is_file():
                continue
            try:
                meta = SessionMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
            if target is not None and meta.workspace != target:
                continue
            found.append(meta)
        found.sort(key=lambda m: m.updated_at, reverse=True)
        return found[:limit]

    def latest(self, *, workspace: Path | str | None = None) -> SessionMeta | None:
        sessions = self.list(workspace=workspace, limit=1)
        return sessions[0] if sessions else None


class UndoStore:
    """Original-file snapshots for `/undo` (plan §13.6).

    One directory per session, one numbered snapshot per write, newest last.
    Restoring pops the newest snapshot for a path, so repeated `/undo` walks
    backwards through that file's history within the session.
    """

    def __init__(self, root: Path, session_id: str) -> None:
        self.root = Path(root)
        self.session_id = session_id
        self.directory = self.root / session_id

    def snapshot(self, rel_path: str, data: bytes) -> Path:
        _secure_mkdir(self.directory)
        stamp = f"{time.time():.6f}".replace(".", "")
        safe = rel_path.replace(os.sep, "__").replace("/", "__")
        target = self.directory / f"{stamp}__{safe}"
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        index = self.directory / "index.jsonl"
        with index.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"path": rel_path, "snapshot": target.name,
                                     "ts": time.time()}) + "\n")
        try:
            index.chmod(0o600)
        except OSError:  # pragma: no cover - platform dependent
            pass
        return target

    def entries(self) -> list[dict[str, Any]]:
        index = self.directory / "index.jsonl"
        if not index.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in index.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:  # pragma: no cover - defensive
                    continue
        return out

    def pop_latest(self, rel_path: str | None = None) -> tuple[str, bytes] | None:
        """Return ``(rel_path, original_bytes)`` for the newest matching snapshot."""
        records = self.entries()
        for index in range(len(records) - 1, -1, -1):
            record = records[index]
            if rel_path is not None and record.get("path") != rel_path:
                continue
            snapshot = self.directory / str(record.get("snapshot", ""))
            if not snapshot.is_file():
                continue
            data = snapshot.read_bytes()
            remaining = records[:index] + records[index + 1:]
            self._rewrite_index(remaining)
            snapshot.unlink(missing_ok=True)
            return str(record.get("path", "")), data
        return None

    def _rewrite_index(self, records: list[dict[str, Any]]) -> None:
        _secure_write(
            self.directory / "index.jsonl",
            "".join(json.dumps(r) + "\n" for r in records),
        )

    @classmethod
    def cleanup_expired(cls, root: Path, *, ttl_seconds: int = UNDO_TTL_SECONDS) -> int:
        """Best-effort TTL sweep on startup. Never raises (plan §13)."""
        removed = 0
        if not Path(root).is_dir():
            return 0
        cutoff = time.time() - ttl_seconds
        for directory in Path(root).iterdir():
            try:
                if not directory.is_dir() or directory.stat().st_mtime > cutoff:
                    continue
                for child in sorted(directory.rglob("*"), reverse=True):
                    child.unlink() if child.is_file() else child.rmdir()
                directory.rmdir()
                removed += 1
            except OSError:
                continue
        return removed
