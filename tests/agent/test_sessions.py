"""Session persistence, resume, undo snapshots, and file permissions."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from gemma_cyber.agent.errors import SessionError
from gemma_cyber.agent.sessions import SessionMeta, SessionStore, UndoStore
from gemma_cyber.agent.types import Message, ToolCall, Usage


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


def test_round_trip(store: SessionStore, tmp_path: Path) -> None:
    session = store.create(workspace=str(tmp_path), model="gemma3:4b", mode="read-only")
    store.append(session.meta.id, Message(role="user", content="hello"))
    store.append(session.meta.id, Message(
        role="assistant", content="calling",
        tool_calls=[ToolCall(name="fs.read", arguments={"path": "a.py"}, id="c1")],
    ))
    store.append(session.meta.id, Message(
        role="tool", content="contents", tool_call_id="c1", name="fs.read"))

    loaded = store.load(session.meta.id)
    assert [m.role for m in loaded.messages] == ["user", "assistant", "tool"]
    assert loaded.messages[1].tool_calls[0].name == "fs.read"
    assert loaded.messages[2].tool_call_id == "c1"
    assert loaded.meta.model == "gemma3:4b"


def test_messages_are_appended_incrementally(store: SessionStore, tmp_path: Path) -> None:
    """A crash mid-turn must leave a resumable session, not an empty one."""
    session = store.create(workspace=str(tmp_path))
    store.append(session.meta.id, Message(role="user", content="one"))
    path = store.session_dir(session.meta.id) / "messages.jsonl"
    assert len(path.read_text().splitlines()) == 1
    store.append(session.meta.id, Message(role="assistant", content="two"))
    assert len(path.read_text().splitlines()) == 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_files_are_private(store: SessionStore, tmp_path: Path) -> None:
    session = store.create(workspace=str(tmp_path))
    store.append(session.meta.id, Message(role="user", content="x"))
    directory = store.session_dir(session.meta.id)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for name in ("meta.json", "messages.jsonl"):
        assert stat.S_IMODE((directory / name).stat().st_mode) == 0o600


def test_unknown_keys_in_meta_are_ignored(store: SessionStore, tmp_path: Path) -> None:
    session = store.create(workspace=str(tmp_path))
    path = store.session_dir(session.meta.id) / "meta.json"
    data = json.loads(path.read_text())
    data["invented_by_a_newer_version"] = {"nested": True}
    path.write_text(json.dumps(data))
    assert store.load(session.meta.id).meta.id == session.meta.id


def test_a_corrupt_line_does_not_lose_the_conversation(
    store: SessionStore, tmp_path: Path
) -> None:
    session = store.create(workspace=str(tmp_path))
    store.append(session.meta.id, Message(role="user", content="good"))
    path = store.session_dir(session.meta.id) / "messages.jsonl"
    with path.open("a") as handle:
        handle.write("{not json\n")
    store.append(session.meta.id, Message(role="assistant", content="also good"))
    messages = store.load(session.meta.id).messages
    assert [m.content for m in messages if m.role in ("user", "assistant")] == [
        "good", "also good"
    ]
    assert any("unreadable record" in m.content for m in messages)


def test_listing_is_newest_first_and_workspace_scoped(
    store: SessionStore, tmp_path: Path
) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    first = store.create(workspace=str(a))
    second = store.create(workspace=str(a))
    other = store.create(workspace=str(b))
    second.meta.updated_at = first.meta.updated_at + 100
    store.update_meta(second.meta)

    ids = [m.id for m in store.list(workspace=a)]
    assert ids[0] == second.meta.id
    assert other.meta.id not in ids
    newest = store.latest(workspace=a)
    assert newest is not None and newest.id == second.meta.id
    assert store.latest(workspace=tmp_path / "nothing") is None


def test_delete_removes_everything(store: SessionStore, tmp_path: Path) -> None:
    session = store.create(workspace=str(tmp_path))
    store.append(session.meta.id, Message(role="user", content="x"))
    store.delete(session.meta.id)
    assert not store.session_dir(session.meta.id).exists()
    with pytest.raises(SessionError):
        store.load(session.meta.id)


def test_invalid_session_ids_are_rejected(store: SessionStore) -> None:
    for bad in ("../escape", "a/b", "with space", ""):
        with pytest.raises(SessionError):
            store.session_dir(bad)


def test_meta_survives_usage_and_files(store: SessionStore, tmp_path: Path) -> None:
    session = store.create(workspace=str(tmp_path))
    session.meta.usage = Usage(input_tokens=10, output_tokens=5)
    session.meta.files_touched = ["src/app.py"]
    store.update_meta(session.meta)
    meta = store.load(session.meta.id).meta
    assert meta.usage.input_tokens == 10 and meta.files_touched == ["src/app.py"]


def test_meta_from_dict_tolerates_missing_fields() -> None:
    meta = SessionMeta.from_dict({"id": "x"})
    assert meta.id == "x" and meta.mode == "read-only"


# -- undo -------------------------------------------------------------------

def test_undo_snapshots_restore_newest_first(tmp_path: Path) -> None:
    store = UndoStore(tmp_path / "undo", "session1")
    store.snapshot("a.py", b"v1")
    store.snapshot("a.py", b"v2")
    store.snapshot("b.py", b"other")

    assert store.pop_latest() == ("b.py", b"other")
    assert store.pop_latest("a.py") == ("a.py", b"v2")
    assert store.pop_latest("a.py") == ("a.py", b"v1")
    assert store.pop_latest("a.py") is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_undo_snapshots_are_private(tmp_path: Path) -> None:
    store = UndoStore(tmp_path / "undo", "s")
    snapshot = store.snapshot("a.py", b"secret-ish contents")
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600


def test_undo_ttl_cleanup_is_best_effort(tmp_path: Path) -> None:
    root = tmp_path / "undo"
    store = UndoStore(root, "old")
    store.snapshot("a.py", b"x")
    os.utime(store.directory, (0, 0))
    assert UndoStore.cleanup_expired(root, ttl_seconds=1) == 1
    assert not store.directory.exists()
    # A missing root is not an error.
    assert UndoStore.cleanup_expired(tmp_path / "absent") == 0


def test_undo_of_a_path_never_written_returns_nothing(tmp_path: Path) -> None:
    assert UndoStore(tmp_path / "undo", "s").pop_latest("never.py") is None
