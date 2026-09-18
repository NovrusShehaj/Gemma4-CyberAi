"""Filesystem tools, including the hash-checked edit contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gemma_cyber.agent.errors import PatchConflict, ToolError, WorkspaceEscape
from gemma_cyber.agent.providers.fake import FakeProvider
from gemma_cyber.agent.sessions import UndoStore
from gemma_cyber.agent.tools.fs import (
    EditArgs,
    EditFileTool,
    GlobArgs,
    GlobTool,
    GrepArgs,
    GrepTool,
    ReadArgs,
    ReadFileTool,
    WriteArgs,
    WriteFileTool,
)
from gemma_cyber.agent.types import PermissionMode


@pytest.fixture
def ctx(make_agent):
    return make_agent(FakeProvider([]), mode=PermissionMode.AGENT).tool_context()


# -- fs.read ----------------------------------------------------------------

def test_read_returns_numbered_lines_and_a_hash(ctx) -> None:
    result = ReadFileTool().run(ReadArgs(path="src/app.py"), ctx)
    assert result.ok
    assert "def main" in result.content
    assert "     1\t" in result.content
    digest = hashlib.sha256((ctx.workspace.root / "src/app.py").read_bytes()).hexdigest()
    assert digest in result.content
    # The hash is recorded so fs.edit can prove freshness.
    assert ctx.state.file_hashes["src/app.py"] == digest


def test_read_supports_a_line_window(ctx) -> None:
    (ctx.workspace.root / "many.txt").write_text("\n".join(str(i) for i in range(1, 51)))
    result = ReadFileTool().run(ReadArgs(path="many.txt", start_line=10, end_line=12), ctx)
    assert "    10\t10" in result.content and "    13\t13" not in result.content


def test_read_missing_file_is_a_tool_error(ctx) -> None:
    with pytest.raises(ToolError, match="no such file"):
        ReadFileTool().run(ReadArgs(path="nope.py"), ctx)


def test_read_directory_is_a_tool_error(ctx) -> None:
    with pytest.raises(ToolError, match="directory"):
        ReadFileTool().run(ReadArgs(path="src"), ctx)


def test_read_binary_is_refused(ctx) -> None:
    (ctx.workspace.root / "b.bin").write_bytes(b"\x00\xff" * 100)
    with pytest.raises(ToolError, match="binary"):
        ReadFileTool().run(ReadArgs(path="b.bin"), ctx)


def test_read_outside_the_jail_is_refused(ctx) -> None:
    with pytest.raises(WorkspaceEscape):
        ReadFileTool().run(ReadArgs(path="../../etc/passwd"), ctx)


# -- fs.glob ----------------------------------------------------------------

def test_glob_finds_files_and_respects_gitignore(ctx) -> None:
    result = GlobTool().run(GlobArgs(pattern="*.py"), ctx)
    assert "src/app.py" in result.content
    assert "secret_notes" not in result.content


def test_glob_hidden_files_need_opting_in(ctx) -> None:
    plain = GlobTool().run(GlobArgs(pattern="*"), ctx)
    assert ".gitignore" not in plain.content
    explicit = GlobTool().run(GlobArgs(pattern=".*"), ctx)
    assert ".gitignore" in explicit.content


def test_glob_rejects_escaping_patterns(ctx) -> None:
    for pattern in ("/etc/*", "../*"):
        with pytest.raises(WorkspaceEscape):
            GlobTool().run(GlobArgs(pattern=pattern), ctx)


def test_glob_caps_results(ctx) -> None:
    for i in range(30):
        (ctx.workspace.root / f"f{i}.txt").write_text("x")
    ctx.glob_max_results = 5
    result = GlobTool().run(GlobArgs(pattern="*.txt"), ctx)
    assert "capped at 5" in result.content


def test_glob_no_match_is_not_an_error(ctx) -> None:
    assert "no files match" in GlobTool().run(GlobArgs(pattern="*.zzz"), ctx).content


# -- fs.grep ----------------------------------------------------------------

def test_grep_finds_matches(ctx) -> None:
    result = GrepTool().run(GrepArgs(pattern=r"def\s+main"), ctx)
    assert "src/app.py" in result.content


def test_grep_python_fallback_matches_ripgrep_shape(ctx, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)  # force the Python scan
    result = GrepTool().run(GrepArgs(pattern="main"), ctx)
    assert "src/app.py:1:" in result.content


def test_grep_skips_ignored_and_binary_files(ctx, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    (ctx.workspace.root / "secret_notes.txt").write_text("needle here")
    (ctx.workspace.root / "blob.bin").write_bytes(b"needle\x00")
    result = GrepTool().run(GrepArgs(pattern="needle"), ctx)
    assert "no matches" in result.content


def test_grep_invalid_regex_is_a_tool_error(ctx) -> None:
    with pytest.raises(ToolError, match="regular expression"):
        GrepTool().run(GrepArgs(pattern="([unclosed"), ctx)


def test_grep_path_is_jailed(ctx) -> None:
    with pytest.raises(WorkspaceEscape):
        GrepTool().run(GrepArgs(pattern="x", path="../.."), ctx)


# -- fs.write ---------------------------------------------------------------

def test_write_creates_a_new_file(ctx) -> None:
    result = WriteFileTool().run(WriteArgs(path="notes/new.md", content="hello\n"), ctx)
    assert result.ok
    assert (ctx.workspace.root / "notes" / "new.md").read_text() == "hello\n"


def test_write_refuses_to_overwrite(ctx) -> None:
    with pytest.raises(ToolError, match="already exists"):
        WriteFileTool().run(WriteArgs(path="README.md", content="clobbered"), ctx)
    assert "# Project" in (ctx.workspace.root / "README.md").read_text()


def test_write_is_jailed(ctx) -> None:
    with pytest.raises(WorkspaceEscape):
        WriteFileTool().run(WriteArgs(path="../escaped.txt", content="x"), ctx)


def test_write_respects_the_size_limit(ctx) -> None:
    ctx.max_file_bytes = 10
    with pytest.raises(ToolError, match="limit"):
        WriteFileTool().run(WriteArgs(path="big.txt", content="x" * 100), ctx)


# -- fs.edit ----------------------------------------------------------------

def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_edit_applies_with_the_correct_hash(ctx) -> None:
    target = ctx.workspace.root / "src" / "app.py"
    result = EditFileTool().run(EditArgs(
        path="src/app.py", old_string="return 42", new_string="return 43",
        expected_sha256=_digest(target),
    ), ctx)
    assert result.ok
    assert "return 43" in target.read_text()
    assert "-    return 42" in result.content and "+    return 43" in result.content


def test_edit_with_a_stale_hash_never_writes(ctx) -> None:
    target = ctx.workspace.root / "src" / "app.py"
    before = target.read_text()
    with pytest.raises(PatchConflict, match="changed since it was read"):
        EditFileTool().run(EditArgs(
            path="src/app.py", old_string="return 42", new_string="return 43",
            expected_sha256="0" * 64,
        ), ctx)
    assert target.read_text() == before


def test_file_changed_between_read_and_edit_conflicts(ctx) -> None:
    target = ctx.workspace.root / "src" / "app.py"
    read = ReadFileTool().run(ReadArgs(path="src/app.py"), ctx)
    stale_hash = read.metadata["sha256"]
    # The user edits the file in another editor.
    target.write_text("def main():\n    return 99\n", encoding="utf-8")
    with pytest.raises(PatchConflict):
        EditFileTool().run(EditArgs(
            path="src/app.py", old_string="return 42", new_string="return 43",
            expected_sha256=stale_hash,
        ), ctx)
    assert "return 99" in target.read_text()


def test_edit_requires_a_unique_match(ctx) -> None:
    target = ctx.workspace.root / "dup.py"
    target.write_text("x = 1\nx = 1\n", encoding="utf-8")
    with pytest.raises(PatchConflict, match="appears 2 times"):
        EditFileTool().run(EditArgs(
            path="dup.py", old_string="x = 1", new_string="x = 2",
            expected_sha256=_digest(target),
        ), ctx)
    assert target.read_text() == "x = 1\nx = 1\n"


def test_edit_missing_old_string_conflicts(ctx) -> None:
    target = ctx.workspace.root / "src" / "app.py"
    with pytest.raises(PatchConflict, match="not found"):
        EditFileTool().run(EditArgs(
            path="src/app.py", old_string="nonexistent", new_string="x",
            expected_sha256=_digest(target),
        ), ctx)


def test_edit_preserves_file_mode(ctx) -> None:
    script = ctx.workspace.root / "run.sh"
    script.write_text("#!/bin/sh\necho old\n", encoding="utf-8")
    script.chmod(0o755)
    EditFileTool().run(EditArgs(
        path="run.sh", old_string="echo old", new_string="echo new",
        expected_sha256=_digest(script),
    ), ctx)
    assert script.stat().st_mode & 0o111, "executable bit must survive an edit"


def test_edit_writes_an_undo_snapshot(ctx) -> None:
    target = ctx.workspace.root / "src" / "app.py"
    original = target.read_bytes()
    EditFileTool().run(EditArgs(
        path="src/app.py", old_string="return 42", new_string="return 43",
        expected_sha256=_digest(target),
    ), ctx)
    store = UndoStore(Path(ctx.undo_dir), ctx.session_id)
    popped = store.pop_latest("src/app.py")
    assert popped is not None
    rel, data = popped
    assert rel == "src/app.py" and data == original


def test_edit_updates_the_recorded_hash_for_chained_edits(ctx) -> None:
    target = ctx.workspace.root / "src" / "app.py"
    EditFileTool().run(EditArgs(
        path="src/app.py", old_string="return 42", new_string="return 43",
        expected_sha256=_digest(target),
    ), ctx)
    # The new hash the tool reported must be usable for the next edit.
    EditFileTool().run(EditArgs(
        path="src/app.py", old_string="return 43", new_string="return 44",
        expected_sha256=ctx.state.file_hashes["src/app.py"],
    ), ctx)
    assert "return 44" in target.read_text()


def test_edit_refuses_binary(ctx) -> None:
    blob = ctx.workspace.root / "b.bin"
    blob.write_bytes(b"\x00abc")
    with pytest.raises(ToolError):
        EditFileTool().run(EditArgs(
            path="b.bin", old_string="abc", new_string="xyz",
            expected_sha256=_digest(blob),
        ), ctx)


def test_edit_leaves_no_temp_file_behind_on_failure(ctx) -> None:
    with pytest.raises(PatchConflict):
        EditFileTool().run(EditArgs(
            path="src/app.py", old_string="return 42", new_string="x",
            expected_sha256="0" * 64,
        ), ctx)
    assert not list((ctx.workspace.root / "src").glob(".*.gemma4.tmp"))
