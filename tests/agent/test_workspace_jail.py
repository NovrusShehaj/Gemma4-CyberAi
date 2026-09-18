"""Workspace jail. These assertions are the reason the jail can be trusted."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gemma_cyber.agent.errors import ToolError, WorkspaceEscape
from gemma_cyber.agent.workspace import Workspace, WorkspaceLimits, discover_root

# -- root discovery ---------------------------------------------------------

def test_explicit_cwd_wins(tmp_path: Path) -> None:
    target = tmp_path / "explicit"
    target.mkdir()
    assert discover_root(target) == target.resolve()


def test_walks_up_to_the_git_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    deep = root / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert discover_root(start=deep) == root.resolve()


def test_falls_back_to_cwd_without_a_marker(tmp_path: Path) -> None:
    here = tmp_path / "plain"
    here.mkdir()
    assert discover_root(start=here) == here.resolve()


def test_home_as_root_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    with pytest.raises(WorkspaceEscape, match="HOME"):
        discover_root(fake_home)
    # ...unless explicitly acknowledged.
    assert discover_root(fake_home, allow_home=True) == fake_home.resolve()


def test_filesystem_root_is_refused() -> None:
    with pytest.raises(WorkspaceEscape):
        discover_root(Path("/"))


def test_nonexistent_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ToolError):
        discover_root(tmp_path / "missing")


# -- traversal --------------------------------------------------------------

@pytest.mark.parametrize(
    "attempt",
    ["../outside.txt", "../../etc/passwd", "src/../../outside.txt",
     "./src/../../../etc/hosts", "/etc/passwd", "~/.bashrc",
     "src/./../../outside.txt"],
)
def test_traversal_is_refused(workspace: Workspace, attempt: str) -> None:
    with pytest.raises(WorkspaceEscape):
        workspace.resolve_in_jail(attempt)


def test_nul_byte_in_path_is_refused(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceEscape):
        workspace.resolve_in_jail("src/app\x00.py")


def test_empty_path_is_refused(workspace: Workspace) -> None:
    with pytest.raises(WorkspaceEscape):
        workspace.resolve_in_jail("   ")


def test_paths_inside_the_jail_resolve(workspace: Workspace) -> None:
    resolved = workspace.resolve_in_jail("src/app.py")
    assert resolved.is_file()
    assert workspace.relative(resolved) == "src/app.py"


def test_absolute_path_inside_the_jail_is_allowed(workspace: Workspace) -> None:
    absolute = workspace.root / "src" / "app.py"
    assert workspace.resolve_in_jail(str(absolute)) == absolute.resolve()


def test_a_path_that_does_not_exist_yet_is_still_jailed(workspace: Workspace) -> None:
    # fs.write targets do not exist; the check must still run.
    assert workspace.resolve_in_jail("src/new_file.py").parent == workspace.root / "src"
    with pytest.raises(WorkspaceEscape):
        workspace.resolve_in_jail("../new_file.py")


# -- symlinks ---------------------------------------------------------------

def test_symlink_pointing_outside_is_refused(workspace: Workspace, tmp_path: Path) -> None:
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = workspace.root / "innocent.txt"
    link.symlink_to(outside)
    with pytest.raises(WorkspaceEscape):
        workspace.resolve_in_jail("innocent.txt")


def test_symlink_to_etc_passwd_is_refused(workspace: Workspace) -> None:
    link = workspace.root / "notes.txt"
    link.symlink_to("/etc/passwd")
    with pytest.raises(WorkspaceEscape):
        workspace.resolve_in_jail("notes.txt")


def test_symlinked_parent_directory_is_refused(workspace: Workspace, tmp_path: Path) -> None:
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    (outside_dir / "target.txt").write_text("nope", encoding="utf-8")
    (workspace.root / "vendor").symlink_to(outside_dir, target_is_directory=True)
    with pytest.raises(WorkspaceEscape):
        workspace.resolve_in_jail("vendor/target.txt")


def test_symlink_inside_the_workspace_is_fine(workspace: Workspace) -> None:
    (workspace.root / "alias.py").symlink_to(workspace.root / "src" / "app.py")
    assert workspace.resolve_in_jail("alias.py").name == "app.py"


def test_symlink_loop_is_refused(workspace: Workspace) -> None:
    a = workspace.root / "loop_a"
    b = workspace.root / "loop_b"
    a.symlink_to(b)
    b.symlink_to(a)
    with pytest.raises(WorkspaceEscape):
        workspace.resolve_in_jail("loop_a/x")


def test_enumeration_does_not_follow_directory_symlinks_out(
    workspace: Workspace, tmp_path: Path
) -> None:
    outside = tmp_path / "big"
    outside.mkdir()
    (outside / "leaked.txt").write_text("x", encoding="utf-8")
    (workspace.root / "escape").symlink_to(outside, target_is_directory=True)
    names = [workspace.relative(p) for p in workspace.iter_files()]
    assert not any("leaked" in n for n in names)


def test_enumeration_skips_file_symlinks_that_escape(
    workspace: Workspace, tmp_path: Path
) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("x", encoding="utf-8")
    (workspace.root / "linked.txt").symlink_to(outside)
    names = [workspace.relative(p) for p in workspace.iter_files()]
    assert "linked.txt" not in names


# -- ignore rules -----------------------------------------------------------

def test_gitignore_is_honoured(workspace: Workspace) -> None:
    assert workspace.is_ignored(workspace.root / "secret_notes.txt")
    assert not workspace.is_ignored(workspace.root / "src" / "app.py")


def test_gemma4ignore_is_honoured(workspace_root: Path) -> None:
    (workspace_root / ".gemma4ignore").write_text("*.log\n", encoding="utf-8")
    (workspace_root / "debug.log").write_text("noise", encoding="utf-8")
    ws = Workspace(workspace_root)
    assert ws.is_ignored(workspace_root / "debug.log")


def test_git_directory_is_always_ignored(workspace_root: Path) -> None:
    (workspace_root / ".git").mkdir()
    (workspace_root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    ws = Workspace(workspace_root)
    assert ws.is_ignored(workspace_root / ".git" / "config")
    listed = [ws.relative(p) for p in ws.iter_files(include_hidden=True)]
    assert not any(part == ".git" for name in listed for part in Path(name).parts)
    assert ".gitignore" in listed  # a dotfile that is NOT the .git directory


# -- file access ------------------------------------------------------------

def test_oversize_file_is_refused(workspace_root: Path) -> None:
    big = workspace_root / "big.txt"
    big.write_text("x" * 5000, encoding="utf-8")
    ws = Workspace(workspace_root, limits=WorkspaceLimits(max_file_bytes=1000))
    with pytest.raises(ToolError, match="limit"):
        ws.read_text(ws.resolve_in_jail("big.txt"))


def test_binary_file_is_refused(workspace: Workspace) -> None:
    (workspace.root / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    with pytest.raises(ToolError, match="binary"):
        workspace.read_text(workspace.resolve_in_jail("blob.bin"))


def test_directory_read_is_refused(workspace: Workspace) -> None:
    with pytest.raises(ToolError):
        workspace.read_text(workspace.resolve_in_jail("src"))


def test_relative_outside_the_root_raises(workspace: Workspace, tmp_path: Path) -> None:
    with pytest.raises(WorkspaceEscape):
        workspace.relative(tmp_path / "elsewhere")


# -- stub and hints ---------------------------------------------------------

def test_tree_stub_is_shallow_and_ignore_aware(workspace_root: Path) -> None:
    deep = workspace_root / "src" / "pkg" / "deeper"
    deep.mkdir(parents=True)
    (deep / "hidden_by_depth.py").write_text("x", encoding="utf-8")
    stub = Workspace(workspace_root).tree_stub(depth=2)
    assert "src/" in stub and "README.md" in stub
    assert "secret_notes.txt" not in stub
    assert "hidden_by_depth.py" not in stub


def test_tree_stub_is_capped(workspace_root: Path) -> None:
    for i in range(50):
        (workspace_root / f"file_{i}.txt").write_text("x", encoding="utf-8")
    stub = Workspace(workspace_root, limits=WorkspaceLimits(tree_max_entries=10)).tree_stub()
    assert "truncated" in stub


def test_project_hints(workspace: Workspace) -> None:
    assert "python" in workspace.project_hints()


# -- instruction files ------------------------------------------------------

def test_instruction_files_are_collected_nearest_last(workspace_root: Path) -> None:
    (workspace_root / "GEMMA4.md").write_text("root rules", encoding="utf-8")
    (workspace_root / ".gemma4").mkdir()
    (workspace_root / ".gemma4" / "instructions.md").write_text("overlay", encoding="utf-8")
    files = Workspace(workspace_root).instruction_files()
    labels = [label for label, _ in files]
    assert "workspace:GEMMA4.md" in labels
    assert "workspace:.gemma4/instructions.md" in labels


def test_instruction_files_are_size_capped(workspace_root: Path) -> None:
    (workspace_root / "GEMMA4.md").write_text("x" * 50_000, encoding="utf-8")
    ws = Workspace(workspace_root, limits=WorkspaceLimits(instructions_max_bytes=100))
    # Over the cap: skipped rather than allowed to evict the system policy.
    assert ws.instruction_files() == []


def test_symlink_race_on_read_is_detected(workspace: Workspace, tmp_path: Path) -> None:
    # Simulate check-then-swap: resolve a real file, then replace it with a
    # symlink before the read. O_NOFOLLOW must catch it.
    target = workspace.root / "racy.txt"
    target.write_text("real", encoding="utf-8")
    resolved = workspace.resolve_in_jail("racy.txt")
    outside = tmp_path / "evil.txt"
    outside.write_text("stolen", encoding="utf-8")
    target.unlink()
    os.symlink(outside, target)
    with pytest.raises(WorkspaceEscape, match="symlink"):
        workspace.open_read_bytes(resolved)
