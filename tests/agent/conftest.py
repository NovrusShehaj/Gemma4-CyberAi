"""Shared fixtures: a temporary workspace, an isolated data dir, and a wired agent.

Every test runs against a throwaway workspace and a `GEMMA4_DATA_DIR` inside
`tmp_path`, so nothing ever touches the developer's real sessions, undo
snapshots, or audit log.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from gemma_cyber.agent.audit import NullAuditLog
from gemma_cyber.agent.context import ContextManager
from gemma_cyber.agent.events import EventBus
from gemma_cyber.agent.permissions import PermissionGuard
from gemma_cyber.agent.providers.fake import FakeProvider
from gemma_cyber.agent.runtime import AgentRuntime
from gemma_cyber.agent.sessions import SessionStore
from gemma_cyber.agent.tools.base import (
    AutoApprover,
    ToolContext,
    ToolRegistry,
    ToolRuntime,
)
from gemma_cyber.agent.tools.fs import builtin_fs_tools
from gemma_cyber.agent.tools.shell import ShellExecTool
from gemma_cyber.agent.types import AgentState, PermissionMode, new_id
from gemma_cyber.agent.workspace import Workspace

#: Programs the agent test suite is allowed to actually execute. Everything else
#: is blocked at the spawn boundary — see `no_destructive_subprocesses` below.
_SPAWN_BLOCKED_TOKENS = frozenset(
    {"rm", "rmdir", "dd", "mkfs", "diskutil", "chmod", "chown", "shred", "wipefs",
     "curl", "wget", "mv", "kill", "killall", "shutdown", "reboot", "launchctl",
     "sudo", "su", "defaults", "crontab", "pip", "uv", "brew", "npm", "git"}
)


@pytest.fixture(autouse=True)
def no_destructive_subprocesses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-safe: a destructive command must never reach a real process.

    Adversarial tests script the model into asking for `rm -rf /`, `rm -rf ~/`
    and friends, and assert that policy refuses them. If a policy rule ever
    regresses, the assertion alone is not enough — the command would run against
    the developer's machine before the test could fail. (That happened once
    during development: a gap in the home-directory deny pattern let
    `rm -rf ~/` through and it deleted part of $HOME.)

    So the spawn boundary itself is stubbed: anything matching the destructive
    policy, or naming a destructive program, raises instead of executing. A
    regression now shows up as a loud test failure with nothing deleted.
    """
    from gemma_cyber.agent.permissions import dangerous_command_rule
    from gemma_cyber.agent.tools.shell import ShellExecTool

    original = ShellExecTool._spawn

    def guarded(self, argv, **kwargs):
        joined = " ".join(argv)
        tokens = {Path(part).name.lower() for part in argv}
        if rule := dangerous_command_rule(joined):
            raise AssertionError(
                f"TEST SAFETY: a command matching the destructive policy [{rule}] reached "
                f"the spawn boundary and would have executed: {joined!r}"
            )
        if blocked := tokens & _SPAWN_BLOCKED_TOKENS:
            raise AssertionError(
                f"TEST SAFETY: refusing to execute {sorted(blocked)} in the test suite: "
                f"{joined!r}"
            )
        return original(self, argv, **kwargs)

    monkeypatch.setattr(ShellExecTool, "_spawn", guarded)


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    data = tmp_path / "xdg-data"
    monkeypatch.setenv("GEMMA4_DATA_DIR", str(data))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    yield data


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("def main():\n    return 42\n", encoding="utf-8")
    (root / "README.md").write_text("# Project\n\nA test project.\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (root / ".gitignore").write_text("secret_notes.txt\nbuild/\n", encoding="utf-8")
    (root / "secret_notes.txt").write_text("ignored\n", encoding="utf-8")
    return root


@pytest.fixture
def workspace(workspace_root: Path) -> Workspace:
    return Workspace(workspace_root)


class AgentHarness:
    """A fully wired agent over a FakeProvider — the shape CI always uses."""

    def __init__(
        self,
        *,
        workspace: Workspace,
        provider: FakeProvider,
        mode: PermissionMode,
        data_dir: Path,
        approver=None,
        shell_allowlist: tuple[str, ...] = (),
        with_shell: bool = True,
        max_iterations: int = 12,
    ) -> None:
        self.workspace = workspace
        self.provider = provider
        self.bus = EventBus()
        self.bus.record_history = True
        self.audit = NullAuditLog()
        self.guard = PermissionGuard(workspace, mode, shell_allowlist=shell_allowlist)
        tools = builtin_fs_tools()
        if with_shell:
            tools.append(ShellExecTool())
        self.registry = ToolRegistry(tools)
        self.sessions = SessionStore(data_dir / "sessions")
        self.session = self.sessions.create(
            workspace=str(workspace.root), model=provider.model, mode=mode.value
        )
        self.state = AgentState(session_id=self.session.meta.id, mode=mode)
        self.undo_dir = data_dir / "undo"
        self.approver = approver or AutoApprover()

        def context_factory(state: AgentState) -> ToolContext:
            return ToolContext(
                workspace=workspace, mode=state.mode, cancel=state.cancel,
                session_id=state.session_id, state=state, undo_dir=self.undo_dir,
            )

        self.tool_runtime = ToolRuntime(
            self.registry, self.guard, context_factory=context_factory,
            audit=self.audit, approver=self.approver,
            on_event=lambda kind, fields: self.bus.emit(kind, **fields),
        )
        self.runtime = AgentRuntime(
            provider=provider, registry=self.registry, tool_runtime=self.tool_runtime,
            context=ContextManager(workspace_stub=workspace.tree_stub()),
            sessions=self.sessions, bus=self.bus, guard=self.guard,
            max_iterations=max_iterations,
        )

    async def ask(self, text: str):
        return await self.runtime.run_turn(self.state, text)

    def tool_context(self) -> ToolContext:
        return ToolContext(
            workspace=self.workspace, mode=self.state.mode, cancel=self.state.cancel,
            session_id=self.state.session_id, state=self.state, undo_dir=self.undo_dir,
        )


@pytest.fixture
def make_agent(workspace: Workspace, isolated_data_dir: Path):
    def factory(
        provider: FakeProvider,
        *,
        mode: PermissionMode = PermissionMode.READ_ONLY,
        **kwargs,
    ) -> AgentHarness:
        return AgentHarness(
            workspace=workspace, provider=provider, mode=mode,
            data_dir=isolated_data_dir, **kwargs,
        )

    return factory


@pytest.fixture
def new_session_id() -> str:
    return new_id()
