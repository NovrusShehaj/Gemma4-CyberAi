"""REPL slash commands and the interactive wiring, driven without a terminal.

The input loop itself needs a tty; the commands do not, so they are exercised
directly against a fully wired `AgentApp`.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from gemma_cyber.agent import cli as cli_module
from gemma_cyber.agent.providers.fake import FakeProvider, ScriptedTurn
from gemma_cyber.agent.sessions import UndoStore
from gemma_cyber.agent.types import Message, PermissionMode
from gemma_cyber.agent.ui.render import Renderer
from gemma_cyber.agent.ui.repl import SLASH_HELP, Repl


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "src" / "app.py").write_text("def main():\n    return 42\n", encoding="utf-8")
    monkeypatch.setenv("GEMMA4_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def repl(project: Path, monkeypatch: pytest.MonkeyPatch):
    """A Repl over a real AgentApp, with output captured to a buffer."""
    buffer = io.StringIO()
    renderer = Renderer(console=Console(file=buffer, width=120, force_terminal=False))
    # In production, diagnostics go to a separate stderr console so `--json`
    # stdout stays machine-clean. Join both streams here so assertions can see
    # warnings and errors too.
    renderer.err = Console(file=buffer, width=120, force_terminal=False)
    provider = FakeProvider([ScriptedTurn(text="ok")], name="fake", model="fake-model")
    monkeypatch.setattr(cli_module, "_make_provider", lambda *a, **k: provider)

    options = cli_module.GlobalOptions(mode="workspace", yes=True)
    app = cli_module.build_app(options, provider=provider, renderer=renderer)
    instance = Repl(app)
    instance.buffer = buffer  # type: ignore[attr-defined]
    return instance


def _output(repl: Repl) -> str:
    return " ".join(repl.buffer.getvalue().split())  # type: ignore[attr-defined]


def _reset_output(repl: Repl) -> None:
    repl.buffer.truncate(0)  # type: ignore[attr-defined]
    repl.buffer.seek(0)  # type: ignore[attr-defined]


# -- basics -----------------------------------------------------------------

async def test_help_lists_every_documented_command(repl: Repl) -> None:
    await repl._slash("/help")
    text = _output(repl)
    for command, _ in SLASH_HELP:
        assert command.split()[0] in text


async def test_there_is_no_yolo_command(repl: Repl) -> None:
    assert not any("yolo" in c for c, _ in SLASH_HELP)
    await repl._slash("/yolo")
    assert "unknown command" in _output(repl)


async def test_exit_ends_the_loop(repl: Repl) -> None:
    assert await repl._slash("/exit") is False
    assert await repl._slash("/quit") is False
    assert await repl._slash("/help") is None


async def test_status_reports_the_session(repl: Repl) -> None:
    await repl._slash("/status")
    text = _output(repl)
    assert "fake-model" in text and "workspace" in text
    assert repl.app.state.session_id in text


# -- mode -------------------------------------------------------------------

async def test_mode_switches_the_ceiling_and_the_guard(repl: Repl) -> None:
    await repl._slash("/mode agent")
    assert repl.app.state.mode is PermissionMode.AGENT
    assert repl.app.guard.mode is PermissionMode.AGENT
    await repl._slash("/mode read-only")
    assert repl.app.guard.mode is PermissionMode.READ_ONLY


async def test_mode_cannot_enter_trusted_at_runtime(repl: Repl) -> None:
    """Trusted is a startup risk decision, never a mid-session keystroke."""
    before = repl.app.state.mode
    await repl._slash("/mode trusted")
    assert repl.app.state.mode is before
    assert repl.app.guard.mode is before


async def test_mode_rejects_an_unknown_value(repl: Repl) -> None:
    before = repl.app.state.mode
    await repl._slash("/mode wide-open")
    assert repl.app.state.mode is before


# -- tools and permissions --------------------------------------------------

async def test_tools_reflects_the_current_mode(repl: Repl) -> None:
    await repl._slash("/tools")
    text = _output(repl)
    assert "fs.read" in text and "fs.edit" in text
    assert "hidden by mode" in text  # shell.exec is not available in workspace mode

    _reset_output(repl)
    await repl._slash("/mode agent")
    await repl._slash("/tools")
    assert "shell.exec" in _output(repl)


async def test_permissions_states_the_absolute_controls(repl: Repl) -> None:
    await repl._slash("/permissions")
    text = _output(repl)
    assert "workspace" in text
    assert "refused in every mode" in text


# -- files, context, diff, undo ---------------------------------------------

async def test_files_is_empty_then_populated(repl: Repl) -> None:
    await repl._slash("/files")
    assert "no files touched yet" in _output(repl)

    repl.app.state.record_file_hash("src/app.py", "abc")
    _reset_output(repl)
    await repl._slash("/files")
    assert "src/app.py" in _output(repl)


async def test_context_reports_the_packing(repl: Repl) -> None:
    repl.app.state.messages.append(Message(role="user", content="hello"))
    await repl._slash("/context")
    text = _output(repl)
    assert "estimated tokens" in text and "dropped blocks" in text


async def test_diff_and_undo_round_trip(repl: Repl, project: Path) -> None:
    target = project / "src" / "app.py"
    original = target.read_bytes()
    UndoStore(repl.app.undo_dir, repl.app.state.session_id).snapshot("src/app.py", original)
    target.write_text("def main():\n    return 999\n", encoding="utf-8")

    await repl._slash("/diff")
    assert "999" in _output(repl)

    await repl._slash("/undo")
    assert target.read_bytes() == original
    assert "restored src/app.py" in _output(repl)


async def test_undo_with_nothing_to_undo(repl: Repl) -> None:
    await repl._slash("/undo")
    assert "nothing to undo" in _output(repl)


# -- session and conversation ----------------------------------------------

async def test_clear_forgets_the_conversation_but_keeps_the_session(repl: Repl) -> None:
    repl.app.state.messages.append(Message(role="user", content="hello"))
    session_id = repl.app.state.session_id
    await repl._slash("/clear")
    assert repl.app.state.messages == []
    assert repl.app.state.session_id == session_id
    reloaded = repl.app.sessions.load(session_id)
    assert any("cleared" in m.content for m in reloaded.messages)


async def test_session_ls_and_rm(repl: Repl) -> None:
    await repl._slash("/session ls")
    assert repl.app.state.session_id in _output(repl)
    await repl._slash("/session rm nosuchsession")
    assert "no such session" in _output(repl)


async def test_compact_summarises(repl: Repl) -> None:
    repl.app.state.messages.extend([
        Message(role="user", content="first"),
        Message(role="assistant", content="answer"),
        Message(role="user", content="second"),
        Message(role="assistant", content="answer two"),
    ])
    repl.app.provider._turns.append(ScriptedTurn(text="SUMMARY-TEXT"))  # type: ignore[attr-defined]
    await repl._slash("/compact")
    assert "compacted" in _output(repl)
    assert len(repl.app.state.messages) < 5


async def test_model_and_provider_report_and_switch(repl: Repl) -> None:
    await repl._slash("/model")
    assert "fake-model" in _output(repl)
    await repl._slash("/model other-tag")
    assert repl.app.provider.model == "other-tag"

    _reset_output(repl)
    await repl._slash("/provider nosuchprovider")
    assert "unknown provider" in _output(repl)


# -- event rendering --------------------------------------------------------

async def test_model_tokens_are_rendered_sanitised(repl: Repl) -> None:
    from gemma_cyber.agent.events import EventKind

    repl.app.bus.emit(EventKind.MODEL_TOKEN, text="\x1b]0;PWNED\x07visible text")
    repl.renderer.end_stream()
    text = _output(repl)
    assert "visible text" in text and "PWNED" not in text
