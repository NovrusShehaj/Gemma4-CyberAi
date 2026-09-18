"""shell.exec: environment scrubbing, jail, timeouts, cancellation, refusals."""

from __future__ import annotations

import os
import sys
import time

import pytest

from gemma_cyber.agent.errors import (
    CommandTimeout,
    PermissionDenied,
    ToolError,
    UserCancelled,
    WorkspaceEscape,
)
from gemma_cyber.agent.providers.fake import FakeProvider
from gemma_cyber.agent.tools.shell import (
    ENV_ALLOWLIST,
    ShellArgs,
    ShellExecTool,
    build_child_env,
)
from gemma_cyber.agent.types import PermissionMode

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics")


@pytest.fixture
def ctx(make_agent):
    return make_agent(FakeProvider([]), mode=PermissionMode.AGENT).tool_context()


# -- environment scrub ------------------------------------------------------

def test_env_is_built_from_an_allowlist() -> None:
    parent = {
        "PATH": "/usr/bin", "HOME": "/home/x", "LANG": "en_US.UTF-8", "LC_ALL": "C",
        "TERM": "xterm", "VIRTUAL_ENV": "/venv",
        "AWS_SECRET_ACCESS_KEY": "leak", "AWS_SESSION_TOKEN": "leak",
        "SSH_AUTH_SOCK": "/tmp/ssh", "GPG_AGENT_INFO": "x", "GNUPGHOME": "/g",
        "GEMMA_CYBER_API_TOKEN": "leak", "GEMMA4_API_KEY": "leak",
        "OPENAI_API_KEY": "leak", "MY_SERVICE_TOKEN": "leak", "DB_PASSWORD": "leak",
        "GITHUB_TOKEN": "leak", "RANDOM_UNLISTED": "value",
    }
    child = build_child_env(parent)
    assert set(child) >= {"PATH", "HOME", "LANG", "LC_ALL", "TERM", "VIRTUAL_ENV"}
    assert "leak" not in "".join(child.values())
    assert "RANDOM_UNLISTED" not in child
    for key in child:
        assert key in ENV_ALLOWLIST or key.startswith("LC_") or key in {
            "NO_COLOR", "CLICOLOR", "GEMMA4_AGENT", "PWD"
        }


def test_env_scrub_survives_a_widened_allowlist(monkeypatch) -> None:
    """Defence in depth: even if the allowlist grows, credentials stay out."""
    monkeypatch.setattr(
        "gemma_cyber.agent.tools.shell.ENV_ALLOWLIST",
        frozenset({"PATH", "AWS_SECRET_ACCESS_KEY", "MY_API_KEY"}),
    )
    child = build_child_env({"PATH": "/bin", "AWS_SECRET_ACCESS_KEY": "x", "MY_API_KEY": "y"})
    assert child.get("AWS_SECRET_ACCESS_KEY") is None
    assert child.get("MY_API_KEY") is None


def test_the_child_process_really_cannot_see_a_secret(ctx, monkeypatch) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret-value")
    result = ShellExecTool().run(ShellArgs(argv=["env"]), ctx)
    assert "super-secret-value" not in result.content


# -- execution --------------------------------------------------------------

def test_argv_command_runs_and_captures_streams(ctx) -> None:
    result = ShellExecTool().run(
        ShellArgs(argv=[sys.executable, "-c",
                        "import sys; print('out'); print('err', file=sys.stderr)"]),
        ctx,
    )
    assert result.ok
    assert "--- stdout ---" in result.content and "out" in result.content
    assert "--- stderr ---" in result.content and "err" in result.content
    assert result.metadata["exit_code"] == 0


def test_nonzero_exit_is_reported_not_raised(ctx) -> None:
    result = ShellExecTool().run(ShellArgs(argv=[sys.executable, "-c", "raise SystemExit(3)"]), ctx)
    assert result.ok  # the tool succeeded; the command failed
    assert result.metadata["exit_code"] == 3


def test_command_runs_in_the_workspace_root(ctx) -> None:
    result = ShellExecTool().run(ShellArgs(argv=["pwd"]), ctx)
    assert str(ctx.workspace.root) in result.content


def test_cwd_is_jailed(ctx) -> None:
    with pytest.raises(WorkspaceEscape):
        ShellExecTool().run(ShellArgs(argv=["pwd"], cwd="../.."), ctx)


def test_cwd_inside_the_workspace_is_allowed(ctx) -> None:
    result = ShellExecTool().run(ShellArgs(argv=["pwd"], cwd="src"), ctx)
    assert "src" in result.content


def test_missing_binary_is_a_tool_error(ctx) -> None:
    with pytest.raises(ToolError, match="not found"):
        ShellExecTool().run(ShellArgs(argv=["definitely-not-a-real-binary-xyz"]), ctx)


def test_shell_string_path_works_for_pipes(ctx) -> None:
    result = ShellExecTool().run(ShellArgs(command="echo hello | tr a-z A-Z"), ctx)
    assert "HELLO" in result.content


def test_shell_string_can_be_disabled_for_noninteractive_runs(ctx) -> None:
    ctx.extra["allow_shell_string"] = False
    with pytest.raises(PermissionDenied, match="noninteractive"):
        ShellExecTool().run(ShellArgs(command="echo hi"), ctx)
    # argv still works.
    assert ShellExecTool().run(ShellArgs(argv=["echo", "hi"]), ctx).ok


# -- refusals ---------------------------------------------------------------

def test_the_tool_re_checks_the_destructive_policy(ctx) -> None:
    """Defence in depth: even called directly, past the guard, it refuses."""
    with pytest.raises(PermissionDenied, match="policy"):
        ShellExecTool().run(ShellArgs(command="rm -rf /"), ctx)


@pytest.mark.parametrize("program", ["vim", "less", "top", "psql", "sudo"])
def test_interactive_commands_are_refused(ctx, program: str) -> None:
    with pytest.raises(ToolError, match="interactive|refused"):
        ShellExecTool().run(ShellArgs(argv=[program]), ctx)


def test_ssh_needs_batch_mode(ctx) -> None:
    with pytest.raises(ToolError, match="BatchMode"):
        ShellExecTool().run(ShellArgs(argv=["ssh", "host", "ls"]), ctx)


def test_argv_and_command_are_mutually_exclusive() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ShellArgs(argv=["ls"], command="ls")
    with pytest.raises(ValidationError):
        ShellArgs()


# -- timeouts and cancellation ---------------------------------------------

def test_timeout_kills_the_command(ctx) -> None:
    ctx.shell_timeout_s = 0.5
    started = time.monotonic()
    with pytest.raises(CommandTimeout):
        ShellExecTool().run(ShellArgs(argv=[sys.executable, "-c",
                                            "import time; time.sleep(30)"]), ctx)
    assert time.monotonic() - started < 10


def test_timeout_kills_the_whole_process_group(ctx, tmp_path) -> None:
    """A child that outlives its parent would keep writing after the timeout."""
    marker = tmp_path / "child_still_alive"
    script = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', "
        f"\"import time,pathlib; time.sleep(3); pathlib.Path(r'{marker}').write_text('x')\"])\n"
        "time.sleep(30)\n"
    )
    ctx.shell_timeout_s = 0.5
    with pytest.raises(CommandTimeout):
        ShellExecTool().run(ShellArgs(argv=[sys.executable, "-c", script]), ctx)
    time.sleep(4)
    assert not marker.exists(), "the grandchild survived the process-group kill"


def test_cancel_token_stops_a_running_command(ctx) -> None:
    import threading

    threading.Timer(0.3, ctx.cancel.cancel).start()
    with pytest.raises(UserCancelled):
        ShellExecTool().run(ShellArgs(argv=[sys.executable, "-c",
                                            "import time; time.sleep(30)"]), ctx)


# -- output hygiene ---------------------------------------------------------

def test_output_is_bounded(ctx) -> None:
    ctx.max_output_bytes = 2048
    result = ShellExecTool().run(
        ShellArgs(argv=[sys.executable, "-c", "print('x' * 100000)"]), ctx
    )
    assert len(result.content) < 100000


def test_stdin_is_closed_so_a_command_cannot_block_on_input(ctx) -> None:
    result = ShellExecTool().run(
        ShellArgs(argv=[sys.executable, "-c",
                        "import sys; print(repr(sys.stdin.read()))"]), ctx
    )
    assert "''" in result.content
