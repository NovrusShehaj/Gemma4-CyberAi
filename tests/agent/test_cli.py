"""`gemma4` command surface: exit codes, machine output, and the trusted gate.

Every test runs with a FakeProvider, so nothing here needs Ollama.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gemma_cyber.agent import cli as cli_module
from gemma_cyber.agent.providers.fake import FakeProvider, ScriptedTurn
from gemma_cyber.agent.types import PermissionMode

runner = CliRunner()


def _tool_call(name: str, **arguments: object) -> str:
    return f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "src" / "app.py").write_text("def main():\n    return 42\n", encoding="utf-8")
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")
    monkeypatch.setenv("GEMMA4_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    """Install a FakeProvider in place of whatever the config selects."""
    holder: dict[str, FakeProvider] = {}

    def install(turns: list[ScriptedTurn]) -> FakeProvider:
        provider = FakeProvider(turns, name="fake", model="fake-model")
        holder["provider"] = provider
        monkeypatch.setattr(cli_module, "_make_provider", lambda *a, **k: provider)
        return provider

    return install


def _invoke(*args: str):
    return runner.invoke(cli_module.app, list(args), catch_exceptions=False)


# -- basics -----------------------------------------------------------------

def test_help_lists_the_command_hierarchy() -> None:
    result = _invoke("--help")
    assert result.exit_code == 0
    for command in ("ask", "run", "resume", "session", "models", "tools", "doctor", "config"):
        assert command in result.stdout


def test_version() -> None:
    result = _invoke("--version")
    assert result.exit_code == 0 and "gemma4" in result.stdout


def test_there_is_no_yolo_command() -> None:
    """Trusted mode is a startup risk decision, never a convenience command."""
    assert "/yolo" not in (_invoke("--help").stdout)
    assert _invoke("yolo").exit_code != 0


# -- run --json -------------------------------------------------------------

def test_run_json_is_deterministic_and_machine_readable(project: Path, scripted) -> None:
    scripted([
        ScriptedTurn(text=_tool_call("fs.read", path="src/app.py")),
        ScriptedTurn(text="It returns 42."),
    ])
    result = _invoke("--json", "run", "What does app.py return?")
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["schema"] == 1
    assert payload["ok"] is True
    assert payload["final_text"] == "It returns 42."
    assert payload["tool_calls"] == 1
    assert payload["mode"] == "read-only"
    assert payload["stop_reason"] == "completed"
    assert payload["session_id"]


def test_run_json_emits_no_rich_chrome_on_stdout(project: Path, scripted) -> None:
    scripted([ScriptedTurn(text="plain answer")])
    result = _invoke("--json", "run", "hello")
    # Exactly one line, and it parses.
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    assert len(lines) == 1
    json.loads(lines[0])


def test_run_jsonl_streams_events_then_the_result(project: Path, scripted) -> None:
    scripted([
        ScriptedTurn(text=_tool_call("fs.glob", pattern="*.md")),
        ScriptedTurn(text="found it"),
    ])
    result = _invoke("--jsonl", "run", "list markdown")
    lines = [json.loads(line) for line in result.stdout.strip().splitlines() if line.strip()]
    kinds = [line.get("event") for line in lines if "event" in line]
    assert "tool_started" in kinds and "turn_finished" in kinds
    assert lines[-1]["ok"] is True


def test_run_json_default_mode_cannot_write(project: Path, scripted) -> None:
    scripted([
        ScriptedTurn(text=_tool_call("fs.write", path="new.txt", content="x")),
        ScriptedTurn(text="I could not write that."),
    ])
    result = _invoke("--json", "run", "create a file")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True  # the model recovered
    assert not (project / "new.txt").exists()


def test_run_json_blocks_on_approval_and_exits_nonzero(project: Path, scripted) -> None:
    """No TTY prompts in machine mode; the default policy fails rather than allows."""
    scripted([
        ScriptedTurn(text=_tool_call("fs.write", path="new.txt", content="x")),
        ScriptedTurn(text="blocked"),
    ])
    result = _invoke("--json", "--mode", "workspace", "run", "create a file")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert result.exit_code == cli_module.EXIT_APPROVAL_REQUIRED
    assert payload["approval_blocked"] is True
    assert not (project / "new.txt").exists()


def test_run_json_reject_policy_lets_the_model_continue(project: Path, scripted) -> None:
    scripted([
        ScriptedTurn(text=_tool_call("fs.write", path="new.txt", content="x")),
        ScriptedTurn(text="understood, not writing"),
    ])
    result = _invoke("--json", "--mode", "workspace", "--approval", "reject",
                     "run", "create a file")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert result.exit_code == 0 and payload["ok"] is True
    assert not (project / "new.txt").exists()


def test_yes_flag_approves_what_the_mode_already_allows(project: Path, scripted) -> None:
    scripted([
        ScriptedTurn(text=_tool_call("fs.write", path="new.txt", content="hi\n")),
        ScriptedTurn(text="created"),
    ])
    result = _invoke("--json", "--mode", "workspace", "-y", "run", "create a file")
    assert result.exit_code == 0
    assert (project / "new.txt").read_text() == "hi\n"


def test_yes_flag_cannot_escalate_the_mode(project: Path, scripted) -> None:
    """-y is an approver, not a permission: read-only still refuses the write."""
    scripted([
        ScriptedTurn(text=_tool_call("fs.write", path="new.txt", content="x")),
        ScriptedTurn(text="denied"),
    ])
    result = _invoke("--json", "-y", "run", "create a file")
    assert result.exit_code == 0
    assert not (project / "new.txt").exists()


def test_run_reports_a_provider_failure_as_an_error(project: Path, scripted, monkeypatch) -> None:
    monkeypatch.setattr("gemma_cyber.agent.runtime.RETRY_BASE_DELAY", 0.0)
    from gemma_cyber.agent.errors import ProviderError

    scripted([ScriptedTurn(raises=ProviderError("nope"))] * 5)
    result = _invoke("--json", "run", "hello")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert result.exit_code == cli_module.EXIT_ERROR
    assert payload["ok"] is False and payload["error_code"] == "provider_error"


def test_fail_on_findings_exit_code(project: Path, scripted) -> None:
    scripted([ScriptedTurn(text="I found a hardcoded credential: one finding.")])
    result = _invoke("--json", "run", "--fail-on-findings", "scan")
    assert result.exit_code == cli_module.EXIT_FINDINGS


# -- trusted-mode gate ------------------------------------------------------

def test_trusted_requires_accept_risk(project: Path, scripted) -> None:
    scripted([ScriptedTurn(text="x")])
    result = _invoke("--mode", "trusted", "run", "hello")
    assert result.exit_code == cli_module.EXIT_USAGE
    assert "i-accept-risk" in result.stdout + str(result.stderr)


def test_trusted_is_refused_with_machine_output(project: Path, scripted) -> None:
    scripted([ScriptedTurn(text="x")])
    result = _invoke("--mode", "trusted", "--i-accept-risk", "--json", "run", "hello")
    assert result.exit_code == cli_module.EXIT_USAGE


def test_trusted_requires_a_tty(project: Path, scripted, monkeypatch) -> None:
    scripted([ScriptedTurn(text="x")])
    monkeypatch.setattr(cli_module.Renderer, "is_tty", staticmethod(lambda: False))
    result = _invoke("--mode", "trusted", "--i-accept-risk", "run", "hello")
    assert result.exit_code == cli_module.EXIT_USAGE


def test_a_project_config_cannot_grant_trusted_mode(project: Path, scripted) -> None:
    (project / ".gemma4").mkdir()
    (project / ".gemma4" / "config.toml").write_text(
        '[permissions]\nmode = "trusted"\n', encoding="utf-8"
    )
    scripted([ScriptedTurn(text=_tool_call("shell.exec", argv=["echo", "pwned"])),
              ScriptedTurn(text="done")])
    result = _invoke("--json", "run", "run something")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["mode"] == "read-only"


# -- ask --------------------------------------------------------------------

def test_ask_uses_no_tools(project: Path, scripted) -> None:
    provider = scripted([ScriptedTurn(text=_tool_call("fs.read", path="src/app.py")),
                         ScriptedTurn(text="nope")])
    result = _invoke("ask", "read app.py")
    assert result.exit_code == 0
    # The request carried no tool schemas, and the system policy said so.
    system = provider.requests[0].messages[0].content
    assert "No tools are available" in system
    assert provider.requests[0].tools == []


# -- inspection commands ----------------------------------------------------

def test_tools_command_shows_what_is_hidden(project: Path) -> None:
    result = _invoke("tools")
    assert result.exit_code == 0
    plain = " ".join(result.stdout.split())
    assert "fs.read" in plain and "visible" in plain
    assert "hidden in read-only" in plain


def test_tools_command_in_agent_mode_shows_shell(project: Path) -> None:
    plain = " ".join(_invoke("--mode", "agent", "tools").stdout.split())
    assert "shell.exec" in plain
    assert "hidden in agent" not in plain


def test_auditor_profile_has_no_edit_or_shell(project: Path) -> None:
    plain = " ".join(_invoke("--profile", "auditor", "--mode", "agent", "tools").stdout.split())
    assert "fs.read" in plain
    assert "fs.edit" not in plain and "shell.exec" not in plain


def test_models_command_does_not_overstate_capability(project: Path) -> None:
    result = _invoke("models")
    assert result.exit_code == 0
    plain = " ".join(result.stdout.split())
    assert "xml-codec" in plain or "native" in plain
    assert "unreliably" in plain


def test_config_init_and_path(project: Path, tmp_path: Path) -> None:
    assert _invoke("config", "init").exit_code == 0
    config_file = tmp_path / "config" / "gemma4" / "config.toml"
    assert config_file.is_file()
    assert "api_key" not in config_file.read_text().replace("GEMMA4_API_KEY", "")
    # A second init without --force is refused.
    assert _invoke("config", "init").exit_code == cli_module.EXIT_USAGE
    assert _invoke("config", "init", "--force").exit_code == 0
    assert "sessions" in " ".join(_invoke("config", "path").stdout.split())


# -- doctor -----------------------------------------------------------------

def test_doctor_is_green_with_a_reachable_runtime(project: Path, monkeypatch) -> None:
    import httpx

    from gemma_cyber.agent.providers.ollama_chat import OllamaChatProvider

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "gemma3:4b"}]})

    monkeypatch.setattr(
        cli_module, "_make_provider",
        lambda *a, **k: OllamaChatProvider(
            model="gemma3:4b", client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    result = _invoke("doctor")
    assert result.exit_code == 0
    plain = " ".join(result.stdout.split())
    assert "workspace" in plain and "1 model(s) available" in plain
    assert "no-tools" in plain  # the honesty line


def test_doctor_reports_a_missing_model(project: Path, monkeypatch) -> None:
    import httpx

    from gemma_cyber.agent.providers.ollama_chat import OllamaChatProvider

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

    monkeypatch.setattr(
        cli_module, "_make_provider",
        lambda *a, **k: OllamaChatProvider(
            model="gemma3:4b", client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    result = _invoke("doctor")
    assert result.exit_code == cli_module.EXIT_ERROR
    assert "ollama pull" in " ".join(result.stdout.split())


def test_doctor_reports_an_unreachable_runtime(project: Path, monkeypatch) -> None:
    import httpx

    from gemma_cyber.agent.providers.ollama_chat import OllamaChatProvider

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(
        cli_module, "_make_provider",
        lambda *a, **k: OllamaChatProvider(
            model="gemma3:4b", client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    result = _invoke("doctor")
    assert result.exit_code == cli_module.EXIT_ERROR
    assert "ollama serve" in " ".join(result.stdout.split())


def test_doctor_never_prints_a_credential(project: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMMA4_API_KEY", "sk-super-secret-value")
    result = _invoke("doctor")
    assert "sk-super-secret" not in result.stdout


def test_doctor_warns_about_an_unsafe_project_config(project: Path) -> None:
    (project / ".gemma4").mkdir()
    (project / ".gemma4" / "config.toml").write_text(
        '[providers.default]\napi_key = "sk-in-a-repo"\n', encoding="utf-8"
    )
    result = _invoke("doctor")
    plain = " ".join(result.stdout.split())
    assert "WARN" in plain and "sk-in-a-repo" not in plain


# -- sessions ---------------------------------------------------------------

def test_session_lifecycle(project: Path, scripted) -> None:
    scripted([ScriptedTurn(text="hello")])
    run = _invoke("--json", "run", "hi")
    session_id = json.loads(run.stdout.strip().splitlines()[-1])["session_id"]

    listing = _invoke("session", "ls")
    assert session_id in " ".join(listing.stdout.split())

    show = _invoke("session", "show", session_id)
    assert "hello" in show.stdout

    assert _invoke("session", "rm", session_id).exit_code == 0
    assert _invoke("session", "show", session_id).exit_code == cli_module.EXIT_ERROR


def test_session_show_of_an_unknown_id_fails_cleanly(project: Path) -> None:
    assert _invoke("session", "show", "deadbeef").exit_code == cli_module.EXIT_ERROR


# -- workspace discovery ----------------------------------------------------

def test_cwd_override_selects_the_workspace(project: Path, tmp_path: Path, scripted) -> None:
    other = tmp_path / "other"
    (other / "sub").mkdir(parents=True)
    (other / "only_here.txt").write_text("x", encoding="utf-8")
    scripted([ScriptedTurn(text=_tool_call("fs.glob", pattern="*.txt")),
              ScriptedTurn(text="listed")])
    result = _invoke("--json", "--cwd", str(other), "run", "list files")
    assert result.exit_code == 0


def test_home_as_workspace_is_refused(project: Path, tmp_path: Path, monkeypatch) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    result = _invoke("--cwd", str(fake_home), "tools")
    assert result.exit_code == cli_module.EXIT_ERROR


def test_mode_flag_changes_the_ceiling(project: Path, scripted) -> None:
    scripted([ScriptedTurn(text="ok")])
    result = _invoke("--json", "--mode", "agent", "run", "hi")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["mode"] == PermissionMode.AGENT.value


# -- resume -----------------------------------------------------------------

def test_resume_picks_up_the_newest_session_for_this_workspace(
    project: Path, scripted, monkeypatch
) -> None:
    scripted([ScriptedTurn(text="first answer")])
    run = _invoke("--json", "run", "remember this")
    session_id = json.loads(run.stdout.strip().splitlines()[-1])["session_id"]

    # `resume` with no id resolves the newest session, then enters the REPL.
    entered: dict[str, str | None] = {}
    monkeypatch.setattr(
        cli_module, "_run_interactive",
        lambda resume=None: (entered.update(resume=resume), 0)[1],
    )
    result = _invoke("resume")
    assert result.exit_code == 0
    assert entered["resume"] == session_id


def test_resume_with_no_prior_session_fails_cleanly(project: Path) -> None:
    result = _invoke("resume")
    assert result.exit_code == cli_module.EXIT_ERROR
    assert "no previous session" in result.stdout + str(result.stderr)


def test_resume_reloads_the_conversation(project: Path, scripted) -> None:
    scripted([ScriptedTurn(text="first answer")])
    run = _invoke("--json", "run", "remember this")
    session_id = json.loads(run.stdout.strip().splitlines()[-1])["session_id"]

    provider = FakeProvider([ScriptedTurn(text="second")], name="fake", model="fake-model")
    app = cli_module.build_app(
        cli_module.GlobalOptions(), provider=provider, resume_session=session_id
    )
    assert app.state.session_id == session_id
    assert any("remember this" in m.content for m in app.state.messages)
    assert any("first answer" in m.content for m in app.state.messages)
