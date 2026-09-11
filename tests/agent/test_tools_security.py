"""Defensive security tools: findings without leaking the credentials they find."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemma_cyber.agent.providers.fake import FakeProvider
from gemma_cyber.agent.tools import builtin_tools
from gemma_cyber.agent.tools.security import (
    DepsAuditArgs,
    DepsAuditTool,
    SecretScanArgs,
    SecretScanTool,
    SemgrepArgs,
    SemgrepTool,
    shannon_entropy,
)
from gemma_cyber.agent.types import PermissionMode, SideEffect

REAL_LOOKING_KEY = "AKIAIOSFODNN7EXAMPLE"
HIGH_ENTROPY = "Zx9qV3mTb8KpRw2LnQ7yUe4Hs6Dg1Fj0"


@pytest.fixture
def ctx(make_agent):
    return make_agent(FakeProvider([]), mode=PermissionMode.READ_ONLY).tool_context()


def _payload(result) -> dict:
    payload: dict = json.loads(result.content)
    return payload


# -- policy shape -----------------------------------------------------------

def test_security_tools_are_read_only_and_offline() -> None:
    for tool in builtin_tools("auditor"):
        assert tool.network is False, f"{tool.name} must not declare network access"
        assert tool.mutating is False, f"{tool.name} must not be mutating"
        assert tool.side_effect in (SideEffect.READ, SideEffect.SECURITY_SENSITIVE)


def test_auditor_profile_excludes_edit_and_shell() -> None:
    names = {t.name for t in builtin_tools("auditor")}
    assert "sec.secrets" in names
    assert not names & {"fs.edit", "fs.write", "shell.exec"}


def test_security_tools_are_available_in_read_only_mode(workspace) -> None:
    from gemma_cyber.agent.permissions import PermissionGuard

    guard = PermissionGuard(workspace, PermissionMode.READ_ONLY)
    for tool in builtin_tools("auditor"):
        assert guard.is_visible(tool)


# -- sec.secrets ------------------------------------------------------------

def test_secret_scan_finds_a_committed_key(ctx, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)  # force the builtin engine
    (ctx.workspace.root / "settings.py").write_text(
        f'AWS_ACCESS_KEY_ID = "{REAL_LOOKING_KEY}"\n', encoding="utf-8"
    )
    payload = _payload(SecretScanTool().run(SecretScanArgs(), ctx))
    assert payload["count"] >= 1
    assert any(f["path"] == "settings.py" for f in payload["findings"])


def test_secret_scan_never_echoes_the_credential(ctx, monkeypatch) -> None:
    """A scanner that prints secrets has made a second copy of the problem."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    (ctx.workspace.root / "settings.py").write_text(
        f'AWS_ACCESS_KEY_ID = "{REAL_LOOKING_KEY}"\n'
        f'SESSION_TOKEN = "{HIGH_ENTROPY}"\n',
        encoding="utf-8",
    )
    result = SecretScanTool().run(SecretScanArgs(), ctx)
    assert REAL_LOOKING_KEY not in result.content
    assert HIGH_ENTROPY not in result.content
    findings = _payload(result)["findings"]
    assert all("fingerprint" in f for f in findings)
    assert any("*" in f.get("masked", "") for f in findings)


def test_secret_scan_reports_a_stable_fingerprint(ctx, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    (ctx.workspace.root / "a.py").write_text(f'KEY = "{HIGH_ENTROPY}"\n', encoding="utf-8")
    first = _payload(SecretScanTool().run(SecretScanArgs(), ctx))["findings"]
    second = _payload(SecretScanTool().run(SecretScanArgs(), ctx))["findings"]
    assert [f["fingerprint"] for f in first] == [f["fingerprint"] for f in second]


def test_secret_scan_filters_test_fixtures_by_default(ctx, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    (ctx.workspace.root / "test_settings.py").write_text(
        f'KEY = "{HIGH_ENTROPY}"\n', encoding="utf-8"
    )
    default = _payload(SecretScanTool().run(SecretScanArgs(), ctx))
    assert default["count"] == 0
    widened = _payload(SecretScanTool().run(SecretScanArgs(include_low_signal=True), ctx))
    assert widened["count"] >= 1


def test_secret_scan_skips_placeholders(ctx, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    (ctx.workspace.root / "conf.py").write_text(
        'API_KEY = "your_api_key_goes_here_placeholder"\n', encoding="utf-8"
    )
    assert _payload(SecretScanTool().run(SecretScanArgs(), ctx))["count"] == 0


def test_secret_scan_excludes_the_git_directory(ctx, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    git_dir = ctx.workspace.root / ".git"
    git_dir.mkdir(exist_ok=True)
    (git_dir / "config").write_text(f'password = "{HIGH_ENTROPY}"\n', encoding="utf-8")
    payload = _payload(SecretScanTool().run(SecretScanArgs(), ctx))
    assert not any(".git" in f["path"] for f in payload["findings"])


def test_secret_scan_is_clean_on_an_innocent_repo(ctx, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    payload = _payload(SecretScanTool().run(SecretScanArgs(), ctx))
    assert payload["count"] == 0
    assert "no committed credentials" in payload["summary"]


def test_entropy_separates_keys_from_identifiers() -> None:
    assert shannon_entropy(HIGH_ENTROPY) > 3.6
    assert shannon_entropy("aaaaaaaaaaaaaaaaaaaaaaaa") < 1.0
    assert shannon_entropy("") == 0.0


def test_secret_scan_path_is_jailed(ctx) -> None:
    from gemma_cyber.agent.errors import WorkspaceEscape

    with pytest.raises(WorkspaceEscape):
        SecretScanTool().run(SecretScanArgs(path="../.."), ctx)


# -- sec.deps ---------------------------------------------------------------

def test_deps_audit_flags_hygiene_problems(ctx) -> None:
    (ctx.workspace.root / "requirements.txt").write_text(
        "--extra-index-url https://internal.example/simple\n"
        "requests\n"
        "flask==3.0.0\n"
        "mylib @ git+https://github.com/example/mylib@main\n"
        "# a comment\n",
        encoding="utf-8",
    )
    payload = _payload(DepsAuditTool().run(DepsAuditArgs(), ctx))
    rules = {f["rule"] for f in payload["findings"]}
    assert "non-default-index" in rules
    assert "unpinned-dependency" in rules
    assert "url-dependency" in rules
    assert "missing-hash" in rules


def test_deps_audit_states_that_it_does_not_look_up_cves(ctx) -> None:
    """Honesty: the tool must not imply it checked for vulnerabilities."""
    payload = _payload(DepsAuditTool().run(DepsAuditArgs(), ctx))
    assert "offline by design" in payload["note"]
    assert "pyproject.toml" in payload["manifests"]


def test_deps_audit_with_no_manifests(ctx) -> None:
    (ctx.workspace.root / "pyproject.toml").unlink()
    payload = _payload(DepsAuditTool().run(DepsAuditArgs(), ctx))
    assert payload["count"] == 0 and "no dependency manifests" in payload["summary"]


# -- sec.semgrep ------------------------------------------------------------

def test_semgrep_is_a_no_op_when_not_installed(ctx, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    payload = _payload(SemgrepTool().run(SemgrepArgs(), ctx))
    assert payload["count"] == 0
    assert "not installed" in payload["summary"]
    assert "will not download" in payload["note"]


def test_semgrep_refuses_to_run_without_a_committed_ruleset(ctx, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/semgrep")
    payload = _payload(SemgrepTool().run(SemgrepArgs(), ctx))
    assert payload["count"] == 0
    assert "no committed Semgrep ruleset" in payload["summary"]
    assert "Remote registry" in payload["note"]


def test_semgrep_uses_the_local_ruleset_and_never_a_registry_config(
    ctx, monkeypatch, tmp_path: Path
) -> None:
    (ctx.workspace.root / ".semgrep.yml").write_text("rules: []\n", encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/semgrep")
    captured: dict[str, list[str]] = {}

    class Completed:
        stdout = json.dumps({"results": [
            {"path": "src/app.py", "start": {"line": 3}, "check_id": "rules.eval",
             "extra": {"severity": "ERROR", "message": "avoid eval"}},
        ]})
        stderr = ""
        returncode = 0

    def fake_run(argv, ctx_arg):
        captured["argv"] = argv
        return Completed()

    monkeypatch.setattr("gemma_cyber.agent.tools.security._run_helper", fake_run)
    payload = _payload(SemgrepTool().run(SemgrepArgs(), ctx))

    assert payload["count"] == 1
    assert payload["findings"][0]["rule"] == "rules.eval"
    argv = captured["argv"]
    assert "--config" in argv
    config_value = argv[argv.index("--config") + 1]
    assert config_value.endswith(".semgrep.yml")
    assert not config_value.startswith("p/"), "registry rulesets must never be fetched"
    assert "--metrics" in argv and argv[argv.index("--metrics") + 1] == "off"


# -- product policy ---------------------------------------------------------

def test_no_offensive_tool_exists(ctx) -> None:
    names = {t.name for t in builtin_tools()}
    for forbidden in ("sec.nmap", "sec.scan", "sec.exploit", "sec.detonate", "sec.yara"):
        assert forbidden not in names


# -- tool source port -------------------------------------------------------

def test_a_source_cannot_smuggle_in_a_tool_without_a_valid_side_effect() -> None:
    """Any future source (entry points, MCP) is validated the same way."""
    from pydantic import BaseModel

    from gemma_cyber.agent.tools.base import (
        BuiltinToolSource,
        Tool,
        ToolContext,
        ToolRegistry,
    )
    from gemma_cyber.agent.types import ToolResult

    class Args(BaseModel):
        pass

    class Rogue(Tool):
        name = "plugin.rogue"
        description = "d"
        input_model = Args
        side_effect = "unrestricted"  # type: ignore[assignment]

        def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:  # pragma: no cover
            return ToolResult(call_id="", name=self.name, ok=True)

    registry = ToolRegistry()
    with pytest.raises(ValueError, match="unknown side effect"):
        registry.add_source(BuiltinToolSource([Rogue()]))
    assert "plugin.rogue" not in registry


def test_add_source_records_the_origin() -> None:
    from gemma_cyber.agent.tools.base import BuiltinToolSource, ToolRegistry
    from gemma_cyber.agent.tools.fs import ReadFileTool

    registry = ToolRegistry()
    registry.add_source(BuiltinToolSource([ReadFileTool()]))
    assert registry.origin("fs.read") == "builtin"
    assert registry.origin("nope") == "unknown"
