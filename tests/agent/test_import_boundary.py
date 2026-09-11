"""The hosted API must never import the agent package (plan §7.1, §35).

Two checks, because each catches what the other misses: a static AST scan finds
an import that a lazy code path would hide, and a real import of the FastAPI app
catches one that arrives through a transitive module.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import gemma_cyber

PACKAGE_ROOT = Path(gemma_cyber.__file__).parent
FORBIDDEN_PREFIX = "gemma_cyber.agent"

#: Surfaces that are contractually no-tools. `cli` is the operator Q&A CLI; the
#: `gemma4` entry point lives in `gemma_cyber.agent.cli`, which is separate.
NO_TOOLS_PACKAGES = ("api", "inference", "evaluation", "clients", "data", "knowledge")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


@pytest.mark.parametrize("package", NO_TOOLS_PACKAGES)
def test_no_tools_packages_do_not_import_the_agent(package: str) -> None:
    directory = PACKAGE_ROOT / package
    offenders: list[str] = []
    for source in directory.rglob("*.py"):
        for module in _imported_modules(source):
            if module == FORBIDDEN_PREFIX or module.startswith(FORBIDDEN_PREFIX + "."):
                offenders.append(f"{source.relative_to(PACKAGE_ROOT)} -> {module}")
    assert not offenders, (
        "the hosted/no-tools surfaces must not import the agent package: " + ", ".join(offenders)
    )


def _leaked_agent_modules(import_statement: str) -> list[str]:
    """Import something in a clean interpreter and report agent modules pulled in.

    A subprocess rather than `del sys.modules[...]` in-process: deleting modules
    here would leave the rest of the suite holding stale class objects (an
    `except AgentError` that no longer matches a freshly imported subclass), and
    a fresh process is the honest question anyway — "does `import X` drag the
    agent in?"
    """
    script = textwrap.dedent(f"""
        import json, sys
        {import_statement}
        print(json.dumps(sorted(
            n for n in sys.modules if n.startswith({FORBIDDEN_PREFIX!r})
        )))
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True,
    )
    import json

    leaked: list[str] = json.loads(completed.stdout.strip().splitlines()[-1])
    return leaked


def test_importing_the_fastapi_app_does_not_pull_in_the_agent() -> None:
    leaked = _leaked_agent_modules("from gemma_cyber.api.app import create_app")
    assert not leaked, f"importing the API pulled in agent modules: {leaked}"


def test_the_operator_cli_does_not_import_the_agent() -> None:
    leaked = _leaked_agent_modules("from gemma_cyber.cli import main")
    assert not leaked, f"importing the operator CLI pulled in agent modules: {leaked}"


def test_the_guard_catches_a_real_leak() -> None:
    """Meta-check: the subprocess probe must actually detect an agent import."""
    leaked = _leaked_agent_modules("import gemma_cyber.agent.permissions")
    assert FORBIDDEN_PREFIX + ".permissions" in leaked


def test_the_agent_does_not_import_the_hosted_api() -> None:
    """The reverse arrow: the agent must not depend on FastAPI either."""
    offenders: list[str] = []
    for source in (PACKAGE_ROOT / "agent").rglob("*.py"):
        for module in _imported_modules(source):
            if module.startswith(("gemma_cyber.api", "fastapi", "uvicorn")):
                offenders.append(f"{source.relative_to(PACKAGE_ROOT)} -> {module}")
    assert not offenders, "the agent must not depend on the hosted API: " + ", ".join(offenders)


def test_tools_do_not_import_the_ui_or_cli() -> None:
    """Dependency direction inside the agent (plan §6.3)."""
    offenders: list[str] = []
    for subpackage in ("tools", "providers"):
        for source in (PACKAGE_ROOT / "agent" / subpackage).rglob("*.py"):
            for module in _imported_modules(source):
                if module.startswith(("gemma_cyber.agent.ui", "gemma_cyber.agent.cli")):
                    offenders.append(f"{source.relative_to(PACKAGE_ROOT)} -> {module}")
    assert not offenders, "tools/providers must not import the UI or CLI: " + ", ".join(offenders)


def test_no_offensive_tooling_is_registered() -> None:
    """Product policy: no live-target, exploit, or detonation tools (plan §20)."""
    from gemma_cyber.agent.tools import builtin_tools

    names = {tool.name for tool in builtin_tools()}
    forbidden = {"nmap", "net.scan", "exploit", "yara.detonate", "msf", "sec.scan_target"}
    assert not (names & forbidden)
    for tool in builtin_tools():
        assert not tool.network, f"{tool.name} declares network access; none should in v1"
