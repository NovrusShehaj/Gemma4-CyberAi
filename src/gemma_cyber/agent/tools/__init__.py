"""The toolbelt. Every tool runs through `ToolRuntime` -> `PermissionGuard`.

Tools must not import `gemma_cyber.agent.cli` or `gemma_cyber.agent.ui`: the
dependency arrow points inward (plan §11).

`builtin_tools` is the single registration point. A profile narrows the belt —
`auditor` never gets edit or shell even though the runtime supports them
(plan §49) — because a small model offered a tool will eventually reach for it.
"""

from __future__ import annotations

from gemma_cyber.agent.tools.base import Tool

__all__ = ["PROFILES", "builtin_tools"]

#: profile -> tool names, or None for "everything the mode allows".
#:
#: `auditor` is read-only by construction: it gets the search tools and the
#: defensive scanners, and never fs.edit or shell.exec. The runtime would happily
#: run them; the profile is what says it must not (plan §49).
PROFILES: dict[str, tuple[str, ...] | None] = {
    "general": None,
    "coder": None,
    "auditor": ("fs.read", "fs.glob", "fs.grep", "sec.secrets", "sec.deps", "sec.semgrep"),
}


def builtin_tools(profile: str = "general") -> list[Tool]:
    """Construct the toolbelt for ``profile``. Unknown profiles get `general`."""
    from gemma_cyber.agent.tools.fs import builtin_fs_tools
    from gemma_cyber.agent.tools.security import builtin_security_tools
    from gemma_cyber.agent.tools.shell import ShellExecTool

    tools: list[Tool] = [*builtin_fs_tools(), ShellExecTool(), *builtin_security_tools()]
    allowed = PROFILES.get(profile, None)
    if allowed is None:
        return tools
    return [t for t in tools if t.name in allowed]
