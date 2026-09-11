"""`gemma4` — the opt-in, local, fail-closed terminal agent.

This package is the ONLY place in `gemma_cyber` where tool execution, shell
execution, and workspace mutation exist. Two hard boundaries hold it in place:

1. `gemma_cyber.api` must never import this package (enforced by
   `tests/agent/test_import_boundary.py`). The hosted product stays no-tools.
2. Nothing here is installed by the base package — `pip install gemma-cyber`
   yields a `gemma4` script that refuses to run until `[agent]` is present.

Importing this module is cheap and dependency-free on purpose: `cli.py` pulls in
typer/rich/prompt_toolkit, and `tests`/`doctor` need to reason about the package
without those being installed.

See `GEMMA4_TERMINAL_AGENT_IMPLEMENTATION_PLAN.md` for the full contract.
"""

from __future__ import annotations

from gemma_cyber import __version__ as _pkg_version

__all__ = ["AGENT_VERSION"]

#: The agent ships with the package; there is no separate version line (plan §25).
AGENT_VERSION = _pkg_version
