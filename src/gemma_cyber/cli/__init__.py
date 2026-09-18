"""gemma-cyber command-line interface (Phase 6).

A thin, scriptable CLI over the shared :mod:`gemma_cyber.inference` layer — it
adds no model logic of its own, so the CLI, the API, and the evaluation harness
all behave identically. Entry point: ``gemma_cyber.cli.main:main``.

Note: we intentionally do NOT re-export the ``main`` function here — doing so
would shadow the ``gemma_cyber.cli.main`` submodule of the same name. Import it
as ``from gemma_cyber.cli.main import main``.
"""

from __future__ import annotations
