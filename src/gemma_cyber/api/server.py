"""Uvicorn entry point for the API. Console script: ``gemma-cyber-serve``.

Reads host/port from ``GEMMA_CYBER_API_HOST`` / ``GEMMA_CYBER_API_PORT`` (defaults
127.0.0.1:8000 — bind to localhost by default; put a reverse proxy in front for
public exposure, see docs/deployment.md).
"""

from __future__ import annotations

import os


def run() -> None:
    import uvicorn

    from gemma_cyber.api.app import create_app

    host = os.environ.get("GEMMA_CYBER_API_HOST", "127.0.0.1")
    port = int(os.environ.get("GEMMA_CYBER_API_PORT", "8000"))
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run()
