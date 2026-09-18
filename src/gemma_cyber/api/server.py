"""Uvicorn entry point for the API. Console script: ``gemma-cyber-serve``.

Reads host/port from ``GEMMA_CYBER_API_HOST`` / ``GEMMA_CYBER_API_PORT`` (defaults
127.0.0.1:8000 — bind to localhost by default; put a reverse proxy in front for
public exposure, see docs/deployment.md).

Logging is structured JSON at the configured ``GEMMA_CYBER_LOG_LEVEL`` (Phase 5),
so operators get correlatable, secret-free access logs out of the box.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("gemma_cyber.api")


def _port_from_env() -> int:
    raw = os.environ.get("GEMMA_CYBER_API_PORT", "8000")
    try:
        port = int(raw)
    except ValueError as exc:
        raise SystemExit(f"GEMMA_CYBER_API_PORT={raw!r} is not an integer") from exc
    if not (1 <= port <= 65535):
        raise SystemExit(f"GEMMA_CYBER_API_PORT={port} is out of range (1-65535)")
    return port


def run() -> None:
    import uvicorn

    from gemma_cyber.api.app import create_app
    from gemma_cyber.api.logging_setup import configure_api_logging
    from gemma_cyber.inference import load_settings

    settings = load_settings()
    configure_api_logging(settings.log_level)

    host = os.environ.get("GEMMA_CYBER_API_HOST", "127.0.0.1")
    port = _port_from_env()

    # Build the app before serving so configuration/auth errors fail fast (the
    # process exits non-zero) rather than surfacing on the first request.
    app = create_app(settings)
    logger.info(
        "starting api", extra={"route": f"{host}:{port}", "model": settings.model}
    )
    # Disable uvicorn's own log config so our JSON formatter stays authoritative.
    uvicorn.run(app, host=host, port=port, log_config=None)


if __name__ == "__main__":
    run()
