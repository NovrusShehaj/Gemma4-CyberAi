"""Structured JSON logging for the API service (Phase 5 observability).

One line per event, machine-parseable, with the operational fields the access-log
middleware already attaches via ``extra=`` (request id, route, status, latency).
Crucially it logs **no secrets and no content**: bearer tokens, prompts, and
responses are never passed to the logger in the first place, and this formatter
only emits an explicit allowlist of ``extra`` keys, so a stray field cannot leak.
"""

from __future__ import annotations

import json
import logging
import sys

# The only ``extra=`` keys we serialise. Anything not here is dropped, so a future
# caller cannot accidentally log a prompt/token by attaching it to a log record.
_ALLOWED_EXTRA = (
    "request_id", "method", "path", "route", "status", "latency_ms",
    "model", "attempt", "chars", "auth_result", "error", "sleep", "subject",
)

# Standard LogRecord attributes we never want in the JSON payload.
_RESERVED = set(vars(logging.makeLogRecord({})))


class JsonFormatter(logging.Formatter):
    """Render a log record as a single compact JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            if key in _ALLOWED_EXTRA and value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_api_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root handler and apply ``level``.

    Idempotent: replaces handlers so repeated calls (e.g. reload) don't duplicate
    output. Uvicorn's access/error loggers propagate to root so they, too, become
    structured. Unknown levels fall back to INFO rather than raising at startup.
    """
    resolved = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(resolved)
    # Let uvicorn's loggers flow through the root handler for uniform JSON output.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True
