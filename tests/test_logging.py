"""Structured JSON logging emits an allowlist of fields only — never content."""

from __future__ import annotations

import json
import logging

from gemma_cyber.api.logging_setup import JsonFormatter


def _fmt(**extra):
    rec = logging.LogRecord(
        name="gemma_cyber.api", level=logging.INFO, pathname=__file__, lineno=1,
        msg="request", args=(), exc_info=None,
    )
    for k, v in extra.items():
        setattr(rec, k, v)
    return json.loads(JsonFormatter().format(rec))


def test_allowed_fields_are_emitted():
    out = _fmt(request_id="abc", status=200, latency_ms=12.3, model="gemma3:4b")
    assert out["request_id"] == "abc"
    assert out["status"] == 200
    assert out["model"] == "gemma3:4b"
    assert out["msg"] == "request"
    assert out["level"] == "INFO"


def test_unlisted_fields_are_dropped():
    # A stray prompt/token attached to a record must NOT appear in the JSON line.
    out = _fmt(prompt="my secret prompt", authorization="Bearer xyz", token="abc")
    assert "prompt" not in out
    assert "authorization" not in out
    assert "token" not in out


def test_valid_json_single_line():
    formatted = JsonFormatter().format(
        logging.LogRecord("l", logging.INFO, __file__, 1, "m", (), None)
    )
    assert "\n" not in formatted
    json.loads(formatted)  # parses
