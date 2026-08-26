"""Operational smoke tests run in-process (no live server, no credentials).

Drives scripts/smoke_test.run_smoke against the app via FastAPI TestClient in two
modes — open (dev) and static-token (auth enforced) — asserting every required
check passes. This is the CI-runnable half of the operational validation; the same
run_smoke also drives a live deployment via scripts/smoke_test.py.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from smoke_test import render, run_smoke  # noqa: E402

from gemma_cyber.api.app import create_app  # noqa: E402
from gemma_cyber.clients.ollama_client import GenerationResult  # noqa: E402
from gemma_cyber.inference.config import Settings  # noqa: E402
from gemma_cyber.inference.engine import (  # noqa: E402
    HealthStatus,
    InferenceEngine,
    StreamChunk,
)
from gemma_cyber.inference.errors import ServiceUnavailableError  # noqa: E402


class FakeEngine:
    model = "gemma3:4b"

    def __init__(self, *, ok=True, raise_exc=None):
        self._ok = ok
        self._raise = raise_exc

    def health(self):
        return HealthStatus(ok=self._ok, service_reachable=self._ok,
                            model_present=self._ok, model=self.model, host="http://fake")

    def generate(self, prompt, **kw):
        if self._raise:
            raise self._raise
        return GenerationResult(text="ok", model=self.model, prompt=prompt,
                                system=kw.get("system"), options={})

    def stream(self, prompt, **kw):
        yield StreamChunk(request_id="rid", text="ok")
        yield StreamChunk(request_id="rid", text="", done=True)


def _required_failures(results):
    return [r for r in results if r.required and not r.passed]


def test_smoke_open_mode():
    app = create_app(Settings(), engine=cast(InferenceEngine, FakeEngine()))
    results = run_smoke(TestClient(app))
    report, ok = render(results)
    assert ok, "required smoke checks failed:\n" + report


def test_smoke_auth_enforced_mode():
    app = create_app(Settings(api_token="secret"),
                     engine=cast(InferenceEngine, FakeEngine()))
    client = TestClient(app)
    results = run_smoke(client, token="secret", expect_auth=True)
    report, ok = render(results)
    assert ok, "required smoke checks failed in auth mode:\n" + report
    # The 401 enforcement check must be present and passing.
    names = {r.name: r.passed for r in results}
    assert names["unauthenticated generate -> 401"] is True


def test_smoke_reports_readiness_warning_when_model_down():
    app = create_app(Settings(), engine=cast(InferenceEngine, FakeEngine(ok=False)))
    results = run_smoke(TestClient(app))
    # Model-down is a non-required WARN, not a hard failure of the smoke run.
    ready200 = next(r for r in results if r.name == "readiness == 200 (model available)")
    assert ready200.passed is False and ready200.required is False
    assert not _required_failures(results)


def test_smoke_detects_broken_service():
    # If generate is broken, the happy-path required check fails the smoke run.
    app = create_app(Settings(),
                     engine=cast(InferenceEngine, FakeEngine(raise_exc=ServiceUnavailableError("down"))))
    results = run_smoke(TestClient(app))
    assert _required_failures(results), "smoke should fail when generate is down"


def test_render_is_stable():
    app = create_app(Settings(), engine=cast(InferenceEngine, FakeEngine()))
    results = run_smoke(TestClient(app))
    report, _ = render(results)
    assert "[PASS] liveness /health == 200" in report
    assert isinstance(time.time(), float)  # sanity
