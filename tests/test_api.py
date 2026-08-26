"""API tests via FastAPI TestClient with an injected fake engine + registry.

Runs with no Ollama. Skipped entirely if the `api` extra (fastapi) is absent.
"""

from __future__ import annotations

from typing import cast

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from gemma_cyber.api.app import create_app  # noqa: E402
from gemma_cyber.clients.ollama_client import GenerationResult  # noqa: E402
from gemma_cyber.inference.config import Settings  # noqa: E402
from gemma_cyber.inference.engine import (  # noqa: E402
    HealthStatus,
    InferenceEngine,
    StreamChunk,
)
from gemma_cyber.inference.errors import ServiceUnavailableError  # noqa: E402
from gemma_cyber.inference.registry import ModelRecord, ModelRegistry  # noqa: E402


class FakeEngine:
    def __init__(self, *, ok=True, text="fake answer", raise_exc=None):
        self.model = "gemma3:4b"
        self._ok = ok
        self._text = text
        self._raise = raise_exc

    def health(self):
        return HealthStatus(ok=self._ok, service_reachable=self._ok,
                            model_present=self._ok, model=self.model,
                            host="http://fake", detail="" if self._ok else "down")

    def generate(self, prompt, **kw):
        if self._raise:
            raise self._raise
        return GenerationResult(text=self._text, model=self.model, prompt=prompt,
                                system=kw.get("system"), options={})

    def stream(self, prompt, **kw):
        for piece in ("fa", "ke"):
            yield StreamChunk(request_id="rid", text=piece)
        yield StreamChunk(request_id="rid", text="", done=True)


def _client(settings=None, engine=None, registry=None):
    # FakeEngine duck-types InferenceEngine (model + generate + stream + health).
    app = create_app(settings or Settings(),
                     engine=cast(InferenceEngine, engine or FakeEngine()),
                     registry=registry)
    return TestClient(app)


def test_health_liveness():
    r = _client().get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ready_ok():
    r = _client(engine=FakeEngine(ok=True)).get("/v1/ready")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_ready_not_ok_returns_503():
    r = _client(engine=FakeEngine(ok=False)).get("/v1/ready")
    assert r.status_code == 503 and r.json()["ok"] is False


def test_generate_json():
    r = _client(engine=FakeEngine(text="lateral movement")).post(
        "/v1/generate", json={"prompt": "what is lateral movement?"})
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "lateral movement"
    assert body["request_id"]
    assert "X-Request-ID" in r.headers


def test_security_headers_present():
    r = _client().get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in r.headers


def test_generate_validation_rejects_empty_prompt():
    r = _client().post("/v1/generate", json={"prompt": ""})
    assert r.status_code == 422


def test_generate_validation_rejects_oversized_prompt():
    r = _client().post("/v1/generate", json={"prompt": "x" * 30000})
    assert r.status_code == 422


def test_generate_service_unavailable_maps_503():
    eng = FakeEngine(raise_exc=ServiceUnavailableError("ollama down"))
    r = _client(engine=eng).post("/v1/generate", json={"prompt": "hi"})
    assert r.status_code == 503
    assert r.json()["error"] == "service_unavailable"


def test_stream_sse():
    r = _client().post("/v1/generate", json={"prompt": "hi", "stream": True})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "fa" in r.text and "done" in r.text


def test_auth_required_when_token_set():
    settings = Settings(api_token="secret")
    client = _client(settings=settings)
    # No token -> 401
    assert client.post("/v1/generate", json={"prompt": "hi"}).status_code == 401
    # Wrong token -> 401
    assert client.post("/v1/generate", json={"prompt": "hi"},
                       headers={"Authorization": "Bearer nope"}).status_code == 401
    # Correct token -> 200
    assert client.post("/v1/generate", json={"prompt": "hi"},
                       headers={"Authorization": "Bearer secret"}).status_code == 200


def test_rate_limit():
    settings = Settings(rate_limit_per_min=2)
    client = _client(settings=settings)
    assert client.post("/v1/generate", json={"prompt": "1"}).status_code == 200
    assert client.post("/v1/generate", json={"prompt": "2"}).status_code == 200
    assert client.post("/v1/generate", json={"prompt": "3"}).status_code == 429


def test_models_endpoint(tmp_path):
    reg = ModelRegistry(tmp_path / "r.json")
    reg.register(ModelRecord(version="gemma3-cyber:v0.2", dataset_version="sft_v0.2"))
    reg.mark_evaluated("gemma3-cyber:v0.2", passed=True)
    reg.promote("gemma3-cyber:v0.2", "candidate")
    reg.promote("gemma3-cyber:v0.2", "production")
    r = _client(registry=reg).get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["production"] == "gemma3-cyber:v0.2"
    assert body["models"][0]["passed_eval"] is True


def test_index_page_served():
    r = _client().get("/")
    assert r.status_code == 200
    assert "Gemma-Cyber" in r.text
    assert "text/html" in r.headers["content-type"]
