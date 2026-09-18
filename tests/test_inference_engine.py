"""InferenceEngine: retries, health, streaming, defaults, SupportsGenerate.

The engine is exercised with a fake OllamaClient subclass (no network), so these
tests run in CI with no Ollama. The engine is the shared path used by the CLI,
API, and the evaluation harness.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest

from gemma_cyber.clients.ollama_client import GenerationResult, OllamaClient, OllamaError
from gemma_cyber.evaluation.harness import SupportsGenerate
from gemma_cyber.inference.config import Settings
from gemma_cyber.inference.engine import InferenceEngine
from gemma_cyber.inference.errors import (
    InferenceTimeoutError,
    ModelUnavailableError,
    ServiceUnavailableError,
)


class FakeClient(OllamaClient):
    """A scriptable OllamaClient that never touches the network."""

    def __init__(
        self,
        *,
        available: bool = True,
        models: list[str] | None = None,
        fail_times: int = 0,
        error: Exception | None = None,
    ) -> None:
        super().__init__(model="gemma3:4b", host="http://fake", timeout=1.0)
        self._available = available
        self._models = models if models is not None else ["gemma3:4b"]
        self._fail_times = fail_times
        self._error = error or OllamaError("transient")
        self.calls = 0
        self.last_kwargs: dict = {}

    def is_available(self) -> bool:
        return self._available

    def list_models(self) -> list[str]:
        return list(self._models)

    def generate(self, prompt, system=None, temperature=0.0, seed=0,
                 num_predict=None, think=None) -> GenerationResult:
        self.calls += 1
        self.last_kwargs = {
            "system": system, "temperature": temperature, "seed": seed,
            "num_predict": num_predict,
        }
        if self.calls <= self._fail_times:
            raise self._error
        return GenerationResult(
            text=f"answer to: {prompt}", model=self.model, prompt=prompt,
            system=system, options={"temperature": temperature, "seed": seed},
        )

    def stream_generate(self, prompt, system=None, temperature=0.0, seed=0,
                        num_predict=None, think=None) -> Iterator[str]:
        if self._error and self._fail_times > 0:
            raise self._error
        yield "hel"
        yield "lo"


def _settings(**kw) -> Settings:
    base = Settings(max_retries=2, retry_backoff=0.0, timeout=1.0)
    return dataclasses.replace(base, **kw)


def test_engine_satisfies_supports_generate():
    eng = InferenceEngine(FakeClient(), _settings())
    # Structural check: the harness only needs .model + .generate.
    probe: SupportsGenerate = eng
    assert probe.model == "gemma3:4b"


def test_generate_uses_settings_defaults():
    s = _settings(temperature=0.0, seed=0, num_predict=256,
                  system_prompt="SYS")
    client = FakeClient()
    eng = InferenceEngine(client, s)
    res = eng.generate("hi")
    assert res.text == "answer to: hi"
    assert client.last_kwargs["system"] == "SYS"
    assert client.last_kwargs["num_predict"] == 256


def test_explicit_system_none_sends_no_prompt():
    client = FakeClient()
    eng = InferenceEngine(client, _settings(system_prompt="SYS"))
    eng.generate("hi", system=None)
    assert client.last_kwargs["system"] is None


def test_retry_then_success():
    client = FakeClient(fail_times=2)  # fails twice, succeeds on 3rd
    eng = InferenceEngine(client, _settings(max_retries=2))
    res = eng.generate("hi")
    assert res.text == "answer to: hi"
    assert client.calls == 3


def test_retry_exhausted_raises_service_error():
    client = FakeClient(fail_times=99)
    eng = InferenceEngine(client, _settings(max_retries=1))
    with pytest.raises(ServiceUnavailableError):
        eng.generate("hi")
    assert client.calls == 2  # 1 + 1 retry


def test_timeout_maps_to_timeout_error():
    client = FakeClient(fail_times=99, error=OllamaError("request timed out"))
    eng = InferenceEngine(client, _settings(max_retries=0))
    with pytest.raises(InferenceTimeoutError):
        eng.generate("hi")


def test_health_ok():
    eng = InferenceEngine(FakeClient(models=["gemma3:4b"]), _settings())
    h = eng.health()
    assert h.ok and h.service_reachable and h.model_present


def test_health_service_down():
    eng = InferenceEngine(FakeClient(available=False), _settings())
    h = eng.health()
    assert not h.ok and not h.service_reachable


def test_health_model_missing():
    eng = InferenceEngine(FakeClient(models=["other:1b"]), _settings())
    h = eng.health()
    assert h.service_reachable and not h.model_present and not h.ok


def test_require_ready_raises_when_model_missing():
    eng = InferenceEngine(FakeClient(models=[]), _settings())
    with pytest.raises(ModelUnavailableError):
        eng.require_ready()


def test_stream_yields_chunks_then_done():
    eng = InferenceEngine(FakeClient(), _settings())
    chunks = list(eng.stream("hi"))
    assert "".join(c.text for c in chunks) == "hello"
    assert chunks[-1].done is True
    assert all(c.request_id == chunks[0].request_id for c in chunks)
