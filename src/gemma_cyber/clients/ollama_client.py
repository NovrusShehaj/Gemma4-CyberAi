"""Thin, deterministic client for the local Ollama HTTP API.

Design goals (PROJECT_PLAN.md §12, §17):
  * Deterministic by default (temperature=0, fixed seed) so evaluations are reproducible.
  * Records the exact model tag used, for provenance in scorecards.
  * No streaming: we want a single complete response per call.

This client is for INFERENCE ONLY. Training never goes through Ollama.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"


class OllamaError(RuntimeError):
    """Raised when the Ollama service is unreachable or returns an error."""


@dataclass
class GenerationResult:
    """A single model response plus the settings that produced it."""

    text: str
    model: str
    prompt: str
    system: str | None
    options: dict[str, Any] = field(default_factory=dict)


class OllamaClient:
    """Minimal wrapper over the Ollama `/api/generate` and `/api/tags` endpoints."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    # -- service / model availability ---------------------------------------

    def is_available(self) -> bool:
        """Return True if the Ollama service responds to /api/tags."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """Return the names of locally available models."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network
            raise OllamaError(f"Could not reach Ollama at {self.host}: {exc}") from exc
        return [m["name"] for m in resp.json().get("models", [])]

    def has_model(self, model: str | None = None) -> bool:
        """Return True if `model` (default: self.model) is pulled locally."""
        target = model or self.model
        return target in self.list_models()

    # -- generation ---------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        seed: int = 0,
        num_predict: int | None = None,
        think: bool | None = None,
    ) -> GenerationResult:
        """Generate a single, non-streamed completion.

        Deterministic defaults (temperature=0, seed=0) make repeated runs
        comparable. `num_predict` caps output tokens (None = model default).

        `think` controls Ollama's reasoning mode for thinking models (e.g. the
        Gemma 4 family). Left as None the field is not sent (preserving behavior
        for non-thinking models like gemma3:4b). Set `think=False` to force a
        direct answer: thinking models otherwise spend the token budget on hidden
        reasoning and return an EMPTY `response`. The LLM judge relies on this.
        """
        options: dict[str, Any] = {"temperature": temperature, "seed": seed}
        if num_predict is not None:
            options["num_predict"] = num_predict

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system is not None:
            payload["system"] = system
        if think is not None:
            payload["think"] = think
            options["think"] = think  # record for provenance

        try:
            resp = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(
                f"Generation failed against {self.host} (model={self.model}): {exc}"
            ) from exc

        data = resp.json()
        return GenerationResult(
            text=data.get("response", ""),
            model=self.model,
            prompt=prompt,
            system=system,
            options=options,
        )

    def stream_generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        seed: int = 0,
        num_predict: int | None = None,
        think: bool | None = None,
    ) -> Iterator[str]:
        """Yield response text incrementally as Ollama produces it.

        Streaming is for interactive surfaces (CLI/web), NOT for evaluation —
        evaluation stays on the deterministic, non-streamed :meth:`generate` so a
        scorecard reflects one complete response. Same deterministic defaults
        apply. Each yielded string is the next token chunk; the generator ends
        when Ollama sends ``done: true``.
        """
        options: dict[str, Any] = {"temperature": temperature, "seed": seed}
        if num_predict is not None:
            options["num_predict"] = num_predict

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": options,
        }
        if system is not None:
            payload["system"] = system
        if think is not None:
            payload["think"] = think

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("response", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break
        except requests.RequestException as exc:
            raise OllamaError(
                f"Streaming failed against {self.host} (model={self.model}): {exc}"
            ) from exc
