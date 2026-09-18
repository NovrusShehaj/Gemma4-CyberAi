"""InferenceEngine — the one object every surface generates through.

Wraps a low-level model client (Ollama today) and adds the cross-cutting concerns
each surface would otherwise re-implement: model-version resolution via the
registry, bounded retries on transient service errors, a health check, streaming,
structured errors, and per-request ids for logs.

It deliberately satisfies the evaluation harness's ``SupportsGenerate`` protocol
(a ``model`` attribute + a ``generate(prompt, system, temperature, seed,
num_predict)`` method returning an object with ``.text``), so the benchmark
harness can run against the engine unchanged — the same code path the CLI and API
use. That shared path is the whole point of Phase 5.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from gemma_cyber.clients.ollama_client import GenerationResult, OllamaClient, OllamaError
from gemma_cyber.inference.config import Settings, load_settings
from gemma_cyber.inference.errors import (
    InferenceTimeoutError,
    ModelUnavailableError,
    ServiceUnavailableError,
)
from gemma_cyber.inference.registry import ModelRegistry

logger = logging.getLogger("gemma_cyber.inference")

# Sentinel distinguishing "system arg not passed -> use the settings default"
# from an explicit `system=None` (send no system prompt at all).
_UNSET: Any = object()


@dataclass
class HealthStatus:
    """Result of :meth:`InferenceEngine.health`."""

    ok: bool
    service_reachable: bool
    model_present: bool
    model: str
    host: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "service_reachable": self.service_reachable,
            "model_present": self.model_present,
            "model": self.model,
            "host": self.host,
            "detail": self.detail,
        }


@dataclass
class StreamChunk:
    """One streamed piece of a response, carrying the request id for correlation."""

    request_id: str
    text: str
    done: bool = False


def _is_timeout(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "timeout" in msg or "timed out" in msg


class InferenceEngine:
    """Provider-agnostic front door to the model runtime.

    Construct with :meth:`from_settings` in normal use; inject a fake ``client``
    and ``registry`` in tests. The engine never imports FastAPI/CLI code — the
    dependency arrow points inward.
    """

    def __init__(
        self,
        client: OllamaClient,
        settings: Settings,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._client = client
        self.settings = settings
        self._registry = registry
        # Plain settable attribute (satisfies SupportsGenerate.model) mirroring the
        # client's fixed model tag.
        self.model: str = client.model

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, *, model: str | None = None
    ) -> InferenceEngine:
        """Build an engine from :class:`Settings`, resolving ``model`` through the
        registry if one exists (so ``model="production"`` serves the promoted tag).
        """
        settings = settings or load_settings()
        registry: ModelRegistry | None = None
        if settings.registry_path and settings.registry_path.exists():
            registry = ModelRegistry(settings.registry_path)

        requested = model or settings.model
        resolved = requested
        if registry is not None:
            try:
                resolved = registry.resolve(requested)
            except Exception:  # unknown ref -> treat as a raw tag
                resolved = requested

        client = OllamaClient(
            model=resolved, host=settings.ollama_host, timeout=settings.timeout
        )
        return cls(client=client, settings=settings, registry=registry)

    @property
    def registry(self) -> ModelRegistry | None:
        return self._registry

    # -- health -------------------------------------------------------------

    def health(self) -> HealthStatus:
        """Check the runtime is reachable and the target model is present.

        Never raises: a health check that throws is useless to a readiness probe.
        """
        host = self._client.host
        model = self._client.model
        try:
            reachable = self._client.is_available()
        except Exception as exc:  # pragma: no cover - defensive
            return HealthStatus(
                ok=False, service_reachable=False, model_present=False,
                model=model, host=host, detail=f"availability check errored: {exc}",
            )
        if not reachable:
            return HealthStatus(
                ok=False, service_reachable=False, model_present=False,
                model=model, host=host, detail=f"Ollama unreachable at {host}",
            )
        try:
            present = self._client.has_model(model)
        except OllamaError as exc:
            return HealthStatus(
                ok=False, service_reachable=True, model_present=False,
                model=model, host=host, detail=str(exc),
            )
        return HealthStatus(
            ok=present,
            service_reachable=True,
            model_present=present,
            model=model,
            host=host,
            detail="" if present else f"model {model!r} not pulled locally",
        )

    # -- generation ---------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: str | None = _UNSET,
        temperature: float | None = None,
        seed: int | None = None,
        num_predict: int | None = None,
        *,
        request_id: str | None = None,
    ) -> GenerationResult:
        """Generate one complete response, with bounded retries on transient errors.

        Defaults (system prompt, temperature, seed, num_predict) come from
        ``settings`` when not passed, so a bare ``generate(prompt)`` is
        deterministic and safety-prompted like the eval baseline. ``system`` uses
        a sentinel so an explicit ``system=None`` (no system prompt) is
        distinguishable from "unset -> use the default".
        """
        rid = request_id or uuid.uuid4().hex[:12]
        sys_prompt = self.settings.system_prompt if system is _UNSET else system
        temp = self.settings.temperature if temperature is None else temperature
        sd = self.settings.seed if seed is None else seed
        npredict = self.settings.num_predict if num_predict is None else num_predict

        attempts = self.settings.max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                logger.info(
                    "generate", extra={"request_id": rid, "model": self.model,
                                       "attempt": attempt, "chars": len(prompt)}
                )
                return self._client.generate(
                    prompt, system=sys_prompt, temperature=temp,
                    seed=sd, num_predict=npredict,
                )
            except OllamaError as exc:
                last_exc = exc
                if _is_timeout(exc):
                    # Timeouts are retried too, but reported distinctly if final.
                    if attempt >= attempts:
                        raise InferenceTimeoutError(
                            f"[{rid}] generation timed out after {attempts} attempt(s): {exc}"
                        ) from exc
                elif attempt >= attempts:
                    raise ServiceUnavailableError(
                        f"[{rid}] inference failed after {attempts} attempt(s): {exc}"
                    ) from exc
                sleep = self.settings.retry_backoff * (2 ** (attempt - 1))
                logger.warning(
                    "generate retry", extra={"request_id": rid, "attempt": attempt,
                                             "sleep": sleep, "error": str(exc)}
                )
                time.sleep(sleep)
        # Unreachable, but keeps type-checkers happy.
        raise ServiceUnavailableError(f"[{rid}] inference failed: {last_exc}")

    def stream(
        self,
        prompt: str,
        system: str | None = _UNSET,
        temperature: float | None = None,
        seed: int | None = None,
        num_predict: int | None = None,
        *,
        request_id: str | None = None,
    ) -> Iterator[StreamChunk]:
        """Yield :class:`StreamChunk`s for interactive use.

        Streaming is not retried mid-response (a partial stream cannot be safely
        replayed); a failure before the first chunk surfaces as a structured
        error, matching :meth:`generate`.
        """
        rid = request_id or uuid.uuid4().hex[:12]
        sys_prompt = self.settings.system_prompt if system is _UNSET else system
        temp = self.settings.temperature if temperature is None else temperature
        sd = self.settings.seed if seed is None else seed
        npredict = self.settings.num_predict if num_predict is None else num_predict
        try:
            for piece in self._client.stream_generate(
                prompt, system=sys_prompt, temperature=temp,
                seed=sd, num_predict=npredict,
            ):
                yield StreamChunk(request_id=rid, text=piece)
        except OllamaError as exc:
            if _is_timeout(exc):
                raise InferenceTimeoutError(f"[{rid}] stream timed out: {exc}") from exc
            raise ServiceUnavailableError(f"[{rid}] stream failed: {exc}") from exc
        yield StreamChunk(request_id=rid, text="", done=True)

    def require_ready(self) -> None:
        """Raise a structured error if the engine cannot serve. Used by the CLI/API
        to fail fast with a precise message instead of a mid-request stack trace."""
        status = self.health()
        if not status.service_reachable:
            raise ServiceUnavailableError(status.detail)
        if not status.model_present:
            raise ModelUnavailableError(status.detail)
