"""Shared inference layer — the single seam every surface talks to a model through.

Phase 5 keystone (see PROJECT_PLAN.md / TODO). The CLI, the web/API service, the
evaluation harness, and future clients must NOT each re-implement model logic
(host, retries, timeouts, model-version resolution, health). They all go through
:class:`~gemma_cyber.inference.engine.InferenceEngine`, which satisfies the
harness's ``SupportsGenerate`` protocol so a benchmark run and a chat request use
the exact same code path.

Public surface::

    from gemma_cyber.inference import InferenceEngine, Settings, ModelRegistry

Everything here is provider-agnostic at the boundary: the engine wraps a client
(today: Ollama) and can be pointed at a different runtime later without changing
the CLI/API/eval callers.
"""

from __future__ import annotations

from gemma_cyber.inference.config import Settings, load_settings
from gemma_cyber.inference.engine import (
    HealthStatus,
    InferenceEngine,
    StreamChunk,
)
from gemma_cyber.inference.errors import (
    InferenceError,
    ModelUnavailableError,
    ServiceUnavailableError,
)
from gemma_cyber.inference.registry import (
    STAGES,
    ModelRecord,
    ModelRegistry,
    PromotionError,
    Stage,
)

__all__ = [
    "STAGES",
    "HealthStatus",
    "InferenceEngine",
    "InferenceError",
    "ModelRecord",
    "ModelRegistry",
    "ModelUnavailableError",
    "PromotionError",
    "ServiceUnavailableError",
    "Settings",
    "Stage",
    "StreamChunk",
    "load_settings",
]
