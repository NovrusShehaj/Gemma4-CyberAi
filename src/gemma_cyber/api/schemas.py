"""Request/response schemas for the HTTP API (validation + stable contracts).

Pydantic validates and bounds every request field at the boundary, so malformed
or oversized input is rejected with a 422 before it reaches the model — part of
the Phase 9 input-validation controls.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Upper bound on prompt size accepted by the API. A 4B model with a large context
# still should not accept unbounded input over HTTP (DoS / cost control). ~24k
# chars ≈ well within a 128k-token context while capping abuse.
MAX_PROMPT_CHARS = 24_000
MAX_SYSTEM_CHARS = 8_000


class GenerateRequest(BaseModel):
    """A single generation request."""

    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    model: str | None = Field(
        default=None,
        description="Model tag or registry alias (e.g. 'production'). None = server default.",
    )
    system: str | None = Field(default=None, max_length=MAX_SYSTEM_CHARS)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    seed: int | None = None
    num_predict: int | None = Field(default=None, ge=1, le=4096)
    stream: bool = False


class GenerateResponse(BaseModel):
    request_id: str
    model: str
    response: str


class ModelInfo(BaseModel):
    version: str
    stage: str
    ollama_tag: str
    base_model: str
    dataset_version: str | None = None
    passed_eval: bool = False
    experiment: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    production: str | None = None


class HealthResponse(BaseModel):
    ok: bool
    service_reachable: bool
    model_present: bool
    model: str
    host: str
    detail: str = ""


class RegisterModelRequest(BaseModel):
    """Admin: register a new model version (requires `admin:models`)."""

    version: str = Field(..., min_length=1, max_length=200)
    base_model: str = Field(default="gemma3:4b", max_length=200)
    dataset_version: str | None = Field(default=None, max_length=200)
    experiment: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)
    overwrite: bool = False


class PromoteRequest(BaseModel):
    """Admin: promote a version to a lifecycle stage (requires `admin:models`)."""

    to: str = Field(..., description="Target stage.")
    reason: str | None = Field(default=None, max_length=500)


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    request_id: str | None = None

    @staticmethod
    def of(error: str, detail: str = "", request_id: str | None = None) -> dict[str, Any]:
        return {"error": error, "detail": detail, "request_id": request_id}
