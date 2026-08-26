"""FastAPI application factory for the Gemma-Cyber inference service.

``create_app`` wires the shared :class:`InferenceEngine` and
:class:`ModelRegistry` behind a small, versioned HTTP surface:

    GET  /                 -> self-contained chat web UI (Phase 7)
    GET  /health           -> liveness (process is up)
    GET  /v1/ready         -> readiness (runtime reachable + model present)
    GET  /v1/models        -> registry listing + current production version
    POST /v1/generate      -> one completion (JSON), or SSE stream when stream=true

Cross-cutting concerns (request ids, security headers, CORS, rate limiting,
bearer auth, structured errors, timeouts) live here; model logic does not. The
engine/registry are injected (via args or app.state) so tests run without Ollama.
"""

# NOTE: intentionally NO `from __future__ import annotations`. FastAPI resolves
# route/dependency annotations at runtime; because FastAPI symbols (Request, etc.)
# are imported locally inside create_app, stringized annotations would fail to
# resolve against module globals and FastAPI would mistype `request` as a query
# parameter. Real (evaluated) annotations bind to the in-scope names.

import json
import logging
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gemma_cyber.api.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
)
from gemma_cyber.api.security import (
    SECURITY_HEADERS,
    RateLimiter,
    token_matches,
)
from gemma_cyber.inference import InferenceEngine, ModelRegistry, Settings, load_settings
from gemma_cyber.inference.errors import (
    InferenceError,
    InferenceTimeoutError,
    ModelUnavailableError,
    ServiceUnavailableError,
)

logger = logging.getLogger("gemma_cyber.api")

_WEB_INDEX = Path(__file__).resolve().parent / "web" / "index.html"

API_VERSION = "v1"


def create_app(
    settings: Settings | None = None,
    *,
    engine: InferenceEngine | None = None,
    registry: ModelRegistry | None = None,
) -> Any:
    """Build the FastAPI app. Import of FastAPI is deferred to here so the core
    package does not require the web stack."""
    from fastapi import Depends, FastAPI, Header, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    settings = settings or load_settings()
    if engine is None:
        engine = InferenceEngine.from_settings(settings)
    if registry is None and settings.registry_path and settings.registry_path.exists():
        registry = ModelRegistry(settings.registry_path)

    limiter = RateLimiter(settings.rate_limit_per_min)

    app = FastAPI(
        title="Gemma-Cyber Inference API",
        version=API_VERSION,
        description="Defensive cybersecurity assistant. Authorized use only.",
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.registry = registry
    app.state.limiter = limiter

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    # -- middleware: request id + security headers --------------------------

    @app.middleware("http")
    async def _request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - defensive catch-all
            logger.exception("unhandled error", extra={"request_id": rid})
            response = JSONResponse(
                status_code=500,
                content=ErrorResponse.of("internal_error", request_id=rid),
            )
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        response.headers["X-Request-ID"] = rid
        return response

    # -- auth + rate-limit dependencies -------------------------------------

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not token_matches(settings.api_token, authorization):
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    def enforce_rate_limit(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        if settings.api_token:
            # When authenticated, bucket by token so shared IPs aren't penalized.
            client = "token"
        if not limiter.allow(client):
            from fastapi import HTTPException

            raise HTTPException(status_code=429, detail="rate limit exceeded")

    # -- routes -------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> Any:
        if _WEB_INDEX.exists():
            return HTMLResponse(_WEB_INDEX.read_text())
        return HTMLResponse("<h1>Gemma-Cyber</h1><p>Web UI asset missing.</p>")

    @app.get("/app.js", include_in_schema=False)
    async def app_js() -> Any:
        from fastapi.responses import Response

        js_path = _WEB_INDEX.parent / "app.js"
        if not js_path.exists():
            return Response("// missing", media_type="application/javascript")
        return Response(js_path.read_text(), media_type="application/javascript")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness: the process is up. Cheap, never touches the model runtime."""
        return {"status": "ok"}

    @app.get(f"/{API_VERSION}/ready", response_model=HealthResponse)
    async def ready() -> Any:
        status = engine.health()
        code = 200 if status.ok else 503
        return JSONResponse(status_code=code, content=status.to_dict())

    @app.get(f"/{API_VERSION}/models", response_model=ModelsResponse)
    async def models() -> Any:
        if registry is None:
            return ModelsResponse(models=[], production=None)
        recs = registry.list()
        prod = registry.production()
        return ModelsResponse(
            models=[
                ModelInfo(
                    version=r.version, stage=r.stage, ollama_tag=r.ollama_tag,
                    base_model=r.base_model, dataset_version=r.dataset_version,
                    passed_eval=r.passed_eval, experiment=r.experiment,
                )
                for r in recs
            ],
            production=prod.version if prod else None,
        )

    def _resolve_engine(model: str | None) -> InferenceEngine:
        """Per-request engine for a requested model alias, else the default."""
        if not model or model == engine.model:
            return engine
        return InferenceEngine.from_settings(settings, model=model)

    @app.post(
        f"/{API_VERSION}/generate",
        response_model=GenerateResponse,
        responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse},
                   503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
        dependencies=[Depends(require_auth), Depends(enforce_rate_limit)],
    )
    async def generate(req: GenerateRequest, request: Request) -> Any:
        rid = request.state.request_id
        try:
            eng = _resolve_engine(req.model)
        except InferenceError as exc:
            return JSONResponse(status_code=400,
                                content=ErrorResponse.of("bad_model", str(exc), rid))

        gen_kwargs: dict[str, Any] = {"request_id": rid}
        if req.system is not None:
            gen_kwargs["system"] = req.system
        if req.temperature is not None:
            gen_kwargs["temperature"] = req.temperature
        if req.seed is not None:
            gen_kwargs["seed"] = req.seed
        if req.num_predict is not None:
            gen_kwargs["num_predict"] = req.num_predict

        if req.stream:
            def _sse() -> Iterator[str]:
                try:
                    for chunk in eng.stream(req.prompt, **gen_kwargs):
                        if chunk.text:
                            yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'request_id': rid})}\n\n"
                except InferenceError as exc:
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n"

            return StreamingResponse(
                _sse(), media_type="text/event-stream",
                headers={"X-Request-ID": rid, "Cache-Control": "no-cache"},
            )

        try:
            result = eng.generate(req.prompt, **gen_kwargs)
        except ModelUnavailableError as exc:
            return JSONResponse(status_code=503,
                                content=ErrorResponse.of("model_unavailable", str(exc), rid))
        except InferenceTimeoutError as exc:
            return JSONResponse(status_code=504,
                                content=ErrorResponse.of("timeout", str(exc), rid))
        except ServiceUnavailableError as exc:
            return JSONResponse(status_code=503,
                                content=ErrorResponse.of("service_unavailable", str(exc), rid))
        except InferenceError as exc:
            return JSONResponse(status_code=500,
                                content=ErrorResponse.of("inference_error", str(exc), rid))

        return GenerateResponse(request_id=rid, model=result.model, response=result.text)

    return app
