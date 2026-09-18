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
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from gemma_cyber.api.auth import (
    ANON_PRINCIPAL_SUBJECT,
    STATIC_PRINCIPAL_SUBJECT,
    AuthError,
    AuthSettings,
    AuthUnavailableError,
    Principal,
    TokenVerifier,
    bearer_token,
)
from gemma_cyber.api.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    PromoteRequest,
    RegisterModelRequest,
)
from gemma_cyber.api.security import (
    Capacity,
    RateLimiter,
    safe_request_id,
    security_headers,
    token_matches,
)
from gemma_cyber.inference import InferenceEngine, ModelRegistry, Settings, load_settings
from gemma_cyber.inference.errors import (
    InferenceError,
    InferenceTimeoutError,
    ModelUnavailableError,
    RegistryError,
    ServiceUnavailableError,
)
from gemma_cyber.inference.registry import RegistryReadOnlyError

logger = logging.getLogger("gemma_cyber.api")

_WEB_INDEX = Path(__file__).resolve().parent / "web" / "index.html"

API_VERSION = "v1"

# Scope required to mutate the model registry (promote/register/mark-evaluated).
SCOPE_ADMIN_MODELS = "admin:models"

# In-process cache for GET /v1/ready so unauthenticated probes do not hammer Ollama.
READY_CACHE_TTL_S = 10.0


def public_ready_payload(status: Any) -> dict[str, Any]:
    """Ready JSON for HTTP clients: no runtime URL and no filesystem paths."""
    detail = status.detail or ""
    lowered = detail.lower()
    if "://" in detail or "ollama" in lowered or "/" in detail:
        if not status.ok:
            if not getattr(status, "service_reachable", True):
                detail = "model runtime unreachable"
            elif not getattr(status, "model_present", True):
                detail = "configured model is not available"
            else:
                detail = "not ready"
        else:
            detail = ""
    return {
        "ok": status.ok,
        "service_reachable": status.service_reachable,
        "model_present": status.model_present,
        "model": status.model,
        "detail": detail,
    }


def _client_inference_error(exc: InferenceError) -> tuple[int, str, str]:
    """Map an inference exception to (status, error code, public detail)."""
    if isinstance(exc, InferenceTimeoutError):
        return 504, "timeout", "generation timed out"
    if isinstance(exc, ModelUnavailableError):
        return 503, "model_unavailable", "requested model is not available"
    if isinstance(exc, ServiceUnavailableError):
        return 503, "service_unavailable", "model runtime unavailable"
    return 500, "inference_error", "generation failed"


def _registry_client_error(exc: RegistryError) -> tuple[int, str, str]:
    """Map a registry exception to (status, error code, public detail)."""
    if isinstance(exc, RegistryReadOnlyError):
        return 503, "registry_read_only", (
            "model registry is read-only (GitOps); manage it via reviewed "
            "source-controlled changes"
        )
    msg = str(exc).lower()
    if "already registered" in msg:
        return 409, "conflict", "model version already registered"
    if "no registered model" in msg:
        return 404, "not_found", "model version not found"
    if (
        "cannot promote" in msg
        or "illegal transition" in msg
        or "unknown stage" in msg
    ):
        return 422, "promotion_gate", "promotion is not allowed for this version"
    return 400, "registry_error", "registry operation failed"


def create_app(
    settings: Settings | None = None,
    *,
    engine: InferenceEngine | None = None,
    registry: ModelRegistry | None = None,
    auth_settings: AuthSettings | None = None,
    verifier: TokenVerifier | None = None,
) -> Any:
    """Build the FastAPI app. Import of FastAPI is deferred to here so the core
    package does not require the web stack.

    Auth mode is resolved from configuration:
      * JWT mode   — ``auth_settings.enabled`` (Auth0 domain + audience set).
      * static dev — no JWT config but ``settings.api_token`` set (shared token,
        never grants admin scopes).
      * open dev   — neither; requests are anonymous.
    In ``GEMMA_CYBER_ENV=prod`` at least one auth mode MUST be configured or the
    app refuses to start (fail closed — no accidental open prod).
    """
    from contextlib import asynccontextmanager

    from fastapi import Depends, FastAPI, Header, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    settings = (settings or load_settings()).validate()
    auth_settings = (auth_settings or AuthSettings.from_env()).validate(
        hosted=settings.hosted
    )
    if engine is None:
        engine = InferenceEngine.from_settings(settings)
    if registry is None and settings.registry_path and settings.registry_path.exists():
        # Hosted deployments open the registry read-only (GitOps): runtime writes
        # are refused at the persistence layer, not just skipped at the route.
        registry = ModelRegistry(
            settings.registry_path, read_only=not settings.registry_writable
        )

    jwt_mode = auth_settings.enabled
    static_mode = bool(settings.api_token)
    auth_configured = jwt_mode or static_mode

    # Fail closed in production: never run an unauthenticated public API by accident.
    if settings.environment == "prod" and not auth_configured:
        raise RuntimeError(
            "GEMMA_CYBER_ENV=prod but no authentication is configured. Set "
            "GEMMA_CYBER_AUTH_DOMAIN + GEMMA_CYBER_AUTH_AUDIENCE (Auth0, recommended) "
            "or GEMMA_CYBER_API_TOKEN (static, dev-grade). Refusing to start."
        )
    if settings.environment == "prod" and static_mode and not jwt_mode:
        logger.warning(
            "production is using the static API token; prefer Auth0 JWT "
            "(GEMMA_CYBER_AUTH_DOMAIN/AUDIENCE) for real identity + authorization."
        )
    if settings.hosted and settings.registry_writable and not auth_configured:
        raise RuntimeError(
            "hosted environment with a writable registry requires authentication. "
            "Set GEMMA_CYBER_AUTH_DOMAIN + GEMMA_CYBER_AUTH_AUDIENCE (Auth0) "
            "or GEMMA_CYBER_API_TOKEN. Refusing to start."
        )

    if jwt_mode and verifier is None:
        verifier = TokenVerifier(auth_settings)

    # Browser SPA (Auth0 Authorization Code + PKCE) configuration. The client id is
    # PUBLIC config (never a secret); the domain/audience mirror the server's JWT
    # settings so the token the browser gets is exactly what the API validates.
    web_client_id = os.environ.get("GEMMA_CYBER_WEB_AUTH0_CLIENT_ID", "").strip()
    web_auth_enabled = jwt_mode and bool(web_client_id)
    if jwt_mode and settings.hosted and not web_client_id:
        logger.warning(
            "JWT auth is enabled but GEMMA_CYBER_WEB_AUTH0_CLIENT_ID is unset; the "
            "shipped browser UI cannot sign in and every /v1/generate will 401. Set "
            "the SPA client id (public) to enable the web login flow."
        )
    # Only widen CSP connect-src/form-action to the Auth0 origin when the SPA needs it.
    resp_headers = security_headers(auth_settings.domain if web_auth_enabled else "")

    limiter = RateLimiter(settings.rate_limit_per_min)
    capacity = Capacity(settings.max_concurrent_generations)

    # In hosted mode hide interactive API docs by default (reduce discovery surface).
    docs_url = None if settings.hosted else "/docs"
    redoc_url = None if settings.hosted else "/redoc"
    openapi_url = None if settings.hosted else "/openapi.json"

    @asynccontextmanager
    async def _lifespan(_app: Any) -> Any:
        # Log resolved posture at startup (secret-free) and shutdown outcome so an
        # operator can confirm the deployment mode from logs alone.
        logger.info(
            "api ready",
            extra={"route": settings.environment, "model": settings.model,
                   "auth_result": app.state.auth_mode},
        )
        try:
            yield
        finally:
            # Uvicorn stops accepting new connections and drains in-flight ones on
            # signal; we record the outcome and how much capacity was still in use.
            logger.info("api shutdown", extra={"model": str(capacity.active)})

    app = FastAPI(
        title="Gemma-Cyber Inference API",
        version=API_VERSION,
        description="Defensive cybersecurity assistant. Authorized use only.",
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.registry = registry
    app.state.limiter = limiter
    app.state.capacity = capacity
    app.state.auth_settings = auth_settings
    app.state.verifier = verifier
    app.state.auth_mode = "jwt" if jwt_mode else ("static" if static_mode else "open")
    app.state._ready_cache = None  # (monotonic_ts, payload, status_code)

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
        rid = safe_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = rid
        started = time.monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:  # pragma: no cover - defensive catch-all
            logger.exception("unhandled error", extra={"request_id": rid})
            response = JSONResponse(
                status_code=500,
                content=ErrorResponse.of("internal_error", request_id=rid),
            )
        for k, v in resp_headers.items():
            response.headers.setdefault(k, v)
        response.headers["X-Request-ID"] = rid
        # Structured access log: no token/prompt content, just operational signal.
        logger.info(
            "request",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
            },
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError):
        # Do not echo request `input` (prompts) back to the client.
        rid = getattr(request.state, "request_id", None)
        errors = []
        for err in exc.errors():
            safe = {k: v for k, v in err.items() if k != "input"}
            ctx = safe.get("ctx")
            if isinstance(ctx, dict):
                safe["ctx"] = {ck: cv for ck, cv in ctx.items() if ck != "input"}
            errors.append(safe)
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "detail": "invalid request",
                "request_id": rid,
                "errors": errors,
            },
        )

    # -- authentication + authorization dependencies ------------------------

    from fastapi import HTTPException

    def get_principal(
        request: Request, authorization: str | None = Header(default=None)
    ) -> Principal:
        """Resolve the caller to a Principal, or raise 401/503.

        JWT mode verifies the token; static mode compares the shared token; open
        mode yields an anonymous principal. Auth failures are logged (never the
        token itself) with the request id for correlation.
        """
        rid = getattr(request.state, "request_id", "-")
        token = bearer_token(authorization)
        if jwt_mode:
            if token is None:
                logger.warning("authn failure: missing bearer token",
                               extra={"request_id": rid})
                raise HTTPException(status_code=401, detail="missing bearer token")
            try:
                principal = verifier.verify(token)  # type: ignore[union-attr]
            except AuthUnavailableError as exc:
                logger.error("authn unavailable", extra={"request_id": rid})
                raise HTTPException(
                    status_code=503, detail="authentication provider unavailable"
                ) from exc
            except AuthError as exc:
                logger.warning("authn failure: %s", exc, extra={"request_id": rid})
                raise HTTPException(status_code=401, detail=str(exc)) from exc
            request.state.principal = principal
            return principal
        if static_mode:
            if not token_matches(settings.api_token, authorization):
                logger.warning("authn failure: bad static token",
                               extra={"request_id": rid})
                raise HTTPException(status_code=401, detail="invalid or missing token")
            principal = Principal(subject=STATIC_PRINCIPAL_SUBJECT, method="static")
            request.state.principal = principal
            return principal
        # Open (dev) mode: anonymous, no scopes.
        principal = Principal(subject=ANON_PRINCIPAL_SUBJECT, method="anonymous")
        request.state.principal = principal
        return principal

    def require_authenticated(
        principal: Principal = Depends(get_principal),
    ) -> Principal:
        # In open mode the anonymous principal is allowed (local dev convenience).
        return principal

    def require_scopes(*needed: str):
        """Dependency factory: require ALL of ``needed`` scopes on the principal.

        Open (dev) mode fail-opens so local development stays frictionless.
        Hosted mode never grants anonymous admin: missing auth is 401.
        """
        def _dep(request: Request, principal: Principal = Depends(get_principal)) -> Principal:
            rid = getattr(request.state, "request_id", "-")
            if not auth_configured:
                if settings.hosted:
                    logger.warning(
                        "authz failure: hosted privileged route with no auth configured",
                        extra={"request_id": rid},
                    )
                    raise HTTPException(status_code=401, detail="authentication required")
                return principal
            if not principal.has_all(needed):
                logger.warning(
                    "authz failure: subject=%s missing scopes=%s",
                    principal.subject, ",".join(needed), extra={"request_id": rid},
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"missing required permission(s): {', '.join(needed)}",
                )
            return principal

        return _dep

    def enforce_rate_limit(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        principal = getattr(request.state, "principal", None)
        if principal is not None and principal.subject not in (
            ANON_PRINCIPAL_SUBJECT,
        ):
            # Bucket authenticated callers by identity, not shared IP.
            client = f"sub:{principal.subject}"
        if not limiter.allow(client):
            retry_after = str(limiter.retry_after_s(client))
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": retry_after},
            )

    # -- routes -------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> Any:
        if _WEB_INDEX.exists():
            # HTML carries auth-state UI, so it must never be cached by shared caches.
            return HTMLResponse(_WEB_INDEX.read_text(),
                                headers={"Cache-Control": "no-store"})
        return HTMLResponse("<h1>Gemma-Cyber</h1><p>Web UI asset missing.</p>")

    def _asset(name: str, media_type: str) -> Any:
        from fastapi.responses import Response

        path = _WEB_INDEX.parent / name
        if not path.exists():
            return Response("", media_type=media_type, status_code=404)
        # Not content-hashed, so revalidate rather than cache immutably.
        return Response(path.read_text(), media_type=media_type,
                        headers={"Cache-Control": "no-cache"})

    @app.get("/app.js", include_in_schema=False)
    async def app_js() -> Any:
        return _asset("app.js", "application/javascript")

    @app.get("/styles.css", include_in_schema=False)
    async def styles_css() -> Any:
        return _asset("styles.css", "text/css")

    @app.get("/config.json", include_in_schema=False)
    async def web_config() -> Any:
        """Public bootstrap config for the browser SPA. No secrets — the SPA client
        id is public; domain/audience mirror the API's JWT validation settings."""
        from fastapi.responses import JSONResponse as _JSON

        return _JSON(
            {
                "env": settings.environment,
                "hosted": settings.hosted,
                "model": settings.model,
                "auth": {
                    "enabled": web_auth_enabled,
                    "domain": auth_settings.domain if web_auth_enabled else "",
                    "clientId": web_client_id if web_auth_enabled else "",
                    "audience": auth_settings.audience if web_auth_enabled else "",
                },
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness: the process is up. Cheap, never touches the model runtime."""
        return {"status": "ok"}

    @app.get(f"/{API_VERSION}/ready", response_model=HealthResponse)
    async def ready() -> Any:
        now = time.monotonic()
        cached = app.state._ready_cache
        if cached is not None and (now - cached[0]) < READY_CACHE_TTL_S:
            payload, code = cached[1], cached[2]
        else:
            status = engine.health()
            payload = public_ready_payload(status)
            code = 200 if status.ok else 503
            app.state._ready_cache = (now, payload, code)
        return JSONResponse(status_code=code, content=payload)

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
        """Per-request engine for a requested model, honoring the override policy.

        When ``allow_client_overrides`` is off (the hosted default), a client may
        only select a model that the registry knows — a registered version or a
        stage alias like ``production``. An arbitrary raw Ollama tag is rejected so
        a caller cannot pull an unreviewed model into the serving path. When the
        policy is on (self-host/expert mode), any tag is accepted as before.
        """
        if not model or model == engine.model:
            return engine
        if not settings.allow_client_overrides:
            known_alias = model in ("production", "candidate", "evaluated", "experimental")
            known_version = registry is not None and model in {
                r.version for r in registry.list()
            }
            if not (known_alias or known_version):
                raise InferenceError(
                    f"model {model!r} is not a released model; "
                    "select a registered version or stage alias"
                )
        return InferenceEngine.from_settings(settings, model=model)

    @app.post(
        f"/{API_VERSION}/generate",
        response_model=GenerateResponse,
        responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse},
                   503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
        dependencies=[Depends(require_authenticated), Depends(enforce_rate_limit)],
    )
    async def generate(req: GenerateRequest, request: Request) -> Any:
        from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

        rid = request.state.request_id
        try:
            eng = _resolve_engine(req.model)
        except InferenceError as exc:
            logger.warning("bad model selection: %s", exc, extra={"request_id": rid})
            return JSONResponse(status_code=400,
                                content=ErrorResponse.of(
                                    "bad_model",
                                    "model is not a released version or stage alias",
                                    rid,
                                ))

        gen_kwargs: dict[str, Any] = {"request_id": rid}
        # Product policy: unless client overrides are explicitly enabled, the
        # server owns the safety/system prompt — a client-supplied `system` is
        # dropped (never silently swaps the production prompt).
        if req.system is not None and settings.allow_client_overrides:
            gen_kwargs["system"] = req.system
        if req.temperature is not None:
            gen_kwargs["temperature"] = req.temperature
        if req.seed is not None:
            gen_kwargs["seed"] = req.seed
        if req.num_predict is not None:
            gen_kwargs["num_predict"] = req.num_predict

        # Admission control: bound concurrent generations; reject deterministically
        # when saturated rather than queueing and starving the single Ollama host.
        if not capacity.acquire():
            logger.warning("generate rejected: at capacity", extra={"request_id": rid})
            return JSONResponse(
                status_code=503,
                content=ErrorResponse.of(
                    "at_capacity", "server is at generation capacity; retry shortly", rid
                ),
                headers={"Retry-After": "5"},
            )

        if req.stream:
            def _sse() -> Iterator[str]:
                try:
                    for chunk in eng.stream(req.prompt, **gen_kwargs):
                        if chunk.text:
                            yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'request_id': rid})}\n\n"
                except InferenceError as exc:
                    logger.warning(
                        "sse inference error: %s", exc, extra={"request_id": rid}
                    )
                    yield (
                        f"data: {json.dumps({'error': 'generation_failed', 'request_id': rid})}\n\n"
                    )

            async def _sse_async() -> Any:
                # Iterate the blocking generator on a worker thread so streaming
                # never blocks the event loop; release capacity when the stream
                # ends, errors, or the client disconnects (GeneratorExit).
                try:
                    async for line in iterate_in_threadpool(_sse()):
                        yield line
                finally:
                    capacity.release()

            return StreamingResponse(
                _sse_async(), media_type="text/event-stream",
                headers={
                    "X-Request-ID": rid,
                    "Cache-Control": "no-store",
                    # Ask compatible proxies (nginx) not to buffer the SSE stream.
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            # Run blocking inference off the event loop, under an optional total
            # request deadline (across retries), distinct from the per-attempt one.
            call = run_in_threadpool(eng.generate, req.prompt, **gen_kwargs)
            if settings.request_deadline_s > 0:
                import asyncio

                result = await asyncio.wait_for(call, timeout=settings.request_deadline_s)
            else:
                result = await call
        except (TimeoutError, InferenceTimeoutError) as exc:
            logger.warning("generate timeout: %s", exc, extra={"request_id": rid})
            return JSONResponse(
                status_code=504,
                content=ErrorResponse.of("timeout", "generation timed out", rid),
            )
        except InferenceError as exc:
            logger.warning("generate failed: %s", exc, extra={"request_id": rid})
            status, code, detail = _client_inference_error(exc)
            return JSONResponse(
                status_code=status, content=ErrorResponse.of(code, detail, rid)
            )
        finally:
            capacity.release()

        return GenerateResponse(request_id=rid, model=result.model, response=result.text)

    # -- admin: model lifecycle (server-side authorization) -----------------
    # These mutate the model registry (the promotion audit trail), so they are
    # gated on the `admin:models` permission carried in the signed token. The
    # authorization decision is made here, server-side — never trusting a
    # client-supplied role — mirroring the CLI's `gemma-cyber models` commands.

    def _require_registry() -> ModelRegistry:
        if registry is None:
            raise HTTPException(status_code=503, detail="no model registry configured")
        if registry.read_only:
            # GitOps/read-only hosted mode: runtime mutation is deliberately
            # disabled. Lifecycle changes go through reviewed source-controlled
            # registry updates, not this endpoint. (Authorization is still checked
            # first, so a non-admin sees 403, an admin sees this 503.)
            raise HTTPException(
                status_code=503,
                detail="model registry is read-only (GitOps); manage it via reviewed "
                "source-controlled changes",
            )
        return registry

    def _subject(request: Request) -> str:
        principal = getattr(request.state, "principal", None)
        return principal.subject if principal is not None else "unknown"

    @app.post(
        f"/{API_VERSION}/admin/models/register",
        dependencies=[Depends(require_scopes(SCOPE_ADMIN_MODELS))],
        responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse},
                   503: {"model": ErrorResponse}},
    )
    async def admin_register(req: RegisterModelRequest, request: Request) -> Any:
        from gemma_cyber.inference.registry import ModelRecord

        rid = getattr(request.state, "request_id", None)
        reg = _require_registry()
        rec = ModelRecord(
            version=req.version, base_model=req.base_model,
            dataset_version=req.dataset_version, experiment=req.experiment,
            notes=req.notes,
        )
        try:
            reg.register(rec, overwrite=req.overwrite, subject=_subject(request))
        except RegistryError as exc:
            logger.warning("admin register failed: %s", exc, extra={"request_id": rid})
            status, code, detail = _registry_client_error(exc)
            return JSONResponse(
                status_code=status, content=ErrorResponse.of(code, detail, rid)
            )
        logger.info("admin register version=%s subject=%s", req.version, _subject(request))
        return {"version": rec.version, "stage": rec.stage}

    @app.post(
        f"/{API_VERSION}/admin/models/{{version}}/mark-evaluated",
        dependencies=[Depends(require_scopes(SCOPE_ADMIN_MODELS))],
        responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse},
                   503: {"model": ErrorResponse}},
    )
    async def admin_mark_evaluated(version: str, passed: bool, request: Request,
                                   eval_ref: str | None = None) -> Any:
        rid = getattr(request.state, "request_id", None)
        reg = _require_registry()
        try:
            rec = reg.mark_evaluated(
                version, passed=passed, eval_ref=eval_ref, subject=_subject(request)
            )
        except RegistryError as exc:
            logger.warning(
                "admin mark-evaluated failed: %s", exc, extra={"request_id": rid}
            )
            status, code, detail = _registry_client_error(exc)
            return JSONResponse(
                status_code=status, content=ErrorResponse.of(code, detail, rid)
            )
        logger.info("admin mark-evaluated version=%s passed=%s subject=%s",
                    version, passed, _subject(request))
        return {"version": rec.version, "stage": rec.stage, "passed_eval": rec.passed_eval}

    @app.post(
        f"/{API_VERSION}/admin/models/{{version}}/promote",
        dependencies=[Depends(require_scopes(SCOPE_ADMIN_MODELS))],
        responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse},
                   422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def admin_promote(version: str, req: PromoteRequest, request: Request) -> Any:
        from typing import cast

        from gemma_cyber.inference.registry import Stage

        rid = getattr(request.state, "request_id", None)
        reg = _require_registry()
        try:
            # promote() validates unknown stages and raises; cast satisfies the typed API.
            rec = reg.promote(
                version, cast(Stage, req.to), reason=req.reason, subject=_subject(request)
            )
        except RegistryError as exc:
            logger.warning("admin promote failed: %s", exc, extra={"request_id": rid})
            status, code, detail = _registry_client_error(exc)
            return JSONResponse(
                status_code=status, content=ErrorResponse.of(code, detail, rid)
            )
        logger.info("admin promote version=%s -> %s subject=%s",
                    version, rec.stage, _subject(request))
        return {"version": rec.version, "stage": rec.stage}

    return app
