# Gemma4/CyberAi Production Readiness Plan

## 1. Executive Summary

### Current posture

The repository is a credible single-host, self-hostable inference product, but it is **not ready for a public, Auth0-protected web launch without the work below**. The core API already has strong foundations: bounded request schemas, a production fail-closed guard, RS256/JWKS/issuer/audience JWT validation, server-side `admin:models` authorization, security headers, request IDs, health endpoints, a non-root API image, and CI checks.

The primary launch blockers are integration and operational correctness rather than a missing framework:

| Priority | Blocker | Why it blocks launch |
|---|---|---|
| P0 | The shipped browser client (`src/gemma_cyber/api/web/app.js`) sends no bearer token. | In JWT/static-auth mode every browser generation request receives 401; there is no login, callback, logout, or token-renewal flow. |
| P0 | `docker-compose.yml` mounts the model registry `:ro`, while authenticated admin endpoints write it. | Model lifecycle mutations will fail in the documented Compose topology; no durable, concurrency-safe write strategy is defined. |
| P0 | The repo’s only registry records show no `production` model and explicitly mark the specialized model unproven. | A production product must not claim or serve an unevaluated specialty model; release selection needs a verified, reproducible model artifact and a deliberate serving configuration. |
| P1 | Dependency resolution is not locked in CI or Docker build. | `uv.lock` exists, but CI uses `uv pip install -e ".[dev]"` and Docker uses `pip install ".[api]"`; both can resolve newer versions than the committed lockfile. |
| P1 | Public deployment boundary is documented but not shipped or tested. | TLS/HSTS, edge rate limits, request-body limits, forwarded-client handling, and production proxy behavior rely on an external reverse proxy with no versioned configuration or acceptance test. |
| P1 | The API does blocking Ollama I/O and retry sleeps from async route execution, with no application concurrency/admission limit. | A small number of long generations can consume request workers and make a single-host instance unreliable or susceptible to resource exhaustion. |

Recommended execution strategy: first establish a production configuration contract and settle the web-session architecture; then make Auth0 usable end-to-end; next protect capacity and registry mutations; then complete UI modernization, operational instrumentation, and deterministic delivery. Keep the current Python/FastAPI + vanilla same-origin web architecture unless an implementation task uncovers a concrete limitation. A wholesale frontend replacement is not justified by repository evidence.

## 2. Repository Architecture Observed

### Verified runtime architecture

```text
Browser (same-origin HTML/JS) ─┐
                              ├─ FastAPI API ─ InferenceEngine ─ OllamaClient ─ Ollama
Local CLI ────────────────────┘       │                │
                                      │                └─ JSON model registry
                                      └─ Auth0/OIDC JWT verifier (when configured)
```

- Python package: `src/gemma_cyber`, Python `>=3.11,<3.13`, Hatchling build; console commands are declared in `pyproject.toml` as `gemma-cyber` and `gemma-cyber-serve`.
- API entry point: `src/gemma_cyber/api/server.py:run` starts Uvicorn; `src/gemma_cyber/api/app.py:create_app` builds the FastAPI app.
- Web client: static self-contained `src/gemma_cyber/api/web/index.html` plus `src/gemma_cyber/api/web/app.js`, served by explicit `/` and `/app.js` routes in `api/app.py`. There is no frontend package manifest, bundler, component framework, or client-side test tooling in the repository.
- API surface: `/health`, `/v1/ready`, `/v1/models`, `/v1/generate` (JSON/SSE), and `/v1/admin/models/*` in `api/app.py`; request and response bounds are in `api/schemas.py`.
- Inference: `src/gemma_cyber/inference/engine.py` centralizes model resolution, readiness, retries, streaming, and error mapping. `src/gemma_cyber/clients/ollama_client.py` calls Ollama’s `/api/tags` and `/api/generate` through `requests`.
- CLI: `src/gemma_cyber/cli/main.py` provides local `ask`, `chat`, `health`, `version`, `eval`, and local JSON-registry `models` commands. It communicates directly with the configured Ollama host through `InferenceEngine`; it is not an HTTP API client and has no Auth0 login/token flow.
- Persistence: `src/gemma_cyber/inference/registry.py` persists model metadata and promotion history to one JSON file. `data/models/registry.json` currently contains `gemma3:4b` at `evaluated` with `passed_eval: false` and `gemma3-cyber:v0.2` at `experimental` with the note that improvement is unproven.
- Authentication: `src/gemma_cyber/api/auth.py` validates JWT signatures via JWKS, issuer, audience, expiry/issued-at, and required claims. `api/app.py` derives permissions from signed `scope` and `permissions` claims and requires `admin:models` on registry write endpoints.
- Containers: `Dockerfile` runs a non-root Python API image; `docker-compose.yml` runs that API plus `ollama/ollama:latest`, exposes API only on `127.0.0.1:8000`, and does not publish Ollama.
- Delivery: `.github/workflows/ci.yml` runs ruff, mypy, pytest, Bandit, pip-audit, and the Gitleaks GitHub Action on push/PR. `uv.lock` is tracked.

### Current request/data flow

1. The page’s `refreshStatus()` fetches public `/v1/ready` and `/v1/models` every 15 seconds.
2. `app.js:send()` posts `{prompt, stream: true}` to `/v1/generate`, parses SSE over `fetch`, and writes model text with `textContent`.
3. The API authenticates `/v1/generate` only when JWT or static mode is configured. It passes client-selected `model`, `system`, sampling values, and prompt to the engine after Pydantic validation.
4. The engine constructs an Ollama client and sends the request to the configured Ollama host. No conversation, user profile, prompt, or response persistence is observed.

## 3. Evidence / Assumptions / Unknowns

### Observed

- The codebase has no user database, session store, RAG system, tool execution layer, agent framework, payment system, or external API client besides Ollama and Auth0 JWKS lookup.
- Production startup fails if `GEMMA_CYBER_ENV=prod` has neither JWT configuration nor a static API token (`api/app.py`).
- The current page is intentionally an unauthenticated local-development UI; `docs/auth.md` explicitly identifies Auth0 PKCE wiring as remaining work.
- API JWT negative cases are covered with a generated test key and fake resolver in `tests/test_api_auth.py`; no live Auth0 tenant test is present.
- API/CLI unit tests use fakes and do not require a live Ollama runtime. `scripts/smoke_test.py` can target a running service.

### Inferred (must be confirmed during implementation)

- The intended public web product is the same-origin page served by FastAPI, because no separate web application or deployment target is present.
- The documented deployment target is a single host, because `docs/deployment.md` and the in-memory limiter describe it as such.
- Admin API registry operations are intended to work in deployed environments, because routes are exposed and documented; the read-only Compose mount conflicts with that intention.

### Missing / unresolved (do not guess)

- Auth0 tenant domain, API identifier, SPA client ID, allowed callback/logout/web-origin URLs, enabled grant types, MFA/attack-protection policy, organization/role model, and environment separation are not inspectable in the repository.
- A production hostname, reverse proxy, TLS certificate issuer, cloud/host provider, registry/image registry, secret manager, deployment orchestrator, backup target, and release owner are not present.
- Required availability/SLO, concurrent-user target, request/token quota, model-memory capacity, data-retention policy, incident escalation path, and privacy/legal requirements have not been established.
- Whether remote/hosted CLI access is a product requirement is not established. Today’s CLI is local-Ollama operator tooling; it must not be represented as an Auth0-enabled remote client.
- No real evaluated model is marked production in `data/models/registry.json`; availability of model weights/artifacts is not verifiable from source control.

## 4. Production Readiness Findings

| Priority | Area | Finding | Evidence | Production Impact | Recommended Action |
|---|---|---|---|---|---|
| P0 | Web/Auth0 | Browser does not obtain or attach a token. | `api/web/app.js` only sets `Content-Type`; `docs/auth.md` says PKCE wiring remains. | Protected chat is unusable; static-token use in browser would expose a shared secret. | Add Authorization Code + PKCE with a public Auth0 SPA client, in-memory token handling, guarded UI states, and E2E tests. |
| P0 | Model release | No registry model is eligible/marked `production`; specialized model is documented as unproven. | `data/models/registry.json`, `README.md`. | Cannot substantiate a specialty-model production release or safe rollback target. | Complete the existing evaluation/promotion process, record artifact provenance/checksum, then set an explicit serving model after human release approval. |
| P0 | Registry/admin | Compose makes `/app/registry` read-only although admin routes call `ModelRegistry.save()`. JSON writes are direct and unlocked. | `docker-compose.yml`; `inference/registry.py:save`; admin routes in `api/app.py`. | Writes fail under the documented stack; concurrent writers can lose/corrupt audit state. | Decide: remove deployed admin mutations and keep registry GitOps/read-only, or introduce a durable writable store with atomic writes/locking and backup. Do not merely remove `:ro` without this decision. |
| P1 | API capacity | Async route invokes synchronous `requests`-based inference and `time.sleep` retry path; no request concurrency cap. | `api/app.py:generate`; `inference/engine.py`; `clients/ollama_client.py`. | Long model calls can starve workers and enable resource exhaustion. | Add bounded admission/concurrency and move blocking inference off the event loop or make the internal client async; expose queue/rejection behavior. |
| P1 | Supply chain | Lockfile is not consumed by Docker/CI; base/runtime images have mutable tags. | `.github/workflows/ci.yml`; `Dockerfile`; `docker-compose.yml`; `uv.lock`. | Non-reproducible builds and unexpected dependency/image changes. | Use lockfile-backed installs, pin immutable base/Ollama image versions or digests, build/test a release image, and publish immutable artifact metadata. |
| P1 | Edge/deploy | TLS/reverse proxy and edge controls are bring-your-own; no deployed configuration test. | `docs/deployment.md`. | Public exposure can omit HTTPS/HSTS, edge rate limit, trusted proxy configuration, and size limits. | Add a versioned reference proxy deployment or formal deployment contract with a deployment-owned implementation; validate it in staging. |
| P1 | Auth/config | Static token remains accepted in production as a warned fallback; auth settings do not validate allowed environment values or configuration completeness beyond domain+audience. | `api/app.py`; `api/auth.py`; `inference/config.py`. | Shared token is not user identity and can leak/replay; invalid deployment config can fail unclearly. | For hosted/public mode reject static auth, validate all production config on startup, and make a documented local/self-host mode explicit. |
| P1 | Authorization/data exposure | `/v1/models`, `/v1/ready`, `/health`, and `/` remain public in JWT mode. | Route definitions and `docs/auth.md`. | Model inventory/runtime detail is externally discoverable; may be acceptable but has not been explicitly decided. | Classify each endpoint; keep only probe-safe data public or require auth/edge restriction. Update smoke tests accordingly. |
| P1 | Safety/product boundary | API accepts arbitrary `system` and `model` values; registry resolution falls back to raw installed Ollama tags. | `api/schemas.py`; `api/app.py:_resolve_engine`; `inference/registry.py:resolve`. | Authenticated users can bypass the default safety prompt or request unreviewed local models, affecting product policy, capacity, and release assurance. | Define a public API policy: server-owned system prompt and allowlisted/released model aliases by default; move override capability behind an explicit privileged/internal mode if retained. |
| P2 | UI/UX | Current page is a minimal light/dark chat shell with inline CSS; no sign-in state, history, cancellation, retries, robust status handling, or automated frontend tests. | `api/web/index.html`, `api/web/app.js`. | Not a polished, accessible professional security product experience. | Modernize in-place with tokenized CSS and small JS modules; add browser-driven E2E/accessibility tests before considering a build pipeline. |
| P2 | HTTP hardening | Security headers exist, but HSTS is appropriately absent from app code (TLS belongs at edge); CSP allows inline styles; streamed responses lack explicit `X-Accel-Buffering: no`; no API cache policy is specified. | `api/security.py`; `api/app.py`. | Edge/browser behavior is incomplete for a public deployment. | Configure HSTS and TLS at proxy only; retain CSP principles, eliminate inline style dependency if practical, set explicit cache/stream proxy headers, and test headers at the external URL. |
| P2 | Observability | Standard logs use free-form formatter and `GEMMA_CYBER_LOG_LEVEL` is loaded but not applied by server startup. Metrics/tracing are documentation-only. | `api/server.py`; `cli/main.py`; `inference/config.py`; `docs/operations.md`. | Correlation is available, but production search/alerts/SLO diagnosis are weak. | Add structured JSON logs/redaction tests, honor configured API log level, and add minimal request/inference metrics or platform integration. |
| P2 | Reliability | Startup does not verify readiness before serving; graceful shutdown/drain and dependency retry policy are not explicit. | `api/server.py`; `api/app.py`; `engine.py`; `docs/deployment.md`. | Traffic can be accepted during a model outage; in-flight responses may be cut on deploy. | Add lifespan/shutdown handling, proxy readiness gating, bounded retry semantics, and release smoke/rollback drills. |
| P2 | Tests | No live Auth0, browser, proxy, load, container, image, or registry-concurrency test is observed. | `tests/`; `.github/workflows/ci.yml`. | Important integration risks are not regression-tested. | Add focused tests in the matrix below; do not claim end-to-end coverage until a staging environment exists. |
| P3 | CLI distribution | CLI has sound local exit codes/JSON behavior but no packaged release, signed artifact, update policy, remote API mode, or Windows installer strategy. | `pyproject.toml`; `cli/main.py`; `docs/cli.md`. | Fine for source/local deployment; not yet a polished distributable product CLI. | Preserve local CLI; establish wheel/release artifacts and only add remote Auth0 device/PKCE flow if product requirements demand it. |

## 5. Web Application Production Plan

### Preserve and modernize the existing architecture

Keep the same-origin, no-framework web client unless an implementation phase demonstrates a need for a build tool. Existing CSP intentionally only permits same-origin script loading, and there are no third-party frontend dependencies to audit. Split the current inline CSS and imperative JS into small first-party files only if FastAPI continues serving them with correct media types and cache policy.

### File-specific work

1. `src/gemma_cyber/api/web/index.html`
   - Replace the single header/chat/footer structure with an accessible application shell: sidebar/navigation on large screens, compact mobile header, main conversation region, composer, status region, and authentication controls.
   - Use landmark elements (`header`, `nav`, `main`, `aside`, `footer`), a visible skip link, meaningful button names, live regions limited to new assistant/error status, and `aria-busy` while generating.
   - Remove the currently hidden `label` pattern and use visually-hidden CSS instead; retain a real accessible label.
   - Add no user-controlled HTML interpolation. Continue rendering model output using text nodes/`textContent`; markdown rendering must not be introduced without a sanitizer and tests.

2. `src/gemma_cyber/api/web/app.js`
   - Add an auth client abstraction and an explicit UI state machine: initializing, signed-out, authenticating, ready, generating, token-expired, unavailable, and error.
   - Send `Authorization: Bearer <access token>` only to same-origin API calls. Do not put a client secret, static API token, access token, or refresh token in source, URL query strings, localStorage, sessionStorage, logs, or error messages.
   - Make sign-in, login redirect/callback restoration, token acquisition, expiry/renewal failure, logout, and failed API responses deterministic. Cancel in-flight `fetch` with `AbortController`; restore input, focus, and send controls reliably.
   - Avoid `innerHTML` for trusted model/stage response too; current `stage-badge` uses it even though API data is server-generated. Construct child elements with `textContent` for a uniform safe rendering rule.
   - Treat a 401 as a recoverable auth state, 429 with user-facing retry guidance, 503/504 as service states, and stream parsing errors as a failed request with request ID when supplied.
   - Do not display server detail strings blindly if their public disclosure policy changes; show a friendly user message plus safe correlation ID.

3. `src/gemma_cyber/api/app.py` and `src/gemma_cyber/api/security.py`
   - Serve every new first-party asset explicitly or replace the ad-hoc asset routes with a tightly scoped static-files implementation. Set correct content types, immutable caching only for content-hashed assets, and `Cache-Control: no-store` for HTML/auth-state-sensitive responses.
   - Document and enforce the endpoint-authentication classification selected in Phase 1. The page must be viable under the selected policy.
   - Add CSP sources only as required by the chosen Auth0 SPA integration. If Auth0 Universal Login is a cross-origin redirect, preserve `script-src 'self'`; do not broaden CSP with unsafe scripts. A custom-domain decision is external.

4. Add browser tests (location to be chosen with the test tool) covering sign-in/out visible behavior, expired token, keyboard sending, cancellation, live messages, 401/429/503 handling, responsive navigation, focus order, and no raw HTML execution from model output. A test framework/dependency is recommended only after choosing a compatible browser runner.

### Web acceptance criteria

- In production JWT mode, a signed-out visitor sees a clear sign-in screen and cannot submit a generation request.
- A valid Auth0 access token for the configured audience enables `/v1/generate`; no sensitive token is persisted in browser storage or exposed in logs/UI.
- The page remains fully usable at 320 CSS pixels, keyboard-only, 200% zoom, and reduced-motion settings.
- Existing same-origin dev mode remains deliberately usable only when the server is configured open for local development; production does not silently become open.
- Automated browser checks verify no XSS from an assistant message and verify no regressions in SSE streaming/cancellation.

## 6. CLI Production Plan

### Current scope and decision

The CLI is a local operator/client tool, not a web/API CLI. `gemma-cyber` calls an Ollama host directly; its `models` commands mutate the local JSON registry directly. This is a legitimate self-host use case, but it bypasses hosted API Auth0 policy by design. Do not bolt Auth0 onto it unless remote hosted CLI access is an approved product requirement.

### Required work

1. `src/gemma_cyber/cli/main.py`, `src/gemma_cyber/inference/config.py`, and `docs/cli.md`
   - Validate CLI-provided host URLs and numeric configuration at parsing/startup; current malformed integer environment values fall back silently in `_env_int`, while malformed `GEMMA_CYBER_AUTH_LEEWAY` or server port can raise at startup. Establish one documented validation policy.
   - Apply `Settings.log_level` in CLI logging or remove the documented behavior. Preserve stdout for successful data and stderr for diagnostics; preserve exit codes in `main.py`.
   - Decide whether `--no-system` and arbitrary `--system` remain self-host-only expert controls. If retained, state that they are not equivalent to a hosted/public policy path.
   - Ensure `--json` outputs a stable documented schema without prompt content when a scripting/privacy mode requires it. The current JSON `ask` output includes the submitted prompt by design.
   - Document supported OS/Python/Ollama versions only after testing them; current repository evidence supports Python 3.11/3.12 but does not demonstrate all OS variants.

2. `src/gemma_cyber/inference/registry.py` and CLI model commands
   - Before enabling registry writes in more than one process, implement atomic replacement, restrictive file permissions, and process locking, with rollback/backup semantics. A local CLI and API must not concurrently edit the current direct-write JSON file.
   - Alternatively make deployed registry management GitOps-only: remove/deprecate write subcommands/API in hosted mode, retain local commands, and require reviewed committed registry changes. This is the preferred smallest production option for the observed single-host architecture.

3. `pyproject.toml`, CI/release workflow, and documentation
   - Build and test a wheel/sdist from the lockfile; publish only immutable versioned artifacts when distribution is authorized.
   - Define installation/update/rollback instructions. No current signed-release, package-index, or auto-update mechanism is observed, so do not claim one.

### Remote CLI (deferred unless approved)

If a hosted API CLI is required, add it as a separate transport mode rather than altering local behavior: explicit `--api-url`, OAuth device authorization or PKCE appropriate to the chosen client type, OS keychain-backed refresh-token storage where supported, no token in command history, explicit non-interactive behavior, HTTPS-only verification, scope-aware errors, and integration tests. This requires Auth0 dashboard changes and a user decision; it is not an automatic Phase 1 task.

## 7. Auth0 Production Plan

### Repository changes

1. Complete the browser authorization flow in `api/web/index.html` and `api/web/app.js` using Authorization Code with PKCE for a public SPA. Use the configured API audience and request only the scopes the web application needs. The existing server expects bearer access tokens and already validates `iss`, `aud`, RS256 signature, `exp`, `iat`, and `sub` in `api/auth.py`.
2. Make `AuthSettings.from_env()` and `load_settings()` strict at production startup: validate allowed algorithm policy (production should explicitly require RS256 unless a reviewed change is made), nonempty/valid domain/audience/issuer relation, nonnegative reasonable leeway, environment enum, URL shape for Ollama/JWKS, CORS origin syntax, rate-limit and inference bound ranges. Never echo secrets in errors.
3. Add an explicit deployment mode policy in `api/app.py`:
   - Local dev may be open only when expressly selected.
   - Hosted/public production must require JWT mode; reject static API-token-only production mode, or require a separately named self-host mode with documentation.
   - Admin endpoints must remain server-side permission-gated; static principals must never receive admin scope.
4. Decide whether public read/status endpoints should remain unauthenticated. If public, reduce their payload to safe probe data and rate-limit at the edge; if private, change the browser initialization/auth flow and probes accordingly.
5. Add tests to `tests/test_api_auth.py` for configuration rejection, token algorithm policy, incomplete bearer syntax/casing as intended, no permission escalation from client input, protected endpoint classification, and any new scope requirement.

### Auth0/environment configuration (human/admin work; not repository-controlled)

For **each isolated environment** (at minimum staging and production), an Auth0 administrator must:

1. Create/configure the API using the exact production API identifier that will become `GEMMA_CYBER_AUTH_AUDIENCE`; require RS256; enable RBAC and add permissions to access tokens.
2. Define least-privilege permissions. `admin:models` is verified in repository code; decide and configure the normal generation permission only if the API begins requiring one rather than accepting every valid audience token.
3. Create a public SPA application for the shipped browser client. Configure exact HTTPS Allowed Callback URLs, Allowed Logout URLs, and Allowed Web Origins for the deployed UI origins; do not use wildcard production origins.
4. Select session lifetime, idle timeout, MFA/adaptive protection, breached-password/attack-protection settings, and consent/privacy settings according to the organization’s policy. These tenant values cannot be inferred from source.
5. Assign the `admin:models` permission only to a tightly controlled admin role/account. Use a separate non-admin test user/client for staging validation.
6. Store only server-side deployment variables (`GEMMA_CYBER_AUTH_DOMAIN`, `GEMMA_CYBER_AUTH_AUDIENCE`, optional reviewed issuer/JWKS URL) in the deployment secret/config system. The SPA client ID is public configuration; never provide a SPA client secret because PKCE clients do not use one.
7. Test tenant key rotation, user/session revocation behavior, and Auth0 outage behavior in staging. Repository code maps JWKS failures to 503; operational policy must decide alerting and user messaging.

### Authentication/authorization validation

- Valid interactive login → code/PKCE callback → access token for exact audience → generate succeeds.
- Missing/malformed/expired/wrong-issuer/wrong-audience/wrong-signature token → 401; JWKS lookup unavailable → 503. The unit suite already covers these server-side cases.
- Valid normal user without `admin:models` → 403 on every registry mutation; admin role token → permitted only for intended lifecycle endpoints.
- Logout clears in-memory app auth state and returns the user to signed-out UI; server access token acceptance ends at expiry, and any Auth0 revocation behavior is verified against the tenant’s chosen configuration.
- Callback with altered/missing state/PKCE verifier is rejected by the client/Auth0 protocol flow; callback URLs are exact environment URLs.

## 8. Security Hardening Plan

### Lightweight threat model

| Element | Repository-grounded assessment |
|---|---|
| Assets | Auth0 access tokens; API availability/model capacity; model registry/provenance audit trail; deployment secrets; prompts/responses in transit; container host/Ollama runtime. |
| Trust boundaries | Browser ↔ public edge/API; API ↔ Auth0 JWKS; API ↔ internal Ollama; CLI/local operator ↔ Ollama and registry filesystem; CI ↔ dependencies/container images. |
| Entry points | HTTP endpoints, SSE stream, CORS browser requests, Auth0 redirect/callback to be added, CLI arguments/stdin/environment, JSON registry file, container image/build inputs. |
| Likely attacker capabilities | Anonymous internet user; authenticated standard user; compromised/static token holder; malicious prompt author; abusive client creating long streams; dependency/image supply-chain attacker; local user able to edit host-mounted registry. |
| Highest-risk abuse paths | Token-less web flow workarounds/static token exposure; capacity exhaustion through generations; privilege abuse of registry mutations; unsafe model/prompt override; registry corruption/lost audit updates; proxy misconfiguration exposing API/Ollama or omitting TLS; dependency/image drift. |

### P0/P1 security tasks

1. **Authentication/browser secret handling** — implement PKCE as described in §7; ban static-token use in browser code. Add a token-redaction test strategy for client/server logs and browser telemetry if later introduced.
2. **Authorization and model governance** — make generation policy explicit. In hosted mode, permit only released model aliases/records and a server-owned safety prompt. Keep `admin:models` server enforcement and apply least privilege to any new admin endpoints.
3. **Availability/abuse controls** — extend `Settings`/`api/app.py` with bounded concurrent generations and a maximum pending queue or immediate 429/503 behavior. Preserve per-identity limits after authentication; configure an edge rate limiter for IP-level pre-auth protection. Enforce request body size at the reverse proxy and app server; Pydantic field bounds alone do not prevent oversized HTTP bodies from reaching the process.
4. **Registry integrity** — choose GitOps/read-only or a proper durable mutable store. If mutable, use atomic write+fsync semantics, lock writers, validate every deserialized field, restrict file ownership/permissions, back up before writes, and test crash/concurrency recovery. Ensure audit identity and reason are stored on admin changes; current history records transition/reason but not the authenticated subject.
5. **Network/edge hardening** — keep Ollama unexposed. Publish API only behind HTTPS; set HSTS, TLS versions/ciphers and trusted-proxy behavior at the proxy; do not configure HSTS from plaintext-local app behavior. Restrict CORS to exact origins. Avoid treating arbitrary `X-Request-ID` as trusted without a length/character policy; otherwise a client can inject large/log-unfriendly values.
6. **HTTP/frontend** — retain default-deny CSP structure, `nosniff`, `DENY`, referrer and permissions policies. Add `Cache-Control: no-store` where auth-sensitive and `X-Accel-Buffering: no` for SSE behind compatible proxies. Keep all output text-rendered. Add a CSP test on new asset/auth flow.
7. **Timeout/retry correctness** — enforce a total request budget, not merely per-Ollama-attempt timeout; only retry failures proven transient and safe; add jitter to retry if retries remain. Ensure client disconnect stops/does not continue needless generation where the server/runtime supports cancellation.

### P2 security tasks

- Disable or restrict `/docs`, `/redoc`, and OpenAPI schema in public production if the security policy does not want API discovery; otherwise protect/admin-document them deliberately. Current FastAPI defaults expose interactive docs.
- Enforce structured, redacted logs: avoid error messages that could include backend host/internal data or prompt material; cap request-ID length; include no authorization header.
- Run a container image scan before publishing and gate known critical/high findings subject to a documented exception process. Keep Bandit, pip-audit, and Gitleaks; do not claim image scanning is already configured.
- Generate an SBOM and attach it to release artifacts only after selecting a release tool; record Python, lockfile, base image digest, Ollama image digest, model tag/digest, Git commit, and registry version for traceability.
- Review dependency update cadence and GitHub Action pinning. Current actions are major tags, not immutable commit SHAs; adopt the organization’s action-pinning policy if supply-chain assurance requires it.

## 9. Cybersecurity UI/UX Modernization Plan

### Design system: dark, restrained, professional

Implement with CSS custom properties in the existing frontend; no framework swap is needed.

| System concern | Direction |
|---|---|
| Color tokens | Dark-first `canvas`, `surface-1/2/3`, `border-subtle/strong`, `text-primary/secondary/tertiary`; one cyan/blue-teal action accent; semantic success/warning/danger/info tokens. Avoid hard-coded component colors and neon green as a primary brand color. |
| Contrast | Meet WCAG AA for normal text and controls; do not rely on the status dot alone—pair color with icon/text. Test dark theme rather than using the current OS-preference light default. |
| Typography | System sans stack for UI plus the existing monospace stack for short identifiers/status values only. Establish display, page title, section title, body, label, metadata, and code scales. |
| Spacing/sizing | Define a compact 4px-based space scale; predictable 32–48px controls; content max widths; 44px minimum target size where touch interaction applies. |
| Shape/elevation | 1px low-contrast borders, 8/12/16px radii, subtle shadow only for floating layers. Prefer layered charcoal surfaces to gradients/glow. |
| Motion | 120–200ms opacity/transform/color transitions, no decorative continuous motion, and `prefers-reduced-motion` fallback. Streaming cursor/status can be understated. |

### Page and component map

| Existing file/element | Implementation target |
|---|---|
| `index.html:header` | Product mark, environment/model status, signed-in identity menu, primary navigation. Show a prominent non-production environment label if staging/dev. |
| `index.html:main#log` | Conversation workspace: safe-use context collapsible after first read, date/request grouping only if data exists, assistant/user response cards with copy button and request ID disclosure. Do not imply retained history because none exists. |
| `index.html:footer/form` | Composer panel with character feedback aligned with the server 24,000-character prompt bound, Send and Cancel states, keyboard hint, validation/error text, and privacy reminder. |
| New/conditional shell regions | A small status/operations panel displaying only data already available (`/v1/ready`, model/stage) and clear empty/unavailable state; no fabricated threat telemetry/dashboard. |
| New auth view | Centered, minimal sign-in experience explaining authorized use and privacy boundaries; Auth0 Universal Login performs credential handling. |
| `app.js:addMessage` | Reusable semantic message-card creator; text only, timestamps only if client-created and useful, status labels, copy feedback, error recovery action. |

### UX behavior

- **Navigation:** Use a simple single-product navigation rather than inventing pages/data the project lacks. On mobile, collapse secondary status/navigation without hiding auth or send controls.
- **Forms/buttons:** Use primary (send/sign in), secondary (cancel/copy), and destructive only where a real destructive action exists. Disabled state must explain why to assistive tech.
- **Statuses:** Model readiness uses icon + text + semantic color. Rate limit errors provide a retry time only if the API provides one; otherwise say the limit was reached without inventing a countdown.
- **Empty/loading/errors:** Keep the existing safety warning but make it scannable. Show skeleton/connection state only while fetching actual status; render SSE streaming progressively; expose errors near the composer and in the assistant card.
- **Accessibility:** Visible focus ring against dark surfaces; skip link; logical tab order; no focus loss during stream completion/error; screen-reader announcements throttled so every streamed token is not announced; test keyboard and reduced-motion behavior.

### Migration sequence

1. Inventory IDs/selectors in `index.html`/`app.js`; write a visual/accessibility acceptance list without changing API contracts.
2. Establish semantic tokens and reset/layout primitives; convert existing page with behavior unchanged.
3. Add auth/session states and error/cancellation behavior alongside the protected API flow.
4. Add reusable message, badge, button, form-field, alert, empty-state, and panel styles in first-party CSS.
5. Add browser/a11y checks and manually review target breakpoints before visual polish. Keep a single controlled dark theme for first release; an optional light theme is not required by the requested direction.

## 10. Deployment & Operational Readiness

### Production configuration and runtime

1. Define an authoritative environment contract in `inference/config.py` and deployment docs. Provide a committed non-secret example only if it does not resemble a working production credential. Validate config before the service accepts traffic.
2. Replace the current commented staging settings in `docker-compose.yml` with separate, explicit development and production composition/configuration paths. Production must set `GEMMA_CYBER_ENV=prod`, JWT domain/audience, approved CORS origins, capacity limits, and an explicit released `GEMMA_CYBER_MODEL`; secrets remain external.
3. Resolve the registry ownership decision before enabling production admin endpoints. Read-only Git-managed registry and disabled runtime mutation is simplest for the observed single-host service. If admin mutation is retained, use a backed-up writable volume/store and prohibit concurrent direct CLI writes.
4. Pin `ollama/ollama:latest` and Python base image to reviewed versions/digests. Build a release image tagged with immutable Git SHA and record its digest. `docs/deployment.md` already recommends immutable production image tags; make this true in the build/release path.
5. Keep Docker’s non-root application user. Review filesystem read/write paths after the registry decision; the app must not require write access to its code directory.

### Health, failure handling, and operations

- Preserve `/health` as cheap liveness and `/v1/ready` as runtime/model readiness. Configure the proxy/orchestrator to send public traffic only after readiness is 200.
- Add an application lifespan/shutdown policy: mark unready, stop accepting new requests, bound drain time, and log shutdown outcome. Exact Uvicorn worker configuration must be selected based on measured model capacity; do not assume multi-worker improves a single Ollama host.
- Set a total generation deadline, capacity/admission policy, retry metrics, and client-disconnect behavior. Use a staging load test to choose the values.
- Convert the existing standard-library logging to structured JSON or explicitly integrate the platform collector. Include timestamp, level, request ID, route, status, latency, model, auth outcome category, and safe error class; redact secrets/prompt/response content.
- Add platform/proxy or application metrics for request count/status, active/queued generations, rejection count, end-to-end and Ollama latency, readiness, and process/resource utilization. Add alerts only after owners and thresholds are agreed.
- Run `scripts/smoke_test.py` against the external TLS endpoint after deploy/rollback. Extend it after endpoint policy changes; it presently exercises health, headers, model listing, validation, generation, optional auth, and page serving.

### Backup/recovery and incident readiness

- Model registry: back up according to the chosen store; test restoration and audit history. A GitOps registry should be protected with branch controls and a documented rollback commit; a writable store needs retained backups and restore validation.
- Model artifacts: maintain a release manifest that links served tag/digest to benchmark/scorecard, source commit, and checksum. Current registry fields support `git_commit`, `gguf_sha256`, and `eval_ref`, but actual values are incomplete for the candidate model.
- Secrets: document rotation/revocation for Auth0 and any temporary static self-host token. Do not log or store tokens in registry/CI artifacts.
- Runbook: update `docs/operations.md` with Auth0 outage, token-expiry/login failure, saturation/429/queue, registry write failure, rollback, and proxy/TLS failure steps; name an owner/on-call process only when supplied externally.

## 11. CI/CD & Supply Chain

### Required changes

1. Make `uv.lock` authoritative. Replace dependency installation in `.github/workflows/ci.yml` with the project’s lockfile-backed UV workflow (for example, commands based on `uv sync --locked` only after verifying the chosen workflow works for this repository). Make Docker install from the same locked dependency graph rather than resolving `.[api]` with pip.
2. Add a CI job that builds the Docker image and performs at least a container startup/configuration check. A full model generation test requires a controlled Ollama test environment and should be a staging/integration job, not faked as container coverage.
3. Add container image vulnerability scanning before publication; use the selected scanner’s nonzero policy and documented reviewed exceptions. Continue existing Bandit, pip-audit, and Gitleaks jobs.
4. Build wheel and sdist, install the built wheel in a clean environment, and run CLI/API import or smoke tests. Validate package contents do include `api/web/index.html` and `api/web/app.js`; this matters because `app.py` reads these at runtime by package-adjacent path.
5. Produce release provenance: source commit, lockfile hash, image digest, package artifacts/checksums, model release metadata. SBOM/attestation tooling should be selected to fit the actual GitHub release/container registry choice, which is currently unknown.
6. Protect release branches/tags and require the CI/security jobs before merge/release through repository settings. Those settings are external to source and must be configured by an administrator.

### Preserve

- `.github/workflows/ci.yml` already runs ruff, mypy, pytest, Bandit medium+, pip-audit, and full-history Gitleaks. Keep these checks and avoid blanket suppression.
- No dependency addition should be introduced for UI styling alone. A browser test runner, Auth0 SPA SDK, or proxy/image scanner is justified only with a pinned, reviewed dependency and accompanying CI coverage.

## 12. Testing & Verification Matrix

| Area | Test | Expected Result | Priority |
|---|---|---|---|
| Existing baseline | `uv run --no-sync ruff check src tests scripts`, `uv run --no-sync mypy src tests scripts`, `uv run --no-sync pytest` | Remain green after changes; replace with lock-backed invocation when CI workflow changes. | P0 |
| Build reproducibility | Fresh CI and Docker builds use committed `uv.lock`; compare resolved packages/artifact checksums. | Same reviewed dependency graph is installed; unexpected resolver drift fails. | P1 |
| API auth unit | Existing `tests/test_api_auth.py` plus config/algorithm/policy tests. | Valid JWT works; malformed/expired/issuer/audience/signature failures are 401; JWKS outage is 503; scope denial is 403. | P0 |
| Auth0 staging E2E | Real public SPA login using staging tenant/client, callback, audience/scopes, logout, expiry/renewal. | Browser obtains a valid token without exposing secrets; protected generation works; sign-out/expiry returns to signed-out state. | P0 |
| Auth protocol negative | Altered callback/state/PKCE verifier; wrong redirect URL; unapproved origin. | Auth0/client rejects flow; no token is accepted or leaked. | P0 |
| Authorization E2E | Normal user and `admin:models` user exercise all admin routes. | Normal user receives 403; only intended admin role succeeds; audit identity is recorded if mutable registry is retained. | P0 |
| API boundary | Empty/24k+ prompt, malformed fields, excessive body at edge, model/system override policy, unknown model. | Bounded errors without inference; policy-restricted values are rejected before Ollama. | P1 |
| API capacity | Concurrent long JSON and SSE generations, cancellation/disconnect, queue/limit saturation. | Event loop stays responsive; bounded active work; deterministic 429/503 behavior; cancelled work does not leak capacity. | P1 |
| Registry integrity | Read-only deployed registry behavior; if mutable, competing writers, interrupted write, restore, permission failure. | Chosen design fails safely, preserves valid JSON/history, and recovers from backup. | P0 |
| Web browser | Keyboard-only sign-in/send/cancel, 320px, 200% zoom, reduced motion, screen-reader smoke, status/error states. | Usable, focused, responsive dark UI with no inaccessible control. | P1 |
| XSS/CSP | Assistant text containing HTML/script-like input; CSP/header test at real proxy URL. | Content stays text; no script executes; expected CSP/HSTS/proxy headers present. | P1 |
| CLI | Existing `tests/test_cli.py`; malformed env/config tests; wheel-installed CLI smoke. | Exit codes/stdout-stderr/JSON remain stable; local mode is explicit and safe. | P1 |
| Container | Build API image, run non-root, inspect no Ollama host publication, verify health and production fail-closed startup. | Image is reproducible, starts only with valid production config, and does not expose runtime unnecessarily. | P1 |
| Security scans | Existing Bandit/pip-audit/Gitleaks plus selected image scan. | No unreviewed actionable findings; exceptions documented. | P1 |
| Operational smoke | `python scripts/smoke_test.py --base-url <external-url> --token "$TOKEN" --expect-auth` after deploying. | Required checks pass against TLS/proxy/Auth0-configured staging and production. | P0 |
| Model release | Run documented benchmark/evaluation, capture scorecard/artifact provenance, promotion/rollback rehearsal. | A named approved model meets documented gate and can be rolled back without registry inconsistency. | P0 |

## 13. Phased Claude Opus 4.8 Execution Plan

### Phase 0 — Confirm release decisions and protect the baseline

**Objective**

Turn unresolved product/deployment choices into explicit inputs before changing security-sensitive interfaces.

**Prerequisites**

- Human answers to §16 open questions, especially hosted-vs-self-host scope, registry ownership, production domain/Auth0 setup, and model-release gate.

**Files / Components**

- `docs/deployment.md`, `docs/auth.md`, `docs/operations.md`, `data/models/registry.json`, `pyproject.toml`, `uv.lock`.

**Tasks**

1. Verify current working tree and baseline test commands; do not overwrite the existing `.claude/PRODUCTION-READINESS-PLAN.md` historical document.
2. Record approved deployment mode(s), endpoint exposure policy, registry write policy, capacity target, and public model/system-override policy in a new or updated operational decision document.
3. Confirm a release candidate model/artifact and promotion evidence, or label public specialty-model launch blocked.

**Acceptance Criteria**

- No later phase has to infer an Auth0 SPA client, hosted CLI requirement, registry writer, or model-release policy.

**Validation**

- Existing CI-equivalent commands from `.github/workflows/ci.yml` pass before modification.

**Do Not Break**

- Local CLI/Ollama self-host flow; existing fail-closed production behavior; honest model-status documentation.

### Phase 1 — Production configuration, model governance, and registry decision

**Objective**

Make production configuration unambiguous and eliminate the read-only registry/admin contradiction.

**Prerequisites**

- Phase 0 registry ownership and model-release decision.

**Files / Components**

- `src/gemma_cyber/inference/config.py`, `src/gemma_cyber/api/app.py`, `src/gemma_cyber/api/schemas.py`, `src/gemma_cyber/inference/registry.py`, `docker-compose.yml`, `Dockerfile`, `docs/deployment.md`, `docs/api.md`, tests.

**Tasks**

1. Implement strict startup validation for production configuration and a clear mode contract.
2. Restrict production generation to approved system-prompt/model behavior according to the approved policy; retain compatible local expert behavior only when explicit.
3. Choose one registry path:
   - GitOps/read-only: disable/omit hosted runtime registry mutation routes and preserve a read-only mount; document reviewed promotion/deploy workflow; or
   - Mutable: add atomic, locked durable persistence, subject attribution, backup/restore, and a writable deployment location; prohibit unsynchronized CLI writers.
4. Set explicit production model selection only after recorded passing evaluation/provenance exists.
5. Add direct tests for config failure, override policy, registry mode, and safe failure.

**Acceptance Criteria**

- Production starts only with complete reviewed settings; registry operations cannot silently fail due to a read-only mount; serving model/prompt policy is enforceable server-side.

**Validation**

- Existing test suite plus new focused tests; `docker compose config` after Compose changes; fail-closed startup test in an API image/container test.

**Do Not Break**

- Existing `admin:models` server-side protection if mutable admin routes remain; model promotion evaluation gate; local registry test behavior unless deliberately migrated with documented compatibility.

### Phase 2 — Auth0 browser integration and end-to-end authorization

**Objective**

Make the shipped web app work securely in JWT production mode.

**Prerequisites**

- Phase 1 endpoint/auth policy; Auth0 staging SPA/API configuration completed by an administrator.

**Files / Components**

- `src/gemma_cyber/api/web/index.html`, `src/gemma_cyber/api/web/app.js`, `src/gemma_cyber/api/app.py`, `src/gemma_cyber/api/auth.py`, `src/gemma_cyber/api/security.py`, `docs/auth.md`, browser/API tests.

**Tasks**

1. Implement Authorization Code + PKCE public-SPA flow and in-memory access-token use.
2. Add signed-out, callback, sign-in, sign-out, expiry, renewal-failure, and 401 UI states.
3. Attach bearer tokens only to same-origin protected calls. Add abort/cancel and robust SSE error handling.
4. Tighten CSP/cache headers/assets to support only the selected flow.
5. Extend JWT tests and add staging browser E2E coverage with normal and admin users.

**Acceptance Criteria**

- JWT-protected browser generation works end-to-end; no secret/static token/token persistence; authorization denials are correctly rendered; Auth0 dashboard configuration is documented by environment.

**Validation**

- `tests/test_api_auth.py`; selected browser runner tests; live staging smoke with a valid non-admin token; separate admin authorization check.

**Do Not Break**

- JWT issuer/audience/signature enforcement; no client-side role trust; same-origin CSP posture; open local development only when explicitly configured.

### Phase 3 — API reliability, capacity, and transport hardening

**Objective**

Protect a single-host inference API from predictable saturation and make streaming/deployment behavior reliable.

**Prerequisites**

- Phase 1 configuration bounds; target concurrency/capacity agreed.

**Files / Components**

- `src/gemma_cyber/api/app.py`, `src/gemma_cyber/api/security.py`, `src/gemma_cyber/inference/engine.py`, `src/gemma_cyber/clients/ollama_client.py`, `src/gemma_cyber/api/server.py`, proxy deployment artifact if authorized, tests/docs.

**Tasks**

1. Move blocking inference work off async route/event-loop execution or change the client path to async; retain all typed error mapping.
2. Add bounded active generation/queue admission, total deadline, cancellation/disconnect semantics, and clear 429/503 responses.
3. Add body-size, trusted-proxy, TLS/HSTS, edge rate-limit, cache/SSE buffering requirements to a versioned reference deployment or deployment contract.
4. Add readiness/drain lifecycle behavior and update smoke/runbook docs.

**Acceptance Criteria**

- Under controlled concurrent long requests, health/readiness remains responsive, work never exceeds configured bound, and shutdown does not accept new work indefinitely.

**Validation**

- Unit/integration concurrency tests; staging load/saturation test; external TLS/proxy header and SSE smoke test.

**Do Not Break**

- Request IDs, generation timeout/error semantics, model isolation (Ollama not publicly exposed), and SSE API contract unless documented versioning is added.

### Phase 4 — Cybersecurity UI system and frontend quality

**Objective**

Deliver the polished dark interface described in §9 without creating a frontend rewrite.

**Prerequisites**

- Phase 2 session states and stable API error taxonomy.

**Files / Components**

- `src/gemma_cyber/api/web/index.html`, `src/gemma_cyber/api/web/app.js`, new first-party CSS/JS assets if introduced, `api/app.py`, browser tests.

**Tasks**

1. Introduce semantic design tokens and responsive app-shell layout.
2. Modernize conversation, composer, status, auth, empty/loading/error states, preserving text-only output rendering.
3. Implement keyboard/focus/reduced-motion/accessibility requirements and response cancellation/copy affordances.
4. Add visual/browser/a11y regression tests and manual breakpoint review.

**Acceptance Criteria**

- Dark-first interface is coherent, high contrast, responsive, keyboard-accessible, and produces no unsupported product claims or fake dashboard data.

**Validation**

- Browser tests at agreed viewports; automated accessibility scan plus manual keyboard/screen-reader smoke; CSP/XSS regression check.

**Do Not Break**

- Same-origin API usage, no third-party tracking, CSP restrictions, SSE behavior, visible safety/privacy guidance.

### Phase 5 — CLI, observability, and operations

**Objective**

Make local CLI behavior/configuration and production diagnostics release-grade.

**Prerequisites**

- Phase 1 registry policy; Phase 3 operational signal design.

**Files / Components**

- `src/gemma_cyber/cli/main.py`, `src/gemma_cyber/inference/config.py`, `src/gemma_cyber/api/server.py`, `src/gemma_cyber/api/app.py`, `docs/cli.md`, `docs/operations.md`, `scripts/smoke_test.py`, tests.

**Tasks**

1. Make CLI/env validation and logging behavior match docs while preserving exit codes and streams.
2. Add structured/redacted API logs and minimal metrics/platform hooks; apply configured API log level.
3. Expand smoke/runbook/rollback instructions for the final endpoint and registry policy.
4. Perform staging rollback and registry restoration drill.

**Acceptance Criteria**

- Operators can correlate a failed request safely, detect saturation/readiness failure, run the documented smoke test, and roll back code/model/registry under the selected architecture.

**Validation**

- Existing CLI/API tests; log redaction tests; staging smoke and rollback drill recorded in release evidence.

**Do Not Break**

- Stable CLI exit codes and JSON shape; no prompt/response/token logging by default; operational health endpoint semantics.

### Phase 6 — Deterministic CI/CD and release verification

**Objective**

Ship reproducible, scanned artifacts and establish a meaningful go/no-go gate.

**Prerequisites**

- Final container/deployment approach; an approved artifact registry/release process.

**Files / Components**

- `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`, `Dockerfile`, `docker-compose.yml`, release docs, potentially new workflow files.

**Tasks**

1. Make lockfile installation authoritative in CI/image build; verify lock freshness policy.
2. Build/test wheel/sdist and container image; pin base/runtime images; scan image.
3. Add release provenance/SBOM/attestation appropriate to selected tooling and protect release controls externally.
4. Execute the full §12 matrix in staging, then production release checklist.

**Acceptance Criteria**

- A release identifies exact source, dependency lock, package/image digest, model provenance, scan results, staging evidence, and rollback target.

**Validation**

- CI required checks, built-artifact installation test, image scan, Docker configuration check, staging Auth0/external smoke, and signed-off checklist.

**Do Not Break**

- Existing security checks; Python support range; non-root container; no secret values in source/build logs/artifacts.

## 14. Production Release Checklist

- [ ] A named model is evaluated, approved, reproducibly identified, and selected for serving; no unproven specialty-model claim is made.
- [ ] Production config passes strict validation; `GEMMA_CYBER_ENV=prod` cannot use an unintended open/static-browser mode.
- [ ] Auth0 production API/SPA settings use exact HTTPS origins, PKCE, RS256, RBAC, least privilege, and an admin role restricted to authorized people.
- [ ] Browser sign-in/generate/logout/expiry and normal-vs-admin authorization pass against the production-like tenant.
- [ ] Registry deployment behavior is intentional: GitOps/read-only or durable mutable storage, with backup/rollback test completed.
- [ ] Ollama is private; API is only behind tested TLS/proxy controls; CORS, HSTS, rate limits, body limits, and forwarded-client policy are verified externally.
- [ ] Capacity limit, deadline, cancellation, health/readiness, graceful shutdown, and saturation behavior pass staging tests.
- [ ] Dark UI passes agreed keyboard, responsive, reduced-motion, contrast, and output-XSS/CSP checks.
- [ ] Locked dependency installation, wheel/image build, static/dependency/secret/image scans, and artifact provenance all pass.
- [ ] Logs/metrics/alerts/runbook/owners are in place; secrets are in deployment-managed storage and rotation procedure is tested.
- [ ] `scripts/smoke_test.py --base-url <production-url> --token "$TOKEN" --expect-auth` passes after deploy; rollback target and drill are verified.

## 15. Deferred / Post-Launch Improvements

- Remote hosted-API CLI mode with OAuth/device flow (only if product demand confirms it).
- Multi-instance deployment and shared distributed rate limiting/queue/session storage; current design documents single-host deployment.
- Prometheus/OpenTelemetry and full tracing after an observability backend/ownership model is selected.
- Persisted conversations, user accounts, teams, organizations, billing, and audit export. None exists now; each introduces retention/privacy/authorization design work.
- RAG, tools, autonomous agents, external URL fetch, or target interaction. `docs/security.md` correctly treats these as a new security boundary; do not add them as UI/product polish.
- Optional light theme and richer markdown/code rendering, only after the security/accessibility treatment is designed and tested.
- Kubernetes/autoscaling/GPU serving: no repository evidence requires this before a measured single-host capacity limit is reached.

## 16. Open Questions

1. Is the release a hosted public web product, a self-host distribution, or both? This determines whether static-token production mode and local direct-Ollama CLI are supported production paths.
2. Who owns model lifecycle changes in production: reviewed GitOps deployment changes, or authenticated runtime admins? If runtime mutation is required, what durable store/backup/audit retention is approved?
3. What exact production/staging domains, Auth0 tenants/API identifiers/SPA client IDs, scopes, session lifetimes, and allowed origins are approved?
4. Must `/health`, `/v1/ready`, `/v1/models`, OpenAPI docs, and the UI be public, edge-restricted, or authenticated in production?
5. What model artifact has passed the release gate, where is its immutable artifact/digest, and what measurable acceptance threshold authorizes promotion to `production`?
6. What peak concurrency, latency/SLO, request-size/token quota, and budget determine the admission-control and worker configuration?
7. Which reverse proxy, hosting platform, secret manager, image registry, logging/metrics backend, backup system, and release approver will operate the service?
8. Are any regulated/privacy, retention, security incident-response, or customer audit requirements applicable to hosted prompts/responses and operational logs?
