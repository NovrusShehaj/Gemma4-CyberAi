# Gemma4-CyberAI — Production-Readiness Plan

> **Authoritative execution roadmap** for taking the project from **ORANGE** toward a
> defensible production-ready state. Owner: engineering. Created 2026-08-26.
> Companion to `PROJECT_PLAN.md` (strategy) and the phase docs in `docs/`.
> Status legend: `TODO` · `IN-PROGRESS` · `DONE` · `BLOCKED(reason)` · `DEFERRED`.

Update this file as work lands. Keep status synchronized with the repository, not aspiration.

---

## 1. Executive summary

The previous session built the product surfaces (shared inference engine, CLI, HTTP
API, web UI, model registry, Docker/compose, docs) — all green (156 tests, ruff +
mypy). The remaining production gap is **not** primarily model quality. It is:
**(P1a) real identity & authorization (Auth0/JWT), (P1b) production-safe API execution
with server-side authorization on privileged operations, (P1c) deployment hardening &
prod fail-closed config, (P1d) automated security scanning in CI**, plus observability
and operational validation. Model training remains externally blocked on a GPU and is
tracked but not a launch blocker for the *application*.

This plan implements everything achievable locally: full JWT verification
(signature via JWKS + rotation, issuer, audience, expiry, claims), server-side
scope/permission enforcement on privileged endpoints, prod fail-closed startup,
request-timing/auth-failure observability, a security-scanning CI stage, and an
executable operational smoke-test suite. Auth0 **dashboard** configuration that cannot
be done from the repo is documented precisely and everything else is validated with an
in-test RS256 keypair + mocked JWKS.

## 2. Current production-readiness assessment

**ORANGE.** Product surfaces work and are tested; the trust/identity boundary and
production operational controls are the blockers.

| Dimension | Before | Target this pass |
|---|---|---|
| Architecture | coherent (shared engine) but no authz layer | add auth/authz layer without new coupling |
| Authentication | static bearer only | Auth0 JWT (RS256, JWKS, iss/aud/exp/claims) |
| Authorization | none (all-or-nothing token) | server-side scopes; privileged endpoints gated |
| API | validated, bounded, rate-limited | + authz, prod fail-closed, timing/observability |
| Web | functional | documented prod config; unchanged security posture |
| CLI | functional | unchanged (already exit-coded, no secrets) |
| Security scanning | none in CI | pip-audit + bandit + secret scan + ruff-sec in CI |
| Observability | request ids, health, logs | + latency, auth/authz failure events |
| Operations | docs only | executable smoke-test suite |

## 3. Verified architecture (evidence-based)

```
WEB (api/web) ─┐
CLI (cli) ─────┼─▶ API (api/app) ─▶ AUTH (api/auth: JWT/JWKS) ─▶ AUTHZ (scopes)
               │                                     │
Eval harness ──┴────────────────────────────────────┴─▶ InferenceEngine (inference/engine)
                                                             │
                                                        OllamaClient ─▶ Ollama runtime
                        ModelRegistry (inference/registry) ◀─ admin endpoints + CLI
```
- Single shared inference path (`gemma_cyber.inference.InferenceEngine`) already used by
  CLI, API, and eval — **preserve; do not duplicate**.
- Model runtime (Ollama) is never published publicly (compose `expose`, api binds
  localhost). **Preserve.**
- **Gap being closed:** no server-side identity/authorization gate in front of privileged
  operations (registry mutation, eval, admin). Add `api/auth.py` between API and services.

## 4. Current implementation status (repo facts)

- `src/gemma_cyber/inference/*` — engine, config, registry, errors. Tested.
- `src/gemma_cyber/api/{app,schemas,security,server}.py` + `web/` — FastAPI + UI. Tested.
  Auth = static bearer only (`security.token_matches`).
- `src/gemma_cyber/cli/main.py` — full CLI. Tested.
- Training scaffold + eval harness + benchmarks v1/v2/v3 + registry seed. Tested.
- CI: ruff + mypy + pytest. **No security scanning.**
- 156 tests green; ruff + mypy clean.

## 5. P1 blockers (production blockers)

### P1a — Auth0 identity (authentication)
- **Objective:** verify caller identity with production-grade JWT validation.
- **Current state:** static shared bearer token only; no signature/issuer/audience/expiry.
- **Desired state:** RS256 JWT verified against Auth0 JWKS (with key rotation), issuer +
  audience + expiry + required-claim validation; misconfig fails closed in prod.
- **Dependencies:** `pyjwt[crypto]` (added to `api`/`dev` extras).
- **Approach:** new `api/auth.py` (`AuthSettings`, `JwksCache`, `verify_token`, `Principal`),
  wired as a FastAPI dependency; keep static-token mode for dev only.
- **Files:** `api/auth.py` (new), `api/app.py`, `inference/config.py`, `pyproject.toml`.
- **Tests:** in-test RS256 keypair + mocked JWKS: valid / missing / malformed / expired /
  wrong iss / wrong aud / bad signature.
- **Security implications:** THE trust boundary. No tokens logged. Fail closed.
- **Acceptance:** all negative cases → 401; valid → 200; prod without config → refuses start.
- **Validation:** `tests/test_api_auth.py`; live tenant documented as external step.
- **Status:** DONE (local; live Auth0 tenant config documented in docs/auth.md)

### P1b — Server-side authorization on privileged operations
- **Objective:** privileged actions (model promote/register/mark-evaluated, eval) require
  a permission in the signed token — never a client-supplied role.
- **Current state:** registry mutation is CLI-only/local; no API authz.
- **Desired state:** `require_scopes()` dependency; admin endpoints gated on `admin:models`;
  generate gated on authentication (or `chat:write`) when auth enabled.
- **Approach:** scope/permission extraction from `scope` + `permissions` claims; admin
  router.
- **Files:** `api/auth.py`, `api/app.py`, `tests/test_api_auth.py`.
- **Tests:** insufficient permission → 403; sufficient → 200.
- **Acceptance:** privileged endpoints reject authenticated-but-unauthorized callers (403).
- **Status:** DONE (local; live Auth0 tenant config documented in docs/auth.md)

### P1c — Deployment hardening / prod fail-closed
- **Objective:** production config cannot silently run insecure.
- **Current state:** Docker/compose exist; app runs open if unconfigured.
- **Desired state:** `GEMMA_CYBER_ENV=prod` requires auth configured (else refuse start);
  security headers/CORS already present; document prod env matrix.
- **Files:** `api/app.py`, `docs/deployment.md`, `docs/security.md`.
- **Tests:** `create_app` in prod w/o auth raises; with auth ok.
- **Acceptance:** prod misconfig is a hard startup failure, tested.
- **Status:** DONE (local; live Auth0 tenant config documented in docs/auth.md)

### P1d — Security scanning in CI
- **Objective:** automated dependency/secret/static security checks gate merges.
- **Current state:** none.
- **Desired state:** CI runs `pip-audit`, `bandit`, a secret scan (gitleaks), and ruff
  security rules; findings actionable; baseline documented.
- **Files:** `.github/workflows/ci.yml`, `pyproject.toml` (ruff `S`), `docs/security.md`.
- **Acceptance:** CI job runs the scans; documented baseline + justified exceptions.
- **Status:** DONE (CI `security` job: bandit + pip-audit + gitleaks; baseline in docs/security.md)

## 6. P2 items (important hardening)

- **P2a Observability:** request latency + auth/authz-failure structured events; document. `TODO`
- **P2b Operational smoke tests:** executable suite (install/startup/health/authn/authz/
  failure modes) runnable locally + in CI without real credentials. `TODO`
- **P2c CLI auth:** allow the CLI to send a bearer token to a protected API (`--token`/env).
  `TODO`
- **P2d Rollback drill test:** registry rollback covered by a test. `TODO`
- **P2e Admin API for model lifecycle:** expose registry mutation via authz'd endpoints
  (also satisfies P1b surface). `TODO`

## 7. P3 improvements

P3a shared multi-instance rate limiter · P3b `==`-locked training deps from a green run ·
P3c metrics export (Prometheus) · P3d frontend dep audit (no build system yet, minimal) ·
P3e request tracing/correlation IDs across proxy · P3f graceful-shutdown lifespan hook ·
P3g structured JSON log formatter option · P3h OpenTelemetry hook · P3i per-endpoint
concurrency limits. All `DEFERRED` unless a P1/P2 needs them.

## 8. AI / model improvements

- **AIa** Execute exp-002 training + evaluation (GPU). `BLOCKED(external GPU)`.
- **AIb** Re-run judge-v2 calibration. `BLOCKED(needs Ollama + a trained model)`.
- **AIc** Model lifecycle states already in the registry (experimental→evaluated→
  candidate→production→archived). Wire admin API (P2e). `IN-PROGRESS via P1b`.

## 9. Dependency graph

```
P1a Auth ──▶ P1b Authz ──▶ P2e Admin API ──▶ P2c CLI auth
   │             │
   └──▶ P1c prod fail-closed
P1d Security CI  (independent)
P2a Observability (independent; small)
P2b Operational smoke tests ──▶ depends on P1a/P1b to test authn/authz
AIa/AIb  BLOCKED(external)
```

## 10. Implementation phases (this pass, ordered)

1. **P1a+P1b Auth+Authz** (crown jewel) → tests → commit.
2. **P1c prod fail-closed** (folded into the auth commit or its own).
3. **P2e Admin model-lifecycle API** (authz'd) → tests → commit.
4. **P2a Observability** (latency + auth events) → tests → commit.
5. **P1d Security CI + ruff `S` + baseline** → run locally → commit.
6. **P2b Operational smoke tests** → run → commit.
7. **P2c CLI `--token`** → tests → commit.
8. **Docs (auth/deployment/security/operations) + plan sync** → commit.
9. **Go/No-Go reassessment.**

## 11. Acceptance criteria — see each item above. Global: ruff + mypy clean, all tests green, no secrets committed, CI updated.

## 12. Testing strategy
Unit + endpoint tests via FastAPI TestClient with a **self-signed RS256 keypair and mocked
JWKS** (no live Auth0 needed). Positive + negative auth cases mandatory. Operational
smoke tests exercised against the app in-process; a scripted procedure documents the
live-infra steps.

## 13. Security validation strategy
`pip-audit` (deps), `bandit` (Python static), `gitleaks` (secrets), ruff `S` rules. No
finding suppressed to go green; exceptions documented with justification in
`docs/security.md`. Auth negative tests are the primary control validation.

## 14. Deployment validation strategy
`docker compose config` (done previously) + documented `compose up` procedure; prod
fail-closed test; env matrix (dev/staging/prod) documented; health/readiness probes.

## 15. Observability requirements
Structured logs with request id, model tag, latency, auth/authz failure events; no
tokens/secrets/PII logged; `/health` + `/v1/ready`. Documented in `docs/operations.md`.

## 16. Operational-readiness requirements
Executable smoke tests (startup/health/authn/authz/failure), runbook, rollback drill.

## 17. Rollback requirements
Model rollback = registry promotion (tested). Code rollback = image tag redeploy
(documented). Config rollback = env change (documented).

## 18. Documentation requirements
Update `docs/security.md`, `docs/deployment.md`, `docs/operations.md`, `docs/api.md`,
add `docs/auth.md`; keep this plan synced.

## 19. Production go/no-go criteria
GREEN requires: Auth0 authn+authz enforced server-side and tested (✔ local, tenant config
documented); prod fail-closed; security scanning in CI with a reviewed baseline;
observability for failures; operational smoke tests passing; a real evaluated model
promoted to `production`. The **model** gate (AIa) is external and keeps the *product*
at ORANGE→"conditionally GREEN pending a promoted model + a configured tenant".

## 20. Risks & mitigations
- *Auth misconfig in prod* → fail-closed startup + tests. 
- *JWKS/network fragility* → cache + timeout + rotation refresh + clear errors.
- *Scanner noise* → curated tools, documented baseline, no blanket suppression.
- *Model unproven* → registry keeps it experimental; no false promotion.

## 21. Deferred work
Multi-instance rate limiting, OpenTelemetry, Prometheus export, managed TLS config,
frontend build pipeline, billing/usage admin.

## 22. Explicitly out of scope (this pass)
RAG, autonomous agents/tools, Kubernetes, microservices, GPU serving, a user/account
database, live Auth0 tenant provisioning, real cloud deployment execution.
