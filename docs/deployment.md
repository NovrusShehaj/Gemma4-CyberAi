# Deployment Guide

The target is a **single-host** deployment: one small API container plus an
Ollama model-runtime container. This is deliberately boring — no Kubernetes, no
message queue, no microservices. The architecture scales to a bigger box before
it needs anything more complex.

```
            HTTPS                     compose network
  client ─────────▶ reverse proxy ─────────▶ api ─────────▶ ollama ──▶ gemma3:4b
        (TLS, rate)     (nginx/caddy)     (FastAPI)     (model runtime)
```

## 1. Local / single host with Docker Compose

```bash
docker compose up -d --build
docker compose exec ollama ollama pull gemma3:4b     # one-time
curl -s localhost:8000/v1/ready                      # {"ok": true, ...}
open http://localhost:8000/
```

The `api` port is published on `127.0.0.1` only; Ollama is not published at all
(reachable only over the compose network). The registry is mounted **read-only**
from `./data/models` (GitOps; see §5). The image installs the **locked** dependency
graph (`uv sync --locked` from the committed `uv.lock`), so the container gets the
exact reviewed deps CI tested — not a fresh pip resolution.

Checked-in Compose uses `GEMMA_CYBER_ENV=staging` so a laptop eval can start
without Auth0. That profile is **not** production.

## 1b. Production overlay (required before public exposure)

```bash
# Fill deploy/prod.env.example, chmod 0600, never commit the filled file.
set -a && source /path/to/prod.env && set +a
docker compose -f docker-compose.yml -f docker-compose.prod.yml config   # must succeed
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

The overlay sets `GEMMA_CYBER_ENV=prod`, requires
`GEMMA_CYBER_AUTH_DOMAIN` / `AUDIENCE` / `WEB_AUTH0_CLIENT_ID` (Compose
`${VAR:?}` fails interpolation if unset), and turns the generate rate limiter
on. The API still binds `127.0.0.1:8000`; put nginx/Caddy in front. TLS
placeholders: [`deploy/TLS-CHECKLIST.md`](../deploy/TLS-CHECKLIST.md).

## 2. Public exposure (add before opening to the internet)

Put a TLS-terminating reverse proxy (Caddy/nginx/Traefik) in front of `api`, and
enable the built-in controls via environment:

```bash
# Auth0 JWT (recommended for real identity + authorization) — see docs/auth.md:
export GEMMA_CYBER_AUTH_DOMAIN="your-tenant.auth0.com"
export GEMMA_CYBER_AUTH_AUDIENCE="https://api.gemma-cyber"
# ...or a static dev-grade token (no per-user identity, cannot administer models):
# export GEMMA_CYBER_API_TOKEN="$(openssl rand -hex 32)"
export GEMMA_CYBER_WEB_AUTH0_CLIENT_ID="<spa-client-id>"  # public SPA id for the web login
export GEMMA_CYBER_ENV=prod                               # fail closed if no auth set
export GEMMA_CYBER_RATE_LIMIT_PER_MIN=60                  # per-identity cap
export GEMMA_CYBER_CORS_ORIGINS="https://app.example.com" # exact origins (no '*' in hosted)
export GEMMA_CYBER_MAX_CONCURRENT_GENERATIONS=4           # bound in-flight generations
export GEMMA_CYBER_MODEL=gemma3:4b                        # default serve; no production alias
```

With `GEMMA_CYBER_ENV=prod` the service **refuses to start** unless authentication
is configured, and `Settings.validate()` rejects structurally-invalid config
(bad URLs, negative bounds, a `'*'` CORS origin in hosted mode) at startup — an
accidental open or misconfigured public API is impossible.

### Configuration contract (validated at startup)

| Variable | Hosted default | Notes |
|---|---|---|
| `GEMMA_CYBER_ENV` | — | one of `dev`/`test`/`staging`/`prod`; `staging`+`prod` are *hosted* (strict) |
| `GEMMA_CYBER_ALLOW_CLIENT_OVERRIDES` | `false` | server owns system prompt + servable models |
| `GEMMA_CYBER_REGISTRY_WRITABLE` | `false` | GitOps read-only registry (see §5) |
| `GEMMA_CYBER_MAX_CONCURRENT_GENERATIONS` | `4` | 0 = unbounded; saturation → 503 `at_capacity` |
| `GEMMA_CYBER_REQUEST_DEADLINE_S` | `0` | total budget across retries; 0 = per-attempt timeout only |
| `GEMMA_CYBER_CORS_ORIGINS` | *(empty)* | exact `https://` origins; `'*'` rejected in hosted mode |
| `GEMMA_CYBER_RATE_LIMIT_PER_MIN` | `60` hosted / `0` dev | hosted `0` or garbage env → startup `ConfigError`; 429 sends `Retry-After` |
| `GEMMA_CYBER_LOG_LEVEL` | `INFO` | applied to structured JSON logs |

### Edge / reverse proxy

A versioned reference config ships at [`deploy/nginx.reference.conf`](../deploy/nginx.reference.conf).
Placeholder inventory: [`deploy/TLS-CHECKLIST.md`](../deploy/TLS-CHECKLIST.md).
It documents the edge contract — HTTPS+HSTS, modern TLS, `client_max_body_size`,
pre-auth IP rate limiting, trusted `X-Forwarded-For`, SSE `proxy_buffering off`,
bounded timeouts, and keeping Ollama private. Hostname/cert paths are operator
values (marked `<-- set`). HSTS is set **at the edge only**, never by the app.

The in-process limiter does **not** cap unauthenticated 401 storms; the proxy
`limit_req` must.

Checklist before public exposure:
- [ ] HTTPS only (proxy redirects 80→443, HSTS).
- [ ] Auth0 JWT configured (`GEMMA_CYBER_AUTH_DOMAIN` + `GEMMA_CYBER_AUTH_AUDIENCE`),
      or at minimum a static token; `admin:models` permission assigned to admins only.
- [ ] Rate limit set (the in-process limiter is single-instance; use the proxy's
      limiter if you run more than one `api` replica).
- [ ] Ollama never published to the host/internet.
- [ ] Secrets from the environment / a secrets manager — never baked into the image.
- [ ] Run `security-review` / dependency scan (`pip-audit`) in CI before release.

## 3. Choosing the model to serve

`GEMMA_CYBER_MODEL` accepts a tag or a registry alias:
```bash
GEMMA_CYBER_MODEL=gemma3:4b    # default; APP-GATE waiver, see docs/model-card.md
# GEMMA_CYBER_MODEL=production  # only after MODEL-GATE; no such entry exists today
```
Only a version that has cleared its evaluation gate can be promoted to
`production` (see the registry lifecycle in `docs/cli.md`).
`gemma3-cyber:v0.2` **failed** that gate and must not be promoted.

## 4. Health, startup, shutdown

- **Liveness:** `GET /health` (cheap; process up). Wired into the Docker
  `HEALTHCHECK` and suitable for a proxy/orchestrator liveness probe.
- **Readiness:** `GET /v1/ready` (503 until Ollama is reachable and the model is
  pulled). Cached in-process (~10s) so public probes do not hammer Ollama; the
  JSON does **not** include the runtime URL. Orchestrators should probe `/health`
  on the published port and `/v1/ready` on loopback when possible.
- **Startup:** compose waits for Ollama's healthcheck before starting `api`.
- **Shutdown:** uvicorn handles SIGTERM; `restart: unless-stopped` recovers crashes.

## 5. Registry ownership (GitOps read-only, default) & rollback

Hosted mode defaults to a **read-only, GitOps-managed registry**
(`GEMMA_CYBER_REGISTRY_WRITABLE=false`): the runtime admin-mutation routes are
disabled (they return 503) and the compose mount stays `:ro`. Model lifecycle
changes are made by **reviewed, source-controlled edits** to
`data/models/registry.json` and redeployed — the audit trail lives in git, and a
CLI and the API can never race to corrupt a host-mounted file.

Writes at the persistence layer are still hardened for the self-host case
(`GEMMA_CYBER_REGISTRY_WRITABLE=true` with a durable writable volume): atomic
temp-file + `os.replace` (crash-safe), owner-only `0600` permissions, and every
transition records the authenticated **subject** + reason in `history`.

Model rollback is a registry operation, not a redeploy:
```bash
# GitOps: revert the registry.json change (git revert) and redeploy, OR self-host:
gemma-cyber models promote <previous-version> --to production
```
Code rollback: redeploy the previous image tag. Keep image tags immutable
(`:<git-sha>`), never overwrite `:latest` in production. Pin the Ollama and Python
base images (`OLLAMA_IMAGE`, and the Dockerfile `FROM ... @sha256:` digest) to
reviewed versions for a reproducible release.

## 6. Backups and restore drill

See `docs/operations.md` (DR) and `python scripts/restore_drill.py`.

- **Registry** (`data/models/registry.json`): in git + the mounted volume. Copy
  the file (`scripts/restore_drill.py backup`) and confirm SHA after restore.
- **Ollama weights:** Compose volume `ollama-models`. Archive with
  `scripts/restore_drill.py volume-commands`; restore onto a clean project;
  `GET /v1/models` must match the backup listing.
- No user database exists (the service is stateless per request).
- RPO = last successful copy. RTO = “restore files/volume and restart compose”
  (hours-scale, host-dependent). No offsite SLA is claimed.

## 7. What is NOT included yet (be honest)

- Ships a **reference** nginx edge config (`deploy/nginx.reference.conf`), not a
  turnkey managed-TLS deployment — hostname, certs, and trusted-proxy CIDRs are
  operator-supplied and must be validated in staging against the real URL
  (`deploy/TLS-CHECKLIST.md`).
- No multi-instance shared rate limiter or session store (single-instance limiter).
- No GPU serving path for a larger model (gemma3:4b runs CPU-fine via Ollama).
- No autoscaling — single host by design at this stage.
- Base/Ollama image **digest** pinning is left to the release owner (see §5); the
  build is dependency-reproducible via `uv.lock` but the base image tag is a moving
  target until pinned.
- **CI does not run Auth0 browser E2E or live TLS smoke.** The `container`/`package`
  jobs build and import the image. JWT negatives run in-process
  (`tests/test_api_auth.py`). Live `--expect-auth` against HTTPS is an **operator**
  step (`scripts/smoke_test.py`); see `docs/auth.md` AUTH-001.
