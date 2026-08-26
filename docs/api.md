# Gemma-Cyber HTTP API

A thin, versioned HTTP surface over the shared inference engine
(`gemma_cyber.inference`). It holds no model logic — the CLI, the web UI, and the
evaluation harness all use the same engine, so behavior is consistent everywhere.

## Run

```bash
uv pip install -e '.[api]'
gemma-cyber-serve                 # binds 127.0.0.1:8000 by default
# or: PYTHONPATH=src python -m gemma_cyber.api.server
```

Open <http://127.0.0.1:8000/> for the web chat UI. Interactive OpenAPI docs are at
`/docs`.

## Configuration (environment variables)

Inherits all `GEMMA_CYBER_*` inference settings (see `docs/cli.md`) plus:

| Variable | Default | Meaning |
|---|---|---|
| `GEMMA_CYBER_API_HOST` | `127.0.0.1` | Bind address (keep localhost behind a proxy) |
| `GEMMA_CYBER_API_PORT` | `8000` | Bind port |
| `GEMMA_CYBER_API_TOKEN` | *(empty)* | If set, `/v1/*` requires `Authorization: Bearer <token>` |
| `GEMMA_CYBER_RATE_LIMIT_PER_MIN` | `0` | Per-client generate cap; 0 disables |
| `GEMMA_CYBER_CORS_ORIGINS` | *(empty)* | Comma-separated CORS allowlist; empty = same-origin only |

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/` | no | Self-contained web chat UI |
| GET | `/health` | no | Liveness (process up; never touches the model) |
| GET | `/v1/ready` | no | Readiness (runtime reachable + model present); 503 when not ready |
| GET | `/v1/models` | no | Registry listing + current production version |
| POST | `/v1/generate` | authenticated | One completion, or SSE stream when `stream=true` |
| POST | `/v1/admin/models/register` | `admin:models` | Register a model version |
| POST | `/v1/admin/models/{version}/mark-evaluated` | `admin:models` | Record an eval outcome (gate input) |
| POST | `/v1/admin/models/{version}/promote` | `admin:models` | Promote a version (gated lifecycle) |

Authentication is Auth0 JWT (or a static dev token); authorization is enforced
server-side from the signed token. See **`docs/auth.md`** for modes, validation,
scopes, and the Auth0 dashboard setup.

### `POST /v1/generate`

Request:
```json
{
  "prompt": "What ATT&CK technique is Kerberoasting?",
  "model": null,
  "system": null,
  "temperature": null,
  "seed": null,
  "num_predict": null,
  "stream": false
}
```
`prompt` is required (1–24000 chars). `model` may be a tag or a registry alias
(`production`, a version); `null` uses the server default. Unset optional fields
fall back to the deterministic server defaults (temperature 0, seed 0).

Response (non-stream):
```json
{ "request_id": "9a10e7efc4b7", "model": "gemma3:4b", "response": "…" }
```

Streaming (`"stream": true`): `text/event-stream`, one JSON object per SSE `data:`
line — `{"text": "…"}` chunks, then `{"done": true, "request_id": "…"}`. Errors
mid-stream arrive as `{"error": "…"}`.

### Errors

Structured JSON: `{ "error": "<code>", "detail": "…", "request_id": "…" }`.

| Status | `error` | When |
|---|---|---|
| 401 | — | Missing/invalid bearer token (when auth enabled) |
| 422 | — | Request validation (empty/oversized prompt, bad field) |
| 429 | — | Rate limit exceeded |
| 503 | `service_unavailable` / `model_unavailable` | Ollama down / model not pulled |
| 504 | `timeout` | Generation timed out after retries |

Every response carries an `X-Request-ID` header for log correlation.

## Security posture (see `docs/security.md`)

- Security headers on every response (CSP, `X-Frame-Options: DENY`, nosniff, …).
- Bearer auth and rate limiting are built in, **off by default**, enabled by env.
- Binds to localhost by default; public exposure expects a TLS-terminating
  reverse proxy (see `docs/deployment.md`). The raw model runtime is never exposed.
- Input is validated and size-bounded before it reaches the model.
- Defensive scope only: no target interaction, no command execution, no tools.
