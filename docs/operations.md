# Operations & Observability

Privacy-conscious telemetry for a small-team/solo-maintained service.

## Structured logging

The API service emits **structured JSON logs** (`gemma_cyber.api.logging_setup`),
one object per line, at `GEMMA_CYBER_LOG_LEVEL` (applied at startup by
`gemma-cyber-serve`). Uvicorn's access/error logs flow through the same JSON
formatter. The formatter emits only an **allowlist** of fields, so a stray prompt
or token attached to a log record is dropped rather than leaked. Each generate/
stream call logs a **request id**, **model tag**, attempt number, input size, and
latency; retries and errors log the error type. `--debug` (CLI) raises CLI verbosity.

Example line:
```json
{"ts":"2026-08-27T01:11:42-0400","level":"INFO","logger":"gemma_cyber.api","msg":"request","request_id":"9a10e7efc4b7","method":"POST","path":"/v1/generate","status":200,"latency_ms":812.4}
```

### What is logged
- request id (`X-Request-ID`), timestamp, level
- model/version tag served
- inference attempt count, retry/backoff events
- error **type** and message (structured; no stack traces to clients)
- request latency (add at the proxy or extend the middleware)

### What is NOT logged (by default)
- prompt or response **content**
- credentials or bearer tokens (auth compares in constant time; tokens never logged)
- user identity / PII (there is no account store yet)

If you later need content logging for debugging, make it opt-in, time-boxed, and
documented — and keep it out of production defaults.

## Metrics to watch (wire to your platform)

The building blocks exist (request ids, health, per-call logs). For a real
deployment, export from the proxy or extend the request middleware:
- request count + error rate (by status code; watch `at_capacity` 503s)
- p50/p95 inference latency (`latency_ms` in the JSON access log)
- active generations vs `MAX_CONCURRENT_GENERATIONS` (saturation signal)
- readiness flaps (`/v1/ready` 503s) and Auth0 JWKS 503s
- model-version usage (from the served tag)
- host CPU/memory (Ollama is the heavy consumer)

Health endpoints for probes: `/health` (liveness), `/v1/ready` (readiness).

## Retention

No content is stored, so there is nothing to retain beyond operational logs. Keep
operational logs per your platform's default (e.g. 14–30 days) and the registry
audit trail in git indefinitely.

## Runbook — common failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| `/v1/ready` 503, `service_reachable=false` | Ollama down/unreachable | `docker compose ps`; restart `ollama`; check `GEMMA_CYBER_OLLAMA_HOST` |
| `/v1/ready` 503, `model_present=false` | model not pulled | `docker compose exec ollama ollama pull gemma3:4b` |
| 504 timeout on generate | slow model / cold start / too-large `num_predict` | raise `GEMMA_CYBER_TIMEOUT`; check host load; lower `num_predict` |
| 429 | rate limit hit | expected under load; raise `RATE_LIMIT_PER_MIN` or scale |
| 503 `at_capacity` | at the concurrent-generation bound | expected backpressure; raise `MAX_CONCURRENT_GENERATIONS` if the host has headroom, or scale |
| 401 (all requests) | JWT on but browser cannot sign in | set `GEMMA_CYBER_WEB_AUTH0_CLIENT_ID` (public SPA id); check Auth0 callback/origin URLs |
| 401 (intermittent) | access token expired | UI returns to signed-out; user signs in again — no action |
| 503 `authentication provider unavailable` | Auth0 **JWKS outage** | check Auth0 status; tokens cannot be verified — this is infra, not an auth decision; alert + wait/rotate |
| 403 on admin routes | valid token lacks `admin:models` | assign the permission/role in Auth0 (least privilege) |
| 503 on admin routes | registry is **read-only** (GitOps) | expected in hosted mode; change `registry.json` in source control and redeploy |
| Web page loads but chat errors | API/Ollama issue | check `X-Request-ID` in the failing response against server logs |
| Startup crash, non-zero exit | invalid/incomplete config | read the `ConfigError`/fail-closed message; fix env (auth, URLs, bounds) |

## Incident response (lightweight)

1. **Detect** — readiness probe failing or error-rate spike.
2. **Contain** — if a bad model version is serving, roll back:
   `gemma-cyber models promote <previous> --to production` and restart `api`.
   If the service is compromised or abused, revoke `GEMMA_CYBER_API_TOKEN` and
   redeploy.
3. **Diagnose** — correlate `X-Request-ID` across proxy/api logs.
4. **Recover** — restart the affected container; confirm `/v1/ready`.
5. **Record** — note cause + fix in `docs/decisions.md`.

## Operational smoke test

`scripts/smoke_test.py` validates a running deployment end to end: liveness,
readiness, security headers, model listing, input validation, the generate path,
and — with `--expect-auth` — that auth is enforced.

```bash
python scripts/smoke_test.py --base-url http://localhost:8000
python scripts/smoke_test.py --base-url https://api.example.com --token "$TOKEN" --expect-auth
```
Exit 0 = all required checks passed. Run it after every deploy and after a
rollback. The same checks run in CI in-process (`tests/test_operational_smoke.py`,
open + auth-enforced modes) so regressions are caught without a live server.

## Rollback drill

Rollback is a registry promotion (model) or an image-tag redeploy (code). Test it
in staging before you need it in production:
```bash
gemma-cyber models promote gemma3:4b --to production   # if a prior prod exists
docker compose up -d api                                # picks up GEMMA_CYBER_MODEL=production
```
