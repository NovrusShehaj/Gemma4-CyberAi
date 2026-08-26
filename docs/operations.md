# Operations & Observability

Privacy-conscious telemetry for a small-team/solo-maintained service.

## Structured logging

The inference layer and API log through the standard `logging` module under the
`gemma_cyber.*` loggers. Each generate/stream call logs a **request id**, the
**model tag**, attempt number, and input size; retries and errors log the error
type. `--debug` (CLI) or `GEMMA_CYBER_LOG_LEVEL=DEBUG` raises verbosity.

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
- request count + error rate (by status code)
- p50/p95 inference latency
- readiness flaps (`/v1/ready` 503s)
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
| 401 | missing/invalid token | supply `Authorization: Bearer <token>` |
| Web page loads but chat errors | API/Ollama issue | check `X-Request-ID` in the failing response against server logs |

## Incident response (lightweight)

1. **Detect** — readiness probe failing or error-rate spike.
2. **Contain** — if a bad model version is serving, roll back:
   `gemma-cyber models promote <previous> --to production` and restart `api`.
   If the service is compromised or abused, revoke `GEMMA_CYBER_API_TOKEN` and
   redeploy.
3. **Diagnose** — correlate `X-Request-ID` across proxy/api logs.
4. **Recover** — restart the affected container; confirm `/v1/ready`.
5. **Record** — note cause + fix in `docs/decisions.md`.

## Rollback drill

Rollback is a registry promotion (model) or an image-tag redeploy (code). Test it
in staging before you need it in production:
```bash
gemma-cyber models promote gemma3:4b --to production   # if a prior prod exists
docker compose up -d api                                # picks up GEMMA_CYBER_MODEL=production
```
