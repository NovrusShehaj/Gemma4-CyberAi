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
(reachable only over the compose network). The registry is mounted read-only from
`./data/models` so promotions are auditable on the host.

## 2. Public exposure (add before opening to the internet)

Put a TLS-terminating reverse proxy (Caddy/nginx/Traefik) in front of `api`, and
enable the built-in controls via environment:

```bash
export GEMMA_CYBER_API_TOKEN="$(openssl rand -hex 32)"   # require Bearer auth
export GEMMA_CYBER_RATE_LIMIT_PER_MIN=60                  # per-client cap
export GEMMA_CYBER_CORS_ORIGINS="https://app.example.com" # if a separate frontend
```

Checklist before public exposure:
- [ ] HTTPS only (proxy redirects 80→443, HSTS).
- [ ] `GEMMA_CYBER_API_TOKEN` set (or a real auth gateway in front).
- [ ] Rate limit set (the in-process limiter is single-instance; use the proxy's
      limiter if you run more than one `api` replica).
- [ ] Ollama never published to the host/internet.
- [ ] Secrets from the environment / a secrets manager — never baked into the image.
- [ ] Run `security-review` / dependency scan (`pip-audit`) in CI before release.

## 3. Choosing the model to serve

`GEMMA_CYBER_MODEL` accepts a tag or a registry alias:
```bash
GEMMA_CYBER_MODEL=production   # serve whatever is promoted to production
GEMMA_CYBER_MODEL=gemma3:4b    # pin a specific tag
```
Only a version that has cleared its evaluation gate can be promoted to
`production` (see the registry lifecycle in `docs/cli.md`).

## 4. Health, startup, shutdown

- **Liveness:** `GET /health` (cheap; process up). Wired into the Docker
  `HEALTHCHECK` and suitable for a proxy/orchestrator liveness probe.
- **Readiness:** `GET /v1/ready` (503 until Ollama is reachable and the model is
  pulled). Use it to gate traffic; on first boot it stays 503 until the one-time
  `ollama pull` completes.
- **Startup:** compose waits for Ollama's healthcheck before starting `api`.
- **Shutdown:** uvicorn handles SIGTERM; `restart: unless-stopped` recovers crashes.

## 5. Rollback

Model rollback is a registry operation, not a redeploy:
```bash
gemma-cyber models promote <previous-version> --to production
```
Code rollback: redeploy the previous image tag. Keep image tags immutable
(`:<git-sha>`), never overwrite `:latest` in production.

## 6. Backups

- **Registry** (`data/models/registry.json`): in git + the mounted volume; it is
  the source of truth for what is promoted. Back up with the repo.
- **Model weights / GGUF:** large, reproducible from a training run's manifest —
  back up the manifest (Drive/HF Hub per `docs/training/README.md`), not
  necessarily the weights.
- No user database exists yet (the service is stateless per request). If
  conversation persistence or accounts are added later, that store needs its own
  backup + retention policy.

## 7. What is NOT included yet (be honest)

- No managed TLS/proxy config shipped (bring your own Caddy/nginx).
- No multi-instance shared rate limiter or session store.
- No GPU serving path for a larger model (gemma3:4b runs CPU-fine via Ollama).
- No autoscaling — single host by design at this stage.
