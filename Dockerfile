# Gemma-Cyber API service image.
#
# Runs the FastAPI inference service ONLY. The model runtime (Ollama) runs as a
# separate service/container (see docker-compose.yml): different resource
# profiles, independent scaling/restart, and a small CPU-only API image.
#
# Dependencies are installed from the COMMITTED uv.lock (`uv sync --locked`), so
# the image gets exactly the reviewed dependency graph CI tested — not a fresh,
# possibly-newer pip resolution. `uv.lock` is authoritative for CI and the image.
#
# RELEASE NOTE: for a reproducible production build, pin the base image by digest
# (FROM python:3.12-slim@sha256:...) after selecting/reviewing one. The tag below
# is a moving target and is intentionally left for the release owner to pin.
FROM python:3.12-slim AS base

# Non-root by default (least privilege).
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

# uv is a build-time tool; the app's dependency versions come from uv.lock, so
# uv's own version does not affect the resolved graph (--locked uses the lock).
RUN pip install --no-cache-dir --upgrade pip uv

# Copy lock + manifest first for layer caching, then source.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install the locked production graph (api extra, no dev tools) into /app/.venv.
# --frozen: never update the lock; fail if it is out of date.
RUN uv sync --locked --frozen --no-dev --extra api \
 && chown -R app:app /app

ENV PATH="/app/.venv/bin:${PATH}"

# ENV=prod is a SAFE default — the app fail-closes (refuses to start) unless auth
# is configured (GEMMA_CYBER_AUTH_DOMAIN+AUDIENCE, or GEMMA_CYBER_API_TOKEN).
# docker-compose overrides ENV=staging for a startable local stack. See docs/auth.md.
ENV GEMMA_CYBER_API_HOST=0.0.0.0 \
    GEMMA_CYBER_API_PORT=8000 \
    GEMMA_CYBER_OLLAMA_HOST=http://ollama:11434 \
    GEMMA_CYBER_ENV=prod \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

# Container-level liveness check hitting the cheap /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["gemma-cyber-serve"]
