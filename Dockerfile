# Gemma-Cyber API service image.
#
# This image runs the FastAPI inference service ONLY. The model runtime (Ollama)
# runs as a separate service/container — see docker-compose.yml — because the two
# have very different resource profiles (the API is tiny; Ollama holds the model).
# Keeping them separate is the "boring, maintainable" choice: scale, restart, or
# swap either independently, and the API image stays small and CPU-only.

FROM python:3.12-slim AS base

# Non-root by default (least privilege).
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

# Install deps first for layer caching. Copy only what the build needs.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package with the API extra. --no-cache-dir keeps the image lean.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[api]"

# The registry is optional metadata; mount or bake it and point the app at it.
# NOTE: ENV=prod is a SAFE default — with it the app fail-closes (refuses to start)
# unless auth is configured (GEMMA_CYBER_AUTH_DOMAIN+AUDIENCE, or GEMMA_CYBER_API_TOKEN).
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
