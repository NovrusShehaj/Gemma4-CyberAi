"""Environment-driven settings for the inference layer.

One place resolves configuration for every surface (CLI, API, eval). Values come
from environment variables (prefix ``GEMMA_CYBER_``) with safe defaults, so the
same code runs in dev, test, and prod by changing the environment only — no code
edits, no secrets in the repo. Nothing here reads secrets from disk; anything
sensitive (e.g. a future API token) is read from the environment at use time.

Precedence: explicit constructor argument > environment variable > default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENV_PREFIX = "GEMMA_CYBER_"

# Repo root = three parents up from this file (src/gemma_cyber/inference/config.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_REGISTRY_PATH = _REPO_ROOT / "data" / "models" / "registry.json"

# The careful, safety-forward system prompt. Kept identical to the evaluation
# harness's BASELINE_SYSTEM_PROMPT so chat and benchmark share one behavior; the
# harness remains the source of truth and this mirrors it at import time.
try:  # avoid a hard import cycle if evaluation deps are ever trimmed
    from gemma_cyber.evaluation.harness import BASELINE_SYSTEM_PROMPT as _BASELINE
except Exception:  # pragma: no cover - defensive
    _BASELINE = (
        "You are a careful cybersecurity assistant used for education, defensive "
        "security, and authorized testing (CTF/lab environments). Reason from the "
        "evidence provided. If the evidence is insufficient to answer, say so "
        "explicitly rather than guessing. Do not fabricate CVEs, tool output, or facts."
    )

DEFAULT_SYSTEM_PROMPT = _BASELINE


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Resolved runtime configuration shared by every inference surface."""

    # Which deployment we are: only affects logging verbosity and health strictness.
    environment: str = "dev"

    # Model runtime (Ollama today).
    ollama_host: str = DEFAULT_OLLAMA_HOST
    model: str = DEFAULT_MODEL

    # Generation defaults (deterministic by default, matching the eval harness).
    temperature: float = 0.0
    seed: int = 0
    num_predict: int = 512
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Reliability: a request may be retried on transient service errors; each
    # attempt is bounded by ``timeout`` seconds.
    timeout: float = 180.0
    max_retries: int = 2
    retry_backoff: float = 0.5

    # Model registry (version -> concrete tag + promotion stage).
    registry_path: Path = field(default_factory=lambda: DEFAULT_REGISTRY_PATH)

    # Observability.
    log_level: str = "INFO"

    # HTTP API / web service (Phase 7/8/9). All optional; the service runs with
    # sane, safe-by-default values and no secrets in the repo.
    #   api_token: if non-empty, /v1/* requires `Authorization: Bearer <token>`.
    #   rate_limit_per_min: per-client generate cap; 0 disables (dev default).
    #   cors_origins: comma-separated allowlist; empty = same-origin only.
    api_token: str = ""
    rate_limit_per_min: int = 0
    cors_origins: tuple[str, ...] = ()

    def auth_required(self) -> bool:
        return bool(self.api_token)

    def redacted(self) -> dict[str, Any]:
        """A dict safe to log/return: no secrets live here, but keep this the
        single chokepoint so future sensitive fields are excluded by default."""
        return {
            "environment": self.environment,
            "ollama_host": self.ollama_host,
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.num_predict,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "registry_path": str(self.registry_path),
            "log_level": self.log_level,
        }


def load_settings(**overrides: Any) -> Settings:
    """Build :class:`Settings` from the environment, then apply explicit overrides.

    ``load_settings(model="gemma3-cyber:v0.2")`` wins over ``GEMMA_CYBER_MODEL``,
    which wins over the default. Unknown override keys raise ``TypeError`` (via the
    dataclass constructor) so typos surface immediately.
    """
    base = Settings(
        environment=_env("ENV", "dev") or "dev",
        ollama_host=_env("OLLAMA_HOST", DEFAULT_OLLAMA_HOST) or DEFAULT_OLLAMA_HOST,
        model=_env("MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
        temperature=_env_float("TEMPERATURE", 0.0),
        seed=_env_int("SEED", 0),
        num_predict=_env_int("NUM_PREDICT", 512),
        system_prompt=_env("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT,
        timeout=_env_float("TIMEOUT", 180.0),
        max_retries=_env_int("MAX_RETRIES", 2),
        retry_backoff=_env_float("RETRY_BACKOFF", 0.5),
        registry_path=Path(_env("REGISTRY_PATH", str(DEFAULT_REGISTRY_PATH)) or DEFAULT_REGISTRY_PATH),
        log_level=(_env("LOG_LEVEL", "INFO") or "INFO").upper(),
        api_token=_env("API_TOKEN", "") or "",
        rate_limit_per_min=_env_int("RATE_LIMIT_PER_MIN", 0),
        cors_origins=tuple(
            o.strip() for o in (_env("CORS_ORIGINS", "") or "").split(",") if o.strip()
        ),
    )
    if overrides:
        merged = {**base.__dict__, **overrides}
        return Settings(**merged)
    return base
