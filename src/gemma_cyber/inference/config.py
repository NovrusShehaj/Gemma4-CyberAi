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

# Deployment modes recognised anywhere config is validated. ``prod``/``staging``
# are "hosted" (public-facing, strict); ``dev``/``test`` are permissive-by-choice.
KNOWN_ENVIRONMENTS: frozenset[str] = frozenset({"dev", "test", "staging", "prod"})
HOSTED_ENVIRONMENTS: frozenset[str] = frozenset({"staging", "prod"})

# Repo root = three parents up from this file (src/gemma_cyber/inference/config.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]


class ConfigError(ValueError):
    """A resolved :class:`Settings` is structurally invalid or unsafe for its mode.

    Raised by :meth:`Settings.validate` at startup so a misconfigured deployment
    fails fast with a precise message instead of surfacing as surprising runtime
    behavior. Messages never include secret values.
    """


def _looks_like_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://")) and "//" in value

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


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


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

    # Capacity / admission control (Phase 3). Bounds concurrent in-flight
    # generations so a burst of long streams cannot starve the single Ollama host.
    #   max_concurrent_generations: hard cap on active generations; 0 disables.
    #   request_deadline_s: total wall-clock budget for a request across retries
    #                       (0 = fall back to per-attempt ``timeout`` only).
    max_concurrent_generations: int = 4
    request_deadline_s: float = 0.0

    # Product policy (Phase 1). In hosted mode the server owns the safety prompt
    # and the set of servable models; clients may not silently swap either unless
    # this is explicitly enabled (self-host / internal expert use).
    #   allow_client_overrides: permit client-supplied ``system`` and arbitrary
    #                           ``model`` tags on /v1/generate. Default off.
    allow_client_overrides: bool = False

    # Registry ownership (Phase 1). Hosted deployments default to a read-only,
    # GitOps-managed registry: runtime admin-mutation routes are disabled and the
    # mount can stay ``:ro``. Set true only with a durable writable volume.
    registry_writable: bool = False

    def auth_required(self) -> bool:
        return bool(self.api_token)

    @property
    def hosted(self) -> bool:
        """True for public-facing environments (staging/prod) that get strict policy."""
        return self.environment in HOSTED_ENVIRONMENTS

    def validate(self) -> Settings:
        """Fail fast on structurally-invalid or mode-unsafe configuration.

        Returns ``self`` so callers can write ``settings = load_settings().validate()``.
        Raises :class:`ConfigError` with a precise, secret-free message. Structural
        checks apply in every mode; a few extra safety checks apply only when
        :attr:`hosted` (staging/prod), where an unsafe default must never slip through.
        """
        if self.environment not in KNOWN_ENVIRONMENTS:
            raise ConfigError(
                f"GEMMA_CYBER_ENV={self.environment!r} is not one of "
                f"{sorted(KNOWN_ENVIRONMENTS)}"
            )
        if not _looks_like_http_url(self.ollama_host):
            raise ConfigError(
                "GEMMA_CYBER_OLLAMA_HOST must be an http(s) URL, "
                f"got {self.ollama_host!r}"
            )
        if self.timeout <= 0:
            raise ConfigError(f"GEMMA_CYBER_TIMEOUT must be > 0, got {self.timeout}")
        if self.max_retries < 0:
            raise ConfigError(f"GEMMA_CYBER_MAX_RETRIES must be >= 0, got {self.max_retries}")
        if self.request_deadline_s < 0:
            raise ConfigError(
                f"GEMMA_CYBER_REQUEST_DEADLINE_S must be >= 0, got {self.request_deadline_s}"
            )
        if self.max_concurrent_generations < 0:
            raise ConfigError(
                "GEMMA_CYBER_MAX_CONCURRENT_GENERATIONS must be >= 0, got "
                f"{self.max_concurrent_generations}"
            )
        if self.rate_limit_per_min < 0:
            raise ConfigError(
                f"GEMMA_CYBER_RATE_LIMIT_PER_MIN must be >= 0, got {self.rate_limit_per_min}"
            )
        if self.num_predict < 1:
            raise ConfigError(f"GEMMA_CYBER_NUM_PREDICT must be >= 1, got {self.num_predict}")
        if self.log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ConfigError(f"GEMMA_CYBER_LOG_LEVEL={self.log_level!r} is not a valid level")
        for origin in self.cors_origins:
            if not _looks_like_http_url(origin):
                raise ConfigError(
                    f"GEMMA_CYBER_CORS_ORIGINS entry {origin!r} must be an http(s) origin"
                )
        if self.hosted:
            # A wildcard CORS origin in a public, credentialed deployment is unsafe.
            if "*" in self.cors_origins:
                raise ConfigError("wildcard CORS origin '*' is not allowed in hosted mode")
            if self.registry_writable and self.environment == "prod":
                # Allowed, but only deliberately: it needs a durable writable volume.
                pass
        return self

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
            "max_concurrent_generations": self.max_concurrent_generations,
            "request_deadline_s": self.request_deadline_s,
            "allow_client_overrides": self.allow_client_overrides,
            "registry_writable": self.registry_writable,
        }


def load_settings(**overrides: Any) -> Settings:
    """Build :class:`Settings` from the environment, then apply explicit overrides.

    ``load_settings(model="gemma3-cyber:v0.2")`` wins over ``GEMMA_CYBER_MODEL``,
    which wins over the default. Unknown override keys raise ``TypeError`` (via the
    dataclass constructor) so typos surface immediately.
    """
    environment = _env("ENV", "dev") or "dev"
    hosted = environment in HOSTED_ENVIRONMENTS
    base = Settings(
        environment=environment,
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
        max_concurrent_generations=_env_int("MAX_CONCURRENT_GENERATIONS", 4),
        request_deadline_s=_env_float("REQUEST_DEADLINE_S", 0.0),
        # Client overrides and runtime registry writes default OFF in hosted mode
        # (server owns the safety prompt + servable models; registry is GitOps).
        allow_client_overrides=_env_bool("ALLOW_CLIENT_OVERRIDES", not hosted),
        registry_writable=_env_bool("REGISTRY_WRITABLE", not hosted),
    )
    if overrides:
        merged = {**base.__dict__, **overrides}
        return Settings(**merged)
    return base
