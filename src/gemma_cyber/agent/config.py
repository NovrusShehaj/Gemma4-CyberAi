"""Configuration for `gemma4` (plan §19).

Precedence, highest first::

    CLI flags > GEMMA4_* env > <project>/.gemma4/config.toml
              > ~/.config/gemma4/config.toml > defaults

Two privilege rules are enforced here rather than documented and hoped for:

* **No credentials in TOML.** Any ``api_key`` / ``*_token`` / ``*_secret``-shaped
  key is dropped from *either* config file with a warning. Keys come from the
  environment (``GEMMA4_API_KEY``) or, later, an OS keyring.
* **No trusted mode from a file.** ``mode = "trusted"`` is dropped from both the
  project and the user config. Trusted is a startup risk decision that requires
  ``--mode trusted --i-accept-risk`` on a TTY (plan §15); a file that a repo can
  ship — or that an injected edit could write — must not be able to express it.

The second rule is stricter than the plan's literal wording (which forbids only
*project* config). Deliberate: the plan's mode table lists CLI flags as the only
enabler, and a `~/.config` file is still not an interactive risk acknowledgement.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from gemma_cyber.agent.errors import ConfigurationError
from gemma_cyber.agent.types import PermissionMode

__all__ = [
    "AgentConfig",
    "ContextConfig",
    "LoadedConfig",
    "PermissionsConfig",
    "ProviderConfig",
    "ToolsConfig",
    "audit_dir",
    "data_dir",
    "history_path",
    "load_config",
    "project_config_path",
    "sessions_dir",
    "undo_dir",
    "user_config_path",
]

CONFIG_SCHEMA_VERSION = 1

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
#: Honest default: the base model, not a promoted cyber model (docs/model-card.md).
DEFAULT_MODEL = "gemma3:4b"
API_KEY_ENV = "GEMMA4_API_KEY"

#: Keys that may never appear in a config file, at any nesting depth.
_FORBIDDEN_KEY_RE = re.compile(
    r"(api_?key|secret|password|passwd|token|credential|bearer|private_key)", re.IGNORECASE
)


# -- XDG locations ----------------------------------------------------------

def _xdg(env_name: str, default: Path) -> Path:
    raw = os.environ.get(env_name, "").strip()
    return Path(raw).expanduser() if raw else default


def user_config_path() -> Path:
    """``$XDG_CONFIG_HOME/gemma4/config.toml`` (default ``~/.config/...``)."""
    return _xdg("XDG_CONFIG_HOME", Path.home() / ".config") / "gemma4" / "config.toml"


def data_dir() -> Path:
    """``$XDG_DATA_HOME/gemma4`` (default ``~/.local/share/gemma4``).

    ``GEMMA4_DATA_DIR`` overrides it outright; tests use that to stay off the
    developer's real session store.
    """
    override = os.environ.get("GEMMA4_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return _xdg("XDG_DATA_HOME", Path.home() / ".local" / "share") / "gemma4"


def sessions_dir() -> Path:
    return data_dir() / "sessions"


def undo_dir() -> Path:
    return data_dir() / "undo"


def audit_dir() -> Path:
    return data_dir() / "audit"


def history_path() -> Path:
    return data_dir() / "history"


def project_config_path(workspace_root: Path) -> Path:
    return workspace_root / ".gemma4" / "config.toml"


# -- typed models -----------------------------------------------------------

class _Strict(BaseModel):
    """Reject unknown keys so a typo becomes an error, not a silent default."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConfig(_Strict):
    """One named provider. Never holds a credential — see module docstring."""

    type: Literal["ollama", "openai-compatible", "fake"] = "ollama"
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_MODEL
    #: Which environment variable carries the key for this provider, if any.
    api_key_env: str = API_KEY_ENV
    #: Provider-reported context budget override, when the endpoint cannot say.
    context_tokens: int | None = None
    #: Opt in to native tool schemas. Off by default: unverified native tool
    #: support on a small local model silently produces no calls (plan §9).
    tools_native: bool = False
    timeout_s: float = Field(default=180.0, gt=0)

    @field_validator("base_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http(s) URL")
        return value.rstrip("/")

    def api_key(self) -> str | None:
        """Read the credential from the environment at use time, never from disk."""
        return os.environ.get(self.api_key_env) or None


class PermissionsConfig(_Strict):
    """Capability ceiling and the narrow exceptions to "always confirm"."""

    mode: PermissionMode = PermissionMode.READ_ONLY
    #: Exact argv prefixes auto-approved for `shell.exec` in `agent` mode, e.g.
    #: ``["pytest", "ruff check", "git status"]``. Never applies in read-only /
    #: workspace mode, where shell is denied outright.
    shell_allowlist: tuple[str, ...] = ()
    allow_network: bool = False
    allow_private_network: bool = False
    allow_home: bool = False

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return PermissionMode(value)
            except ValueError:
                raise ValueError(
                    f"unknown permission mode {value!r}; expected one of "
                    f"{[m.value for m in PermissionMode]}"
                ) from None
        return value


class ContextConfig(_Strict):
    """Budgets. Every one of these bounds something the model can grow."""

    max_file_bytes: int = Field(default=1_048_576, gt=0)
    tool_result_max_chars: int = Field(default=16_000, gt=0)
    #: Hard cap applied by ToolRuntime before truncation bookkeeping (plan §11.6).
    tool_output_max_bytes: int = Field(default=32_768, gt=0)
    max_iterations: int = Field(default=12, ge=1)
    turn_timeout_s: float = Field(default=600.0, gt=0)
    #: Repeat count of an identical tool+args call that trips the loop detector.
    loop_repeat_limit: int = Field(default=3, ge=2)
    instructions_max_bytes: int = Field(default=16_384, gt=0)
    tree_max_entries: int = Field(default=200, gt=0)


class ToolsConfig(_Strict):
    shell_timeout_s: float = Field(default=60.0, gt=0)
    grep_max_results: int = Field(default=200, gt=0)
    glob_max_results: int = Field(default=500, gt=0)


class AgentConfig(_Strict):
    """The fully-resolved configuration handed to the runtime."""

    schema_version: int = CONFIG_SCHEMA_VERSION
    default_provider: str = "default"
    providers: dict[str, ProviderConfig] = Field(
        default_factory=lambda: {"default": ProviderConfig()}
    )
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    profile: str = "general"

    def provider(self, name: str | None = None) -> ProviderConfig:
        key = name or self.default_provider
        try:
            return self.providers[key]
        except KeyError:
            raise ConfigurationError(
                f"unknown provider {key!r}; configured: {sorted(self.providers)}"
            ) from None


class LoadedConfig(BaseModel):
    """``config`` plus the provenance and warnings the CLI shows at startup."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    config: AgentConfig
    sources: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


# -- TOML reading + privilege stripping -------------------------------------

def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigurationError(f"cannot read config {path.name}: {exc.strerror}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"invalid TOML in {path.name}: {exc}") from exc
    if not isinstance(data, dict):  # pragma: no cover - tomllib always returns dict
        raise ConfigurationError(f"invalid TOML in {path.name}: top level must be a table")
    return data


def _strip_forbidden(
    data: dict[str, Any], *, origin: str, warnings: list[str], _prefix: str = ""
) -> dict[str, Any]:
    """Recursively drop credential-shaped keys, warning once per occurrence."""
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{_prefix}{key}"
        if _FORBIDDEN_KEY_RE.search(key):
            warnings.append(
                f"{origin}: ignoring credential-shaped key {dotted!r}. "
                f"Set {API_KEY_ENV} in the environment instead; never store keys in TOML."
            )
            continue
        if isinstance(value, dict):
            cleaned[key] = _strip_forbidden(
                value, origin=origin, warnings=warnings, _prefix=f"{dotted}."
            )
        else:
            cleaned[key] = value
    return cleaned


def _strip_trusted(data: dict[str, Any], *, origin: str, warnings: list[str]) -> dict[str, Any]:
    """Remove ``[permissions] mode = "trusted"``. Files cannot grant trusted mode."""
    perms = data.get("permissions")
    if isinstance(perms, dict) and str(perms.get("mode", "")).strip() == PermissionMode.TRUSTED:
        warnings.append(
            f"{origin}: ignoring mode = \"trusted\". Trusted mode requires "
            "`--mode trusted --i-accept-risk` on a TTY and can never come from a file."
        )
        perms = {k: v for k, v in perms.items() if k != "mode"}
        data = {**data, "permissions": perms}
    return data


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overlay`` onto ``base`` (overlay wins per leaf key)."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


# -- environment layer ------------------------------------------------------

def _env_overlay(warnings: list[str], environ: dict[str, str]) -> dict[str, Any]:
    """Build a config overlay from ``GEMMA4_*`` variables.

    ``GEMMA4_API_KEY`` is intentionally *not* mapped into the config tree: it is
    read at use time by :meth:`ProviderConfig.api_key` so it never lands in a
    model dump, a log line, or `gemma4 config path` output.
    """
    overlay: dict[str, Any] = {}
    provider: dict[str, Any] = {}

    if model := environ.get("GEMMA4_MODEL", "").strip():
        provider["model"] = model
    base_url = environ.get("GEMMA4_BASE_URL", "").strip()
    if not base_url:
        # Reuse the repo's existing Ollama host convention (plan §9) as a fallback
        # so one variable configures both the generate path and the agent.
        base_url = environ.get("GEMMA_CYBER_OLLAMA_HOST", "").strip()
    if base_url:
        provider["base_url"] = base_url.rstrip("/")
    if provider:
        overlay.setdefault("providers", {})["__env__"] = provider

    if name := environ.get("GEMMA4_PROVIDER", "").strip():
        overlay["default_provider"] = name

    if raw_mode := environ.get("GEMMA4_MODE", "").strip():
        if raw_mode == PermissionMode.TRUSTED:
            # Env is more trusted than a repo file but is still not an interactive
            # acknowledgement; the CLI gate (--i-accept-risk + TTY) decides.
            overlay.setdefault("permissions", {})["mode"] = raw_mode
        else:
            try:
                PermissionMode(raw_mode)
            except ValueError:
                warnings.append(
                    f"env: ignoring GEMMA4_MODE={raw_mode!r} (not a valid permission mode)"
                )
            else:
                overlay.setdefault("permissions", {})["mode"] = raw_mode

    if profile := environ.get("GEMMA4_PROFILE", "").strip():
        overlay["profile"] = profile

    for env_name, section, key, caster in (
        ("GEMMA4_MAX_ITERATIONS", "context", "max_iterations", int),
        ("GEMMA4_TURN_TIMEOUT_S", "context", "turn_timeout_s", float),
        ("GEMMA4_MAX_FILE_BYTES", "context", "max_file_bytes", int),
        ("GEMMA4_TOOL_RESULT_MAX_CHARS", "context", "tool_result_max_chars", int),
        ("GEMMA4_SHELL_TIMEOUT_S", "tools", "shell_timeout_s", float),
    ):
        raw = environ.get(env_name, "").strip()
        if not raw:
            continue
        try:
            overlay.setdefault(section, {})[key] = caster(raw)
        except ValueError:
            warnings.append(f"env: ignoring {env_name}={raw!r} (not a number)")

    return overlay


def _retarget_env_provider(overlay: dict[str, Any], target: str) -> dict[str, Any]:
    """Rewrite the synthetic ``__env__`` provider onto the provider actually in use.

    ``GEMMA4_MODEL`` overrides whichever provider is selected rather than defining
    a new one. This happens *within* the env layer so a later CLI ``--model``
    still wins — folding it after the CLI merge would invert the precedence.
    """
    providers = overlay.get("providers")
    if not isinstance(providers, dict) or "__env__" not in providers:
        return overlay
    providers = dict(providers)
    env_provider = providers.pop("__env__")
    providers[target] = _merge(providers.get(target, {}) or {}, env_provider)
    return {**overlay, "providers": providers}


def _select_default_provider(layers: list[dict[str, Any]]) -> str:
    """Highest-precedence ``default_provider`` across layers (last layer wins)."""
    target = "default"
    for layer in layers:
        name = layer.get("default_provider")
        if isinstance(name, str) and name.strip():
            target = name.strip()
    return target


# -- public loader ----------------------------------------------------------

def load_config(
    *,
    workspace_root: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
    user_config: Path | None = None,
) -> LoadedConfig:
    """Resolve configuration across all five layers.

    ``cli_overrides`` is a nested dict in the same shape as the TOML (e.g.
    ``{"permissions": {"mode": "agent"}}``); the CLI builds it from flags. It is
    the only layer allowed to select trusted mode.
    """
    environ = dict(os.environ if environ is None else environ)
    warnings: list[str] = []
    sources: list[str] = ["defaults"]

    layers: list[dict[str, Any]] = []

    user_path = user_config if user_config is not None else user_config_path()
    if user_path.is_file():
        raw = _read_toml(user_path)
        raw = _strip_forbidden(raw, origin="user config", warnings=warnings)
        raw = _strip_trusted(raw, origin="user config", warnings=warnings)
        layers.append(raw)
        sources.append(str(user_path))

    if workspace_root is not None:
        proj_path = project_config_path(workspace_root)
        if proj_path.is_file():
            raw = _read_toml(proj_path)
            # Project config is repo content: the least trusted file layer.
            raw = _strip_forbidden(raw, origin="project config", warnings=warnings)
            raw = _strip_trusted(raw, origin="project config", warnings=warnings)
            layers.append(raw)
            sources.append(str(proj_path))

    env_overlay = _env_overlay(warnings, environ)
    if env_overlay:
        layers.append(env_overlay)
        sources.append("env")

    if cli_overrides:
        layers.append(dict(cli_overrides))
        sources.append("cli")

    # Resolve the provider name first so GEMMA4_MODEL lands on the provider in
    # use, then merge strictly in precedence order.
    target = _select_default_provider(layers)
    if env_overlay:
        layers[layers.index(env_overlay)] = _retarget_env_provider(env_overlay, target)

    merged: dict[str, Any] = {}
    for layer in layers:
        merged = _merge(merged, layer)
    merged = _ensure_default_provider(merged)

    declared = merged.get("schema_version", CONFIG_SCHEMA_VERSION)
    if isinstance(declared, int) and declared > CONFIG_SCHEMA_VERSION:
        warnings.append(
            f"config schema_version {declared} is newer than this build "
            f"({CONFIG_SCHEMA_VERSION}); unknown keys will be rejected"
        )

    try:
        config = AgentConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigurationError(_format_validation_error(exc)) from exc

    return LoadedConfig(config=config, sources=tuple(sources), warnings=tuple(warnings))


def _ensure_default_provider(data: dict[str, Any]) -> dict[str, Any]:
    """Guarantee ``providers[default_provider]`` exists so lookups cannot KeyError."""
    providers = dict(data.get("providers") or {})
    target = str(data.get("default_provider", "default"))
    providers.setdefault(target, {})
    providers.setdefault("default", {})
    return {**data, "providers": providers}


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "__root__")
        parts.append(f"{loc or 'config'}: {err.get('msg', 'invalid value')}")
    return "invalid gemma4 configuration — " + "; ".join(parts)


def write_default_config(path: Path) -> Path:
    """Write a commented starter config for `gemma4 config init`. Never a key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# gemma4 configuration. Precedence:
#   CLI flags > GEMMA4_* env > <project>/.gemma4/config.toml > this file > defaults
#
# NEVER put an API key here. Credentials come from the {API_KEY_ENV} environment
# variable; any key-shaped entry in this file is ignored with a warning.
# `mode = "trusted"` is also ignored here: it requires
# `gemma4 --mode trusted --i-accept-risk` on a TTY.
schema_version = {CONFIG_SCHEMA_VERSION}
default_provider = "default"

[providers.default]
type = "ollama"
base_url = "{DEFAULT_OLLAMA_BASE_URL}"
model = "{DEFAULT_MODEL}"

[permissions]
mode = "read-only"
# Exact argv prefixes auto-approved for shell.exec in `agent` mode only:
# shell_allowlist = ["pytest", "ruff check", "git status", "git diff"]

[context]
max_file_bytes = 1048576
tool_result_max_chars = 16000
max_iterations = 12
""",
        encoding="utf-8",
    )
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass
    return path
