"""Config precedence and the two privilege rules that must hold in code.

The interesting assertions are not "TOML parses" but "a repo cannot hand itself
trusted mode" and "a key in a file never becomes a credential".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma_cyber.agent.config import (
    API_KEY_ENV,
    AgentConfig,
    ProviderConfig,
    load_config,
    write_default_config,
)
from gemma_cyber.agent.errors import ConfigurationError
from gemma_cyber.agent.types import PermissionMode


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_are_fail_closed(tmp_path: Path) -> None:
    loaded = load_config(workspace_root=tmp_path, environ={}, user_config=tmp_path / "none.toml")
    cfg = loaded.config
    assert cfg.permissions.mode is PermissionMode.READ_ONLY
    assert cfg.permissions.allow_network is False
    assert cfg.permissions.allow_private_network is False
    assert cfg.permissions.shell_allowlist == ()
    assert cfg.provider().model == "gemma3:4b"


def test_precedence_cli_beats_env_beats_project_beats_user(tmp_path: Path) -> None:
    user = _write(
        tmp_path / "user.toml",
        '[providers.default]\nmodel = "from-user"\n[context]\nmax_iterations = 4\n',
    )
    _write(
        tmp_path / "ws" / ".gemma4" / "config.toml",
        '[providers.default]\nmodel = "from-project"\n',
    )
    ws = tmp_path / "ws"

    # user only
    c = load_config(workspace_root=tmp_path, environ={}, user_config=user).config
    assert c.provider().model == "from-user"

    # project beats user
    c = load_config(workspace_root=ws, environ={}, user_config=user).config
    assert c.provider().model == "from-project"
    assert c.context.max_iterations == 4  # untouched user value survives the merge

    # env beats project
    c = load_config(
        workspace_root=ws, environ={"GEMMA4_MODEL": "from-env"}, user_config=user
    ).config
    assert c.provider().model == "from-env"

    # cli beats env
    c = load_config(
        workspace_root=ws,
        environ={"GEMMA4_MODEL": "from-env"},
        user_config=user,
        cli_overrides={"providers": {"default": {"model": "from-cli"}}},
    ).config
    assert c.provider().model == "from-cli"


def test_project_config_cannot_set_trusted_mode(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write(ws / ".gemma4" / "config.toml", '[permissions]\nmode = "trusted"\n')
    loaded = load_config(workspace_root=ws, environ={}, user_config=tmp_path / "none.toml")
    assert loaded.config.permissions.mode is PermissionMode.READ_ONLY
    assert any("trusted" in w for w in loaded.warnings)


def test_user_config_cannot_set_trusted_mode(tmp_path: Path) -> None:
    user = _write(tmp_path / "user.toml", '[permissions]\nmode = "trusted"\n')
    loaded = load_config(workspace_root=tmp_path, environ={}, user_config=user)
    assert loaded.config.permissions.mode is PermissionMode.READ_ONLY
    assert any("trusted" in w for w in loaded.warnings)


def test_config_file_may_still_select_a_non_trusted_mode(tmp_path: Path) -> None:
    user = _write(tmp_path / "user.toml", '[permissions]\nmode = "workspace"\n')
    loaded = load_config(workspace_root=tmp_path, environ={}, user_config=user)
    assert loaded.config.permissions.mode is PermissionMode.WORKSPACE


def test_only_cli_layer_can_express_trusted(tmp_path: Path) -> None:
    loaded = load_config(
        workspace_root=tmp_path,
        environ={},
        user_config=tmp_path / "none.toml",
        cli_overrides={"permissions": {"mode": "trusted"}},
    )
    # The loader permits it; the CLI still gates on --i-accept-risk + TTY.
    assert loaded.config.permissions.mode is PermissionMode.TRUSTED


@pytest.mark.parametrize(
    "line",
    [
        'api_key = "sk-live-abc"',
        'apikey = "sk-live-abc"',
        'token = "ghp_xxx"',
        'auth_token = "ghp_xxx"',
        'client_secret = "s3cr3t"',
        'password = "hunter2"',
        'private_key = "-----BEGIN"',
    ],
)
def test_credential_shaped_keys_are_stripped_from_project_config(
    tmp_path: Path, line: str
) -> None:
    ws = tmp_path / "ws"
    _write(ws / ".gemma4" / "config.toml", f"[providers.default]\ntype = \"ollama\"\n{line}\n")
    loaded = load_config(workspace_root=ws, environ={}, user_config=tmp_path / "none.toml")
    # Stripped, not merely rejected: an extra="forbid" model would have raised.
    assert loaded.config.provider().type == "ollama"
    assert any(API_KEY_ENV in w for w in loaded.warnings)
    dumped = loaded.config.model_dump_json()
    for secret in ("sk-live-abc", "ghp_xxx", "s3cr3t", "hunter2", "BEGIN"):
        assert secret not in dumped


def test_credential_shaped_keys_are_stripped_from_user_config(tmp_path: Path) -> None:
    user = _write(tmp_path / "user.toml", '[providers.remote]\napi_key = "sk-user"\n')
    loaded = load_config(workspace_root=tmp_path, environ={}, user_config=user)
    assert "sk-user" not in loaded.config.model_dump_json()
    assert loaded.warnings


def test_api_key_comes_from_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ProviderConfig(type="openai-compatible", base_url="http://127.0.0.1:8000/v1")
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    assert provider.api_key() is None
    monkeypatch.setenv(API_KEY_ENV, "sk-from-env")
    assert provider.api_key() == "sk-from-env"
    # ...and it is not part of the serialised config.
    assert "sk-from-env" not in provider.model_dump_json()


def test_unknown_key_is_a_configuration_error(tmp_path: Path) -> None:
    user = _write(tmp_path / "user.toml", "[permissions]\nmoed = \"agent\"\n")
    with pytest.raises(ConfigurationError) as err:
        load_config(workspace_root=tmp_path, environ={}, user_config=user)
    assert "moed" in str(err.value)


def test_invalid_mode_string_is_a_configuration_error(tmp_path: Path) -> None:
    user = _write(tmp_path / "user.toml", '[permissions]\nmode = "yolo"\n')
    with pytest.raises(ConfigurationError):
        load_config(workspace_root=tmp_path, environ={}, user_config=user)


def test_invalid_env_mode_is_ignored_with_warning(tmp_path: Path) -> None:
    loaded = load_config(
        workspace_root=tmp_path,
        environ={"GEMMA4_MODE": "yolo"},
        user_config=tmp_path / "none.toml",
    )
    assert loaded.config.permissions.mode is PermissionMode.READ_ONLY
    assert any("GEMMA4_MODE" in w for w in loaded.warnings)


def test_malformed_toml_fails_closed(tmp_path: Path) -> None:
    user = _write(tmp_path / "user.toml", "this is not = = toml")
    with pytest.raises(ConfigurationError):
        load_config(workspace_root=tmp_path, environ={}, user_config=user)


def test_legacy_ollama_host_env_is_honoured(tmp_path: Path) -> None:
    loaded = load_config(
        workspace_root=tmp_path,
        environ={"GEMMA_CYBER_OLLAMA_HOST": "http://10.0.0.5:11434/"},
        user_config=tmp_path / "none.toml",
    )
    assert loaded.config.provider().base_url == "http://10.0.0.5:11434"
    # GEMMA4_BASE_URL wins over the legacy variable.
    loaded = load_config(
        workspace_root=tmp_path,
        environ={
            "GEMMA_CYBER_OLLAMA_HOST": "http://10.0.0.5:11434",
            "GEMMA4_BASE_URL": "http://127.0.0.1:9999",
        },
        user_config=tmp_path / "none.toml",
    )
    assert loaded.config.provider().base_url == "http://127.0.0.1:9999"


def test_env_model_overrides_the_selected_provider_not_a_new_one(tmp_path: Path) -> None:
    user = _write(
        tmp_path / "user.toml",
        'default_provider = "remote"\n'
        '[providers.remote]\ntype = "openai-compatible"\n'
        'base_url = "http://127.0.0.1:8000/v1"\nmodel = "configured"\n',
    )
    cfg = load_config(
        workspace_root=tmp_path, environ={"GEMMA4_MODEL": "env-model"}, user_config=user
    ).config
    assert cfg.default_provider == "remote"
    assert cfg.provider().type == "openai-compatible"
    assert cfg.provider().model == "env-model"


def test_unknown_provider_name_raises(tmp_path: Path) -> None:
    cfg = AgentConfig()
    with pytest.raises(ConfigurationError):
        cfg.provider("nope")


def test_written_starter_config_contains_no_key_and_loads(tmp_path: Path) -> None:
    path = write_default_config(tmp_path / "config.toml")
    text = path.read_text()
    assert "api_key" not in text.replace(f"{API_KEY_ENV}", "")
    loaded = load_config(workspace_root=tmp_path, environ={}, user_config=path)
    assert loaded.config.permissions.mode is PermissionMode.READ_ONLY
    assert not loaded.warnings
