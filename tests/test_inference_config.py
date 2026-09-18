"""Settings resolve from env with correct precedence and safe defaults."""

from __future__ import annotations

from pathlib import Path

from gemma_cyber.inference.config import DEFAULT_MODEL, Settings, load_settings


def test_defaults_when_env_empty(monkeypatch):
    for k in list(dict(**__import__("os").environ)):
        if k.startswith("GEMMA_CYBER_"):
            monkeypatch.delenv(k, raising=False)
    s = load_settings()
    assert s.model == DEFAULT_MODEL
    assert s.temperature == 0.0
    assert s.seed == 0
    assert s.environment == "dev"
    assert isinstance(s.registry_path, Path)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("GEMMA_CYBER_MODEL", "gemma3-cyber:v0.2")
    monkeypatch.setenv("GEMMA_CYBER_TEMPERATURE", "0.7")
    monkeypatch.setenv("GEMMA_CYBER_MAX_RETRIES", "5")
    monkeypatch.setenv("GEMMA_CYBER_ENV", "prod")
    s = load_settings()
    assert s.model == "gemma3-cyber:v0.2"
    assert s.temperature == 0.7
    assert s.max_retries == 5
    assert s.environment == "prod"


def test_explicit_override_beats_env(monkeypatch):
    monkeypatch.setenv("GEMMA_CYBER_MODEL", "from-env")
    s = load_settings(model="explicit")
    assert s.model == "explicit"


def test_malformed_numeric_env_falls_back(monkeypatch):
    monkeypatch.setenv("GEMMA_CYBER_SEED", "not-an-int")
    s = load_settings()
    assert s.seed == 0  # default, not a crash


def test_redacted_has_no_unexpected_keys():
    s = Settings()
    red = s.redacted()
    assert "system_prompt" not in red  # keep the chokepoint minimal
    assert red["model"] == DEFAULT_MODEL
