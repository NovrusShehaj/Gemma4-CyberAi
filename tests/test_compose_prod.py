"""Production Compose overlay is fail-closed (auth env required)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROD = (ROOT / "docker-compose.prod.yml").read_text()
BASE = (ROOT / "docker-compose.yml").read_text()


def test_prod_overlay_is_prod_and_requires_auth_env():
    assert "GEMMA_CYBER_ENV: prod" in PROD
    assert "GEMMA_CYBER_AUTH_DOMAIN: ${GEMMA_CYBER_AUTH_DOMAIN:?set GEMMA_CYBER_AUTH_DOMAIN}" in PROD
    assert "GEMMA_CYBER_AUTH_AUDIENCE: ${GEMMA_CYBER_AUTH_AUDIENCE:?set GEMMA_CYBER_AUTH_AUDIENCE}" in PROD
    assert "GEMMA_CYBER_WEB_AUTH0_CLIENT_ID: ${GEMMA_CYBER_WEB_AUTH0_CLIENT_ID:?set GEMMA_CYBER_WEB_AUTH0_CLIENT_ID}" in PROD
    assert "GEMMA_CYBER_RATE_LIMIT_PER_MIN" in PROD


def test_base_compose_stays_localhost_and_registry_ro():
    assert "127.0.0.1:8000:8000" in BASE
    assert "./data/models:/app/registry:ro" in BASE
    assert "GEMMA_CYBER_ENV: staging" in BASE
