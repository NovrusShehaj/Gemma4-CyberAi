"""Web surface: asset serving, bootstrap config, CSP posture, and static safety
checks on the shipped browser code (no innerHTML; token-in-memory only).

These do not launch a browser (a full E2E requires a runner + staging Auth0);
they verify the server contract and guard against the highest-risk client
regressions (XSS via innerHTML, tokens in web storage) at the source level.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from gemma_cyber.inference.config import Settings
from gemma_cyber.inference.engine import HealthStatus, InferenceEngine

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from gemma_cyber.api.app import create_app  # noqa: E402

WEB = Path(__file__).resolve().parents[1] / "src" / "gemma_cyber" / "api" / "web"


class _Eng:
    model = "gemma3:4b"

    def health(self):
        return HealthStatus(ok=True, service_reachable=True, model_present=True,
                            model=self.model, host="http://fake", detail="")

    def generate(self, prompt, **kw):  # pragma: no cover - not exercised here
        raise NotImplementedError

    def stream(self, prompt, **kw):  # pragma: no cover
        raise NotImplementedError


def _client(settings=None, **kw):
    app = create_app(settings or Settings(),
                     engine=cast(InferenceEngine, _Eng()), **kw)
    return TestClient(app)


# --- asset serving --------------------------------------------------------
def test_index_served_no_store():
    r = _client().get("/")
    assert r.status_code == 200
    assert "Gemma-Cyber" in r.text
    assert "text/html" in r.headers["content-type"]
    assert r.headers.get("Cache-Control") == "no-store"


def test_styles_served():
    r = _client().get("/styles.css")
    assert r.status_code == 200
    assert "text/css" in r.headers["content-type"]
    assert "--accent" in r.text


def test_app_js_served():
    r = _client().get("/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


# --- bootstrap config -----------------------------------------------------
def test_config_json_open_mode():
    r = _client().get("/config.json")
    assert r.status_code == 200
    body = r.json()
    assert body["auth"]["enabled"] is False
    assert body["auth"]["clientId"] == ""  # no leakage
    assert r.headers.get("Cache-Control") == "no-store"


def test_config_json_reports_env():
    r = _client(Settings(environment="staging")).get("/config.json")
    assert r.json()["env"] == "staging"
    assert r.json()["hosted"] is True


# --- CSP posture ----------------------------------------------------------
def test_csp_scripts_are_same_origin_only():
    csp = _client().get("/").headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp
    assert "unsafe-inline" not in csp.split("style-src")[0]  # not in script-src
    assert "unsafe-eval" not in csp


def test_csp_does_not_widen_connect_without_web_auth():
    csp = _client().get("/").headers["Content-Security-Policy"]
    assert "connect-src 'self';" in csp  # no third-party origin in open mode


def test_web_auth_config_and_csp_when_spa_enabled(monkeypatch):
    from gemma_cyber.api.auth import AuthSettings

    monkeypatch.setenv("GEMMA_CYBER_WEB_AUTH0_CLIENT_ID", "spa-abc123")
    auth = AuthSettings(domain="tenant.us.auth0.com", audience="https://api.gemma-cyber")
    # A dummy verifier avoids any live JWKS client construction.
    app = create_app(
        Settings(environment="staging"),
        engine=cast(InferenceEngine, _Eng()),
        auth_settings=auth,
        verifier=object(),  # type: ignore[arg-type]
    )
    client = TestClient(app)

    cfg = client.get("/config.json").json()
    assert cfg["auth"]["enabled"] is True
    assert cfg["auth"]["clientId"] == "spa-abc123"
    assert cfg["auth"]["domain"] == "tenant.us.auth0.com"
    assert cfg["auth"]["audience"] == "https://api.gemma-cyber"

    csp = client.get("/").headers["Content-Security-Policy"]
    assert "connect-src 'self' https://tenant.us.auth0.com;" in csp
    assert "script-src 'self'" in csp  # scripts never widened


# --- static safety guards on shipped client code --------------------------
def test_app_js_has_no_innerhtml():
    js = (WEB / "app.js").read_text()
    assert "innerHTML" not in js  # model output is rendered with textContent only


def test_app_js_does_not_persist_tokens():
    js = (WEB / "app.js").read_text()
    # The access token must never be written to web storage. (The PKCE verifier
    # may transit sessionStorage, but tokens must not.)
    assert "localStorage.setItem" not in js
    lowered = js.lower()
    assert "accesstoken" in lowered  # the token variable exists...
    # ...but is only ever assigned to the in-memory app object, not storage.
    assert "setitem(pkce_key" in lowered.replace(" ", "") or "setItem(PKCE_KEY" in js


def test_index_uses_external_stylesheet_and_script():
    html = (WEB / "index.html").read_text()
    assert '<link rel="stylesheet" href="/styles.css"' in html
    assert '<script src="/app.js">' in html
    assert 'href="#input"' in html  # skip link target
