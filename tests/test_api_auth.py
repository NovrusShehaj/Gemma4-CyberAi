"""Auth0/JWT authentication + server-side authorization tests.

The FULL verification path is exercised locally with a self-signed RS256 keypair
and an injected key resolver (mocked JWKS) — no live Auth0 tenant required. Covers
every negative case the trust boundary must reject, plus authorization on the
privileged admin endpoints.
"""

from __future__ import annotations

import time
from typing import cast

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jwt")

import jwt  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from gemma_cyber.api.app import create_app  # noqa: E402
from gemma_cyber.api.auth import AuthSettings, AuthUnavailableError, TokenVerifier  # noqa: E402
from gemma_cyber.clients.ollama_client import GenerationResult  # noqa: E402
from gemma_cyber.inference.config import Settings  # noqa: E402
from gemma_cyber.inference.engine import HealthStatus, InferenceEngine  # noqa: E402
from gemma_cyber.inference.registry import ModelRecord, ModelRegistry  # noqa: E402

AUDIENCE = "https://api.gemma-cyber"
DOMAIN = "test-tenant.auth0.com"
ISSUER = f"https://{DOMAIN}/"

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PRIVATE_PEM = _KEY.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
)
_PUBLIC_KEY = _KEY.public_key()

# A DIFFERENT key, to forge a token with a valid structure but a bad signature.
_OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_PRIVATE_PEM = _OTHER_KEY.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
)


def make_token(*, sub="user-1", aud=AUDIENCE, iss=ISSUER, scope="", permissions=None,
               exp_delta=3600, signing_key=_PRIVATE_PEM):
    now = int(time.time())
    payload: dict = {"sub": sub, "aud": aud, "iss": iss, "iat": now, "exp": now + exp_delta}
    if scope:
        payload["scope"] = scope
    if permissions is not None:
        payload["permissions"] = permissions
    return jwt.encode(payload, signing_key, algorithm="RS256")


class FakeEngine:
    model = "gemma3:4b"

    def health(self):
        return HealthStatus(ok=True, service_reachable=True, model_present=True,
                            model=self.model, host="http://fake")

    def generate(self, prompt, **kw):
        return GenerationResult(text="ok", model=self.model, prompt=prompt,
                                system=kw.get("system"), options={})

    def stream(self, prompt, **kw):
        from gemma_cyber.inference.engine import StreamChunk
        yield StreamChunk(request_id="rid", text="ok")
        yield StreamChunk(request_id="rid", text="", done=True)


def _jwt_client(registry=None, resolver=None):
    auth_settings = AuthSettings(domain=DOMAIN, audience=AUDIENCE, issuer=ISSUER)
    verifier = TokenVerifier(auth_settings, key_resolver=resolver or (lambda t: _PUBLIC_KEY))
    app = create_app(
        Settings(environment="dev"),
        engine=cast(InferenceEngine, FakeEngine()),
        registry=registry,
        auth_settings=auth_settings,
        verifier=verifier,
    )
    return TestClient(app)


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# -- authentication negatives -----------------------------------------------

def test_valid_token_allows_generate():
    r = _jwt_client().post("/v1/generate", json={"prompt": "hi"}, headers=_auth(make_token()))
    assert r.status_code == 200


def test_missing_token_401():
    r = _jwt_client().post("/v1/generate", json={"prompt": "hi"})
    assert r.status_code == 401


def test_malformed_token_401():
    r = _jwt_client().post("/v1/generate", json={"prompt": "hi"},
                           headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401


def test_expired_token_401():
    r = _jwt_client().post("/v1/generate", json={"prompt": "hi"},
                           headers=_auth(make_token(exp_delta=-3600)))
    assert r.status_code == 401


def test_wrong_issuer_401():
    r = _jwt_client().post("/v1/generate", json={"prompt": "hi"},
                           headers=_auth(make_token(iss="https://evil.example/")))
    assert r.status_code == 401


def test_wrong_audience_401():
    r = _jwt_client().post("/v1/generate", json={"prompt": "hi"},
                           headers=_auth(make_token(aud="https://other-api")))
    assert r.status_code == 401


def test_bad_signature_401():
    # Token forged with a different private key; resolver returns the real public key.
    r = _jwt_client().post("/v1/generate", json={"prompt": "hi"},
                           headers=_auth(make_token(signing_key=_OTHER_PRIVATE_PEM)))
    assert r.status_code == 401


def test_jwks_unavailable_503():
    def broken_resolver(_token):
        raise AuthUnavailableError("jwks down")

    r = _jwt_client(resolver=broken_resolver).post(
        "/v1/generate", json={"prompt": "hi"}, headers=_auth(make_token()))
    assert r.status_code == 503


# -- authorization on privileged endpoints ----------------------------------

def _registry(tmp_path):
    reg = ModelRegistry(tmp_path / "r.json")
    reg.register(ModelRecord(version="gemma3-cyber:v0.2", dataset_version="sft_v0.2"))
    return reg


def test_admin_requires_scope_403(tmp_path):
    client = _jwt_client(registry=_registry(tmp_path))
    # Authenticated but NO admin:models permission.
    r = client.post("/v1/admin/models/gemma3-cyber:v0.2/mark-evaluated?passed=true",
                    headers=_auth(make_token(scope="openid profile")))
    assert r.status_code == 403


def test_admin_with_scope_allows(tmp_path):
    client = _jwt_client(registry=_registry(tmp_path))
    tok = make_token(permissions=["admin:models"])
    r = client.post("/v1/admin/models/gemma3-cyber:v0.2/mark-evaluated?passed=true",
                    headers=_auth(tok))
    assert r.status_code == 200
    assert r.json()["passed_eval"] is True


def test_admin_promote_gate_returns_422(tmp_path):
    # Promotion to candidate without a passing eval must fail the registry gate.
    client = _jwt_client(registry=_registry(tmp_path))
    tok = make_token(permissions=["admin:models"])
    r = client.post("/v1/admin/models/gemma3-cyber:v0.2/promote",
                    json={"to": "candidate"}, headers=_auth(tok))
    assert r.status_code == 422


def test_admin_full_promotion_flow(tmp_path):
    client = _jwt_client(registry=_registry(tmp_path))
    tok = make_token(permissions=["admin:models"])
    assert client.post("/v1/admin/models/gemma3-cyber:v0.2/mark-evaluated?passed=true",
                       headers=_auth(tok)).status_code == 200
    assert client.post("/v1/admin/models/gemma3-cyber:v0.2/promote",
                       json={"to": "candidate"}, headers=_auth(tok)).status_code == 200
    r = client.post("/v1/admin/models/gemma3-cyber:v0.2/promote",
                    json={"to": "production"}, headers=_auth(tok))
    assert r.status_code == 200 and r.json()["stage"] == "production"


def test_scope_from_space_delimited_scope_claim(tmp_path):
    client = _jwt_client(registry=_registry(tmp_path))
    tok = make_token(scope="openid admin:models")
    r = client.post("/v1/admin/models/gemma3-cyber:v0.2/mark-evaluated?passed=true",
                    headers=_auth(tok))
    assert r.status_code == 200


# -- production fail-closed --------------------------------------------------

def test_prod_without_auth_refuses_to_start():
    with pytest.raises(RuntimeError):
        create_app(Settings(environment="prod"),
                   engine=cast(InferenceEngine, FakeEngine()))


def test_prod_with_static_token_starts():
    app = create_app(Settings(environment="prod", api_token="secret"),
                     engine=cast(InferenceEngine, FakeEngine()))
    client = TestClient(app)
    # Static mode: token required, admin still forbidden (no scopes on static).
    assert client.post("/v1/generate", json={"prompt": "hi"}).status_code == 401
    assert client.post("/v1/generate", json={"prompt": "hi"},
                       headers=_auth("secret")).status_code == 200


def test_static_mode_admin_forbidden(tmp_path):
    app = create_app(Settings(api_token="secret"),
                     engine=cast(InferenceEngine, FakeEngine()),
                     registry=_registry(tmp_path))
    client = TestClient(app)
    r = client.post("/v1/admin/models/gemma3-cyber:v0.2/mark-evaluated?passed=true",
                    headers=_auth("secret"))
    assert r.status_code == 403  # static token carries no admin scope
