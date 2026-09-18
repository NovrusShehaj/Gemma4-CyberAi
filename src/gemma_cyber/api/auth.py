"""Production identity & authorization for the API (Auth0 / OIDC JWT).

This is the trust boundary. An access token is accepted only if ALL hold:
  * signature verifies against the tenant's JWKS (RS256), with automatic key
    rotation (unknown ``kid`` triggers a JWKS refresh);
  * ``iss`` matches the configured issuer;
  * ``aud`` matches the configured API audience (identifier);
  * ``exp`` is in the future (and ``iat`` present);
  * the required baseline claims are present.

Authorization is SEPARATE from authentication: a verified token still must carry
the required scope/permission for a privileged operation. Roles/permissions are
read ONLY from the signed token (`scope` string + Auth0 `permissions` array),
never from client-supplied fields.

Design notes:
  * Tokens are NEVER logged. Errors are typed so the API maps token problems to
    401, missing-permission to 403, and JWKS/network problems to 503 (a transient
    infra issue, not an auth decision).
  * The verifier accepts an injected key resolver so the full path is unit-tested
    with an in-test RS256 keypair and a mocked JWKS — no live tenant required.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gemma_cyber.inference.config import ConfigError

# PyJWT is an optional (api-extra) dependency; import lazily so the core package
# and non-auth API modes don't require it.

# Hosted/prod JWT: asymmetric only. HS256 would let anyone who learns a shared
# secret forge tokens; it is rejected at startup when settings.hosted is true.
HOSTED_JWT_ALGORITHMS: frozenset[str] = frozenset({"RS256"})
MAX_AUTH_LEEWAY_S = 120


class AuthError(Exception):
    """Token is missing, malformed, expired, or fails a claim/signature check -> 401."""


class AuthForbiddenError(Exception):
    """Token is valid but lacks the required scope/permission -> 403."""


class AuthUnavailableError(Exception):
    """The identity provider / JWKS could not be reached to make a decision -> 503."""


@dataclass(frozen=True)
class AuthSettings:
    """Auth0/OIDC configuration, from ``GEMMA_CYBER_AUTH_*`` env vars.

    ``enabled`` is true only when both a domain and an audience are set — that is
    the switch between "verify real JWTs" and "no JWT auth configured".
    """

    domain: str = ""
    audience: str = ""
    issuer: str = ""
    jwks_url: str = ""
    algorithms: tuple[str, ...] = ("RS256",)
    leeway: int = 60  # seconds of clock skew allowed on exp/iat

    @property
    def enabled(self) -> bool:
        return bool(self.domain and self.audience)

    def resolved_issuer(self) -> str:
        return self.issuer or f"https://{self.domain}/"

    def resolved_jwks_url(self) -> str:
        return self.jwks_url or f"https://{self.domain}/.well-known/jwks.json"

    def validate(self, *, hosted: bool = False) -> AuthSettings:
        """Fail fast on JWT config that is unsafe for the deployment mode.

        Dev may use looser settings (documented in docs/auth.md). Hosted/prod
        requires RS256 and HTTPS issuer/JWKS. Always caps clock-skew leeway.
        """
        if self.leeway < 0 or self.leeway > MAX_AUTH_LEEWAY_S:
            raise ConfigError(
                f"GEMMA_CYBER_AUTH_LEEWAY must be between 0 and {MAX_AUTH_LEEWAY_S}"
            )
        if not self.enabled:
            return self
        if not self.algorithms:
            raise ConfigError("GEMMA_CYBER_AUTH_ALGORITHMS must not be empty")
        if hosted:
            extra = [a for a in self.algorithms if a not in HOSTED_JWT_ALGORITHMS]
            if extra:
                raise ConfigError(
                    "hosted JWT algorithms must be RS256 only, "
                    f"got {self.algorithms}"
                )
            if "://" in self.domain or "/" in self.domain:
                raise ConfigError(
                    "GEMMA_CYBER_AUTH_DOMAIN must be a hostname (no scheme or path)"
                )
            issuer = self.resolved_issuer()
            jwks = self.resolved_jwks_url()
            if not issuer.startswith("https://"):
                raise ConfigError(
                    "GEMMA_CYBER_AUTH_ISSUER must be an https URL in hosted mode"
                )
            if not jwks.startswith("https://"):
                raise ConfigError(
                    "GEMMA_CYBER_AUTH_JWKS_URL must be an https URL in hosted mode"
                )
        return self

    @classmethod
    def from_env(cls) -> AuthSettings:
        def _e(name: str, default: str = "") -> str:
            return os.environ.get("GEMMA_CYBER_AUTH_" + name, default) or default

        algos = _e("ALGORITHMS", "RS256")
        raw_leeway = _e("LEEWAY", "60") or "60"
        try:
            leeway = int(raw_leeway)
        except ValueError:
            raise ConfigError(
                f"GEMMA_CYBER_AUTH_LEEWAY must be an integer, got {raw_leeway!r}"
            ) from None
        return cls(
            domain=_e("DOMAIN"),
            audience=_e("AUDIENCE"),
            issuer=_e("ISSUER"),
            jwks_url=_e("JWKS_URL"),
            algorithms=tuple(a.strip() for a in algos.split(",") if a.strip()),
            leeway=leeway,
        )


@dataclass
class Principal:
    """The authenticated caller, derived only from verified token claims."""

    subject: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    claims: dict[str, Any] = field(default_factory=dict)
    # How the caller authenticated: "jwt", "static" (dev token), or "anonymous".
    method: str = "jwt"

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def has_all(self, scopes: tuple[str, ...]) -> bool:
        return all(s in self.scopes for s in scopes)

    @classmethod
    def from_claims(cls, claims: dict[str, Any]) -> Principal:
        scopes: set[str] = set()
        raw_scope = claims.get("scope")
        if isinstance(raw_scope, str):
            scopes.update(s for s in raw_scope.split() if s)
        perms = claims.get("permissions")
        if isinstance(perms, list):
            scopes.update(str(p) for p in perms)
        return cls(
            subject=str(claims.get("sub", "")),
            scopes=frozenset(scopes),
            claims=claims,
            method="jwt",
        )


# A subject id for callers authenticated by the dev static token (never admin).
STATIC_PRINCIPAL_SUBJECT = "static-token"
# Subject used only when auth is fully disabled (dev/open mode).
ANON_PRINCIPAL_SUBJECT = "anonymous"


class TokenVerifier:
    """Verifies Auth0 access tokens. Inject ``key_resolver`` in tests."""

    def __init__(
        self,
        settings: AuthSettings,
        *,
        key_resolver: Callable[[str], Any] | None = None,
    ) -> None:
        self.settings = settings
        self._key_resolver = key_resolver
        self._jwk_client: Any | None = None

    def _signing_key(self, token: str) -> Any:
        if self._key_resolver is not None:
            return self._key_resolver(token)
        # Lazily build a caching JWKS client (handles kid lookup + rotation).
        if self._jwk_client is None:
            try:
                from jwt import PyJWKClient
            except ImportError as exc:  # pragma: no cover - api extra guarantees it
                raise AuthUnavailableError(
                    "PyJWT not installed; install the 'api' extra"
                ) from exc
            self._jwk_client = PyJWKClient(
                self.settings.resolved_jwks_url(), cache_keys=True
            )
        try:
            from jwt import PyJWKClientError

            return self._jwk_client.get_signing_key_from_jwt(token).key
        except PyJWKClientError as exc:
            # Unknown kid / JWKS fetch problem: an infra/availability condition.
            raise AuthUnavailableError(f"could not resolve signing key: {exc}") from exc
        except Exception as exc:  # network/parse
            raise AuthUnavailableError(f"JWKS unavailable: {exc}") from exc

    def verify(self, token: str) -> Principal:
        import jwt

        if not token:
            raise AuthError("missing token")
        try:
            key = self._signing_key(token)
            claims = jwt.decode(
                token,
                key,
                algorithms=list(self.settings.algorithms),
                audience=self.settings.audience,
                issuer=self.settings.resolved_issuer(),
                leeway=self.settings.leeway,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthError("token expired") from exc
        except jwt.InvalidAudienceError as exc:
            raise AuthError("invalid audience") from exc
        except jwt.InvalidIssuerError as exc:
            raise AuthError("invalid issuer") from exc
        except jwt.MissingRequiredClaimError as exc:
            raise AuthError(f"missing required claim: {exc}") from exc
        except jwt.InvalidSignatureError as exc:
            raise AuthError("invalid signature") from exc
        except AuthError:
            raise
        except AuthUnavailableError:
            raise
        except jwt.InvalidTokenError as exc:
            raise AuthError(f"invalid token: {exc}") from exc
        return Principal.from_claims(claims)


def bearer_token(authorization: str | None) -> str | None:
    """Extract the token from an ``Authorization: Bearer <token>`` header."""
    if not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    return authorization[len(prefix):].strip() or None
