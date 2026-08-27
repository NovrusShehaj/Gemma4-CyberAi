"""Security primitives for the API: auth, rate limiting, security headers.

All are safe-by-default and dependency-free (no Redis, no external service):
  * Bearer-token auth is OFF unless ``GEMMA_CYBER_API_TOKEN`` is set. When set,
    it is compared in constant time.
  * The rate limiter is an in-process token bucket keyed by client. It is OFF
    (limit 0) by default and intended for a single-instance deployment; a
    multi-instance deployment would move this to a shared store.
  * Security headers + a strict CSP are always applied.

These implement the Phase 9 controls that belong at the transport edge; model
safety (system prompt, refusal behavior) lives in the inference layer and the
model itself.
"""

from __future__ import annotations

import hmac
import re
import time
import uuid
from collections import defaultdict

# A request/correlation id we are willing to echo into logs and responses. We
# accept a client-supplied ``X-Request-ID`` only if it matches this — otherwise a
# caller could inject newlines (log forging) or an unbounded string (memory abuse).
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def safe_request_id(supplied: str | None) -> str:
    """Return a safe correlation id: the client's if well-formed, else a fresh one."""
    if supplied and _REQUEST_ID_RE.match(supplied):
        return supplied
    return uuid.uuid4().hex[:12]


class Capacity:
    """Bounded admission control for concurrent generations (single process).

    ``limit <= 0`` disables the bound (unlimited). :meth:`acquire` is non-blocking
    and returns False immediately when at capacity, giving the caller a
    deterministic reject (503) instead of queueing unboundedly and starving the
    single Ollama host. Integer inc/dec is safe under the GIL for one process; a
    multi-instance deployment would move this to a shared limiter.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._active = 0

    @property
    def active(self) -> int:
        return self._active

    def acquire(self) -> bool:
        if self.limit <= 0:
            self._active += 1
            return True
        if self._active >= self.limit:
            return False
        self._active += 1
        return True

    def release(self) -> None:
        if self._active > 0:
            self._active -= 1


def token_matches(expected: str, provided: str | None) -> bool:
    """Constant-time bearer-token check. Empty expected -> auth disabled (allow)."""
    if not expected:
        return True
    if not provided:
        return False
    prefix = "Bearer "
    if not provided.startswith(prefix):
        return False
    return hmac.compare_digest(expected, provided[len(prefix):])


class RateLimiter:
    """Fixed-window in-memory rate limiter (requests per 60s per key).

    ``limit <= 0`` disables limiting. Not shared across processes; adequate for a
    single-instance service, which is the documented deployment target.
    """

    def __init__(self, limit_per_min: int, *, now=time.monotonic) -> None:
        self.limit = limit_per_min
        self._now = now
        self._hits: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = self._now()
        window_start = now - 60.0
        hits = [t for t in self._hits[key] if t >= window_start]
        if len(hits) >= self.limit:
            self._hits[key] = hits
            return False
        hits.append(now)
        self._hits[key] = hits
        return True


def content_security_policy(auth0_domain: str = "") -> str:
    """Build the CSP. Scripts are always same-origin only (the core XSS control).

    ``script-src 'self'`` is never relaxed — no third-party or inline scripts. When
    an Auth0 domain is configured for the browser SPA, only ``connect-src`` and
    ``form-action`` are widened to that exact HTTPS origin so the PKCE token
    exchange (fetch) and the login redirect are permitted; nothing else changes.
    ``style-src`` keeps ``'unsafe-inline'`` for the composer's dynamic auto-grow
    height only — styles, never scripts.
    """
    connect = "connect-src 'self'"
    form_action = "form-action 'self'"
    if auth0_domain:
        origin = f"https://{auth0_domain}"
        connect = f"connect-src 'self' {origin}"
        form_action = f"form-action 'self' {origin}"
    return (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        f"{connect}; "
        "img-src 'self' data:; "
        "base-uri 'none'; "
        f"{form_action}; "
        "frame-ancestors 'none'"
    )


def security_headers(auth0_domain: str = "") -> dict[str, str]:
    """The full set of response security headers, with a CSP tuned to the SPA."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": content_security_policy(auth0_domain),
        "Cross-Origin-Opener-Policy": "same-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }


# Default header set (no browser Auth0 configured). Kept as a module constant for
# back-compat; ``create_app`` uses ``security_headers(domain)`` when the SPA needs
# the Auth0 origin in connect-src.
CONTENT_SECURITY_POLICY = content_security_policy()
SECURITY_HEADERS = security_headers()
