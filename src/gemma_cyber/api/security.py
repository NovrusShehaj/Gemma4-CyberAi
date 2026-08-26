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
import time
from collections import defaultdict


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


# A conservative CSP for the self-contained chat page: same-origin only, inline
# styles allowed (the page ships a <style> block), no third-party anything.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; "
    "img-src 'self' data:; "
    "base-uri 'none'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}
