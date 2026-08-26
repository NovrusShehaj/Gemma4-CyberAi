#!/usr/bin/env python3
"""Operational smoke test for the Gemma-Cyber API.

Validates a *running* deployment (or, in CI, the app in-process) end to end:
liveness, readiness, security headers, model listing, input validation, the
generate path, and — when auth is enabled — that it is actually enforced.

The checks are written against a minimal client interface (``.get``/``.post``
returning an object with ``.status_code``/``.json()``/``.headers``/``.text``), so
the SAME suite runs against a live server (via ``requests``) or the app in-process
(via FastAPI's ``TestClient`` in tests/test_operational_smoke.py) — no duplication.

Usage (against a live server):
    python scripts/smoke_test.py --base-url http://localhost:8000
    python scripts/smoke_test.py --base-url https://api.example.com --token "$TOKEN"
    python scripts/smoke_test.py --base-url ... --expect-auth   # assert 401 w/o token

Exit code 0 = all required checks passed; 1 = a required check failed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Protocol


class _Resp(Protocol):
    status_code: int
    text: str
    @property
    def headers(self) -> Any: ...
    def json(self) -> Any: ...


class _Client(Protocol):
    def get(self, url: str, **kw: Any) -> _Resp: ...
    def post(self, url: str, **kw: Any) -> _Resp: ...


@dataclass
class Result:
    name: str
    passed: bool
    detail: str = ""
    required: bool = True


def run_smoke(
    client: _Client,
    *,
    token: str | None = None,
    expect_auth: bool = False,
) -> list[Result]:
    """Run the smoke checks against ``client``; return one Result per check."""
    results: list[Result] = []

    def check(name: str, cond: bool, detail: str = "", required: bool = True) -> None:
        results.append(Result(name, bool(cond), detail, required))

    auth_headers = {"Authorization": f"Bearer {token}"} if token else {}

    # 1. Liveness
    r = client.get("/health")
    check("liveness /health == 200", r.status_code == 200, f"got {r.status_code}")
    try:
        check("liveness body ok", r.json().get("status") == "ok")
    except Exception as exc:  # noqa: BLE001
        check("liveness body ok", False, str(exc))

    # 2. Security headers on every response
    hdrs = r.headers
    check("security header CSP present", "Content-Security-Policy" in hdrs)
    check("security header X-Frame-Options DENY", hdrs.get("X-Frame-Options") == "DENY")
    check("request id header present", "X-Request-ID" in hdrs)

    # 3. Readiness (200 ready; 503 acceptable but flagged if model runtime down)
    r = client.get("/v1/ready")
    check("readiness reachable", r.status_code in (200, 503), f"got {r.status_code}")
    check("readiness == 200 (model available)", r.status_code == 200,
          f"got {r.status_code}", required=False)

    # 4. Model listing
    r = client.get("/v1/models")
    check("models listing == 200", r.status_code == 200, f"got {r.status_code}")
    try:
        check("models listing has 'models'", "models" in r.json())
    except Exception as exc:  # noqa: BLE001
        check("models listing has 'models'", False, str(exc))

    # 5. Input validation: empty + oversized prompt -> 422
    r = client.post("/v1/generate", json={"prompt": ""}, headers=auth_headers)
    check("empty prompt -> 422", r.status_code == 422, f"got {r.status_code}")
    r = client.post("/v1/generate", json={"prompt": "x" * 30000}, headers=auth_headers)
    check("oversized prompt -> 422", r.status_code == 422, f"got {r.status_code}")

    # 6. Auth enforcement (only when expected)
    if expect_auth:
        r = client.post("/v1/generate", json={"prompt": "hi"})
        check("unauthenticated generate -> 401", r.status_code == 401, f"got {r.status_code}")

    # 7. Happy-path generate (requires a token if auth is on; skip if none and expect_auth)
    if token or not expect_auth:
        r = client.post("/v1/generate", json={"prompt": "In one word, say ok."},
                        headers=auth_headers)
        check("generate happy path -> 200", r.status_code == 200,
              f"got {r.status_code}", required=not expect_auth)

    # 8. Web page served
    r = client.get("/")
    check("web UI served", r.status_code == 200 and "Gemma-Cyber" in r.text)

    return results


def render(results: list[Result]) -> tuple[str, bool]:
    lines = []
    ok = True
    for res in results:
        mark = "PASS" if res.passed else ("FAIL" if res.required else "WARN")
        if not res.passed and res.required:
            ok = False
        suffix = f"  ({res.detail})" if res.detail and not res.passed else ""
        lines.append(f"[{mark}] {res.name}{suffix}")
    return "\n".join(lines), ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gemma-Cyber API operational smoke test.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--token", default=None, help="Bearer token for a protected API.")
    parser.add_argument("--expect-auth", action="store_true",
                        help="Assert the API rejects unauthenticated generate (401).")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    import requests

    class Session:
        def __init__(self, base: str, timeout: float) -> None:
            self.base = base.rstrip("/")
            self.timeout = timeout
            self.s = requests.Session()

        def get(self, url: str, **kw: Any) -> Any:
            return self.s.get(self.base + url, timeout=self.timeout, **kw)

        def post(self, url: str, **kw: Any) -> Any:
            return self.s.post(self.base + url, timeout=self.timeout, **kw)

    try:
        results = run_smoke(Session(args.base_url, args.timeout),
                            token=args.token, expect_auth=args.expect_auth)
    except Exception as exc:  # noqa: BLE001
        print(f"SMOKE TEST ABORTED: could not reach {args.base_url}: {exc}", file=sys.stderr)
        return 1

    report, ok = render(results)
    print(report)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
