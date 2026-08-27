# Authentication & Authorization (Auth0 / JWT)

The API's trust boundary. Three modes, chosen by configuration:

| Mode | When | Who is trusted |
|---|---|---|
| **JWT (Auth0)** | `GEMMA_CYBER_AUTH_DOMAIN` + `GEMMA_CYBER_AUTH_AUDIENCE` set | Callers with a valid Auth0 access token |
| **static (dev)** | no JWT config but `GEMMA_CYBER_API_TOKEN` set | Callers presenting the shared token (never admin) |
| **open (dev)** | neither set | Anonymous (local development only) |

**Production fails closed:** with `GEMMA_CYBER_ENV=prod` and no auth configured,
the app **refuses to start**. This prevents an accidentally-public unauthenticated
API. Static-token mode in prod logs a warning recommending Auth0.

## What is validated (JWT mode)

A token is accepted only if **all** hold (`gemma_cyber.api.auth.TokenVerifier`):
- **Signature** verifies against the tenant JWKS (RS256). Unknown `kid` triggers a
  JWKS refresh, so **key rotation** is handled automatically.
- **Issuer** (`iss`) equals the configured issuer.
- **Audience** (`aud`) equals the configured API audience.
- **Expiry** (`exp`) is in the future (±`leeway`, default 60s); `iat` present.
- **Required claims** present: `exp, iat, iss, aud, sub`.

Tokens are **never logged**. Auth failures log the reason + request id, not the token.

## Authorization (server-side, never client-trusted)

Permissions come only from the **signed** token — the `scope` string and Auth0's
`permissions` array are merged into the caller's scope set. Privileged endpoints
require a permission:

| Endpoint | Requires |
|---|---|
| `POST /v1/generate` | authenticated (any valid token) |
| `POST /v1/admin/models/register` | `admin:models` |
| `POST /v1/admin/models/{version}/mark-evaluated` | `admin:models` |
| `POST /v1/admin/models/{version}/promote` | `admin:models` |
| `GET /v1/models`, `GET /v1/ready`, `GET /health`, `GET /`, `GET /config.json` | public (probe-safe) |
| `GET /docs`, `/redoc`, `/openapi.json` | **disabled in hosted mode** (staging/prod) |

A valid token **without** `admin:models` gets **403** on admin routes. The static
dev token carries no scopes, so it can chat but never administer models.

## Environment variables

| Variable | Example | Meaning |
|---|---|---|
| `GEMMA_CYBER_AUTH_DOMAIN` | `your-tenant.eu.auth0.com` | Auth0 tenant domain |
| `GEMMA_CYBER_AUTH_AUDIENCE` | `https://api.gemma-cyber` | API identifier (audience) |
| `GEMMA_CYBER_AUTH_ISSUER` | `https://your-tenant.eu.auth0.com/` | Optional; defaults to `https://<domain>/` |
| `GEMMA_CYBER_AUTH_JWKS_URL` | *(derived)* | Optional; defaults to `https://<domain>/.well-known/jwks.json` |
| `GEMMA_CYBER_AUTH_ALGORITHMS` | `RS256` | Signing algorithms (comma-separated) |
| `GEMMA_CYBER_AUTH_LEEWAY` | `60` | Clock-skew seconds |
| `GEMMA_CYBER_WEB_AUTH0_CLIENT_ID` | `abc123…` | **Public** SPA client id for the browser login flow (never a secret) |

> In hosted mode, if JWT auth is on but `GEMMA_CYBER_WEB_AUTH0_CLIENT_ID` is
> **unset**, the shipped browser UI cannot sign in and every `/v1/generate` returns
> 401 (the server logs a warning at startup). Non-browser clients are unaffected.

## Auth0 dashboard configuration (external — cannot be done from the repo)

1. **Create an API** in Auth0 → Applications → APIs, Identifier =
   `GEMMA_CYBER_AUTH_AUDIENCE` (e.g. `https://api.gemma-cyber`), signing alg RS256.
2. **Enable RBAC** on that API and **"Add Permissions in the Access Token"** so the
   `permissions` claim is populated.
3. **Define a permission** `admin:models` on the API; assign it to an admin role;
   assign the role to admin users.
4. Create a **Single-Page Application** for the web UI (public client, PKCE — no
   client secret). Per environment, set **exact HTTPS** values (no wildcards):
   - **Allowed Callback URLs**: your UI origin + `/` (e.g. `https://app.example.com/`)
   - **Allowed Logout URLs**: same origin + `/`
   - **Allowed Web Origins**: your UI origin (e.g. `https://app.example.com`)
   Copy the SPA **Client ID** into `GEMMA_CYBER_WEB_AUTH0_CLIENT_ID` (public config).
   For non-browser service clients use an M2M application instead.
5. Set `GEMMA_CYBER_AUTH_DOMAIN` + `GEMMA_CYBER_AUTH_AUDIENCE` (+ issuer if custom)
   and `GEMMA_CYBER_WEB_AUTH0_CLIENT_ID` in the API service environment. Store only
   these (all non-secret) — a SPA has **no** client secret.
6. Configure tenant security policy (session lifetime, idle timeout, MFA / adaptive
   protection, breached-password protection) per your organization — these are
   tenant values not inferable from source. Assign `admin:models` to a tightly
   controlled admin role only; use a separate non-admin account for staging tests.

Everything below the dashboard — token validation, claim checks, rotation,
authorization — is implemented and tested in-repo (`tests/test_api_auth.py`, using
a self-signed RS256 keypair + mocked JWKS, so no live tenant is needed for CI).

## Testing (what the suite proves, no live tenant)

`tests/test_api_auth.py` covers: valid token → 200; missing / malformed / expired /
wrong-issuer / wrong-audience / bad-signature → 401; JWKS unavailable → 503;
authenticated-but-missing-scope → 403; `admin:models` → 200; the gated promotion
flow; scopes from both `scope` and `permissions` claims; prod fail-closed; and that
the static token cannot administer models.

## Web UI (Authorization Code + PKCE — implemented)

The shipped SPA (`web/app.js`) implements the browser flow directly (no third-party
SDK, so CSP `script-src` stays `'self'`):

1. On load it fetches `/config.json` (public: env, model, and `auth.{enabled,
   domain, clientId, audience}` — never a secret).
2. If auth is enabled and there is no session, it shows a **sign-in gate**; the API
   requires a token so no generation is possible signed-out.
3. **Sign in** generates a PKCE verifier + `state`, redirects to Auth0
   `/authorize`. Only the one-time verifier/state transit `sessionStorage`; they
   are deleted the moment the callback is handled.
4. The `?code=&state=` callback is exchanged at Auth0 `/oauth/token` for an access
   token held **in memory only** — never `localStorage`, never a URL, never logged.
   The code/state are stripped from the URL immediately.
5. The token is attached as `Authorization: Bearer …` to same-origin `/v1/*` calls
   only. On **401** the UI drops the token and returns to signed-out; **403/429/
   503/504** render distinct, safe messages with the request id.
6. **Sign out** clears in-memory state and redirects to Auth0 `/v2/logout`.

CSP is widened *only* to the Auth0 origin, and *only* for `connect-src`/`form-action`
(the token exchange + login redirect), when `GEMMA_CYBER_WEB_AUTH0_CLIENT_ID` is set.
Scripts are never widened. Model output is rendered with `textContent` exclusively.

**Server-side enforcement remains the authority** — the browser never asserts a
role; `admin:models` and token validity are decided from the signed token on the
server. A full live-tenant browser E2E (real login/callback/logout) requires a
staging Auth0 SPA and a browser runner; the source-level contract is covered by
`tests/test_web.py` (config shape, CSP posture, no-innerHTML, no token persistence).

## Generation policy (hosted vs self-host)

In hosted mode (`staging`/`prod`) the server **owns the safety/system prompt and the
set of servable models**: a client-supplied `system` is dropped, and a client
`model` must resolve to a registered version or stage alias (an arbitrary Ollama
tag is rejected with 400). Set `GEMMA_CYBER_ALLOW_CLIENT_OVERRIDES=true` to permit
client `system`/arbitrary `model` (self-host / internal expert use only).
