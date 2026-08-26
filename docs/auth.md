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
| `GET /v1/models`, `GET /v1/ready`, `GET /health`, `GET /` | public |

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

## Auth0 dashboard configuration (external — cannot be done from the repo)

1. **Create an API** in Auth0 → Applications → APIs, Identifier =
   `GEMMA_CYBER_AUTH_AUDIENCE` (e.g. `https://api.gemma-cyber`), signing alg RS256.
2. **Enable RBAC** on that API and **"Add Permissions in the Access Token"** so the
   `permissions` claim is populated.
3. **Define a permission** `admin:models` on the API; assign it to an admin role;
   assign the role to admin users.
4. Create an **Application** (SPA for the web UI, or M2M for service clients) and
   authorize it for the API.
5. Set `GEMMA_CYBER_AUTH_DOMAIN` + `GEMMA_CYBER_AUTH_AUDIENCE` (+ issuer if custom)
   in the API service environment.

Everything below the dashboard — token validation, claim checks, rotation,
authorization — is implemented and tested in-repo (`tests/test_api_auth.py`, using
a self-signed RS256 keypair + mocked JWKS, so no live tenant is needed for CI).

## Testing (what the suite proves, no live tenant)

`tests/test_api_auth.py` covers: valid token → 200; missing / malformed / expired /
wrong-issuer / wrong-audience / bad-signature → 401; JWKS unavailable → 503;
authenticated-but-missing-scope → 403; `admin:models` → 200; the gated promotion
flow; scopes from both `scope` and `permissions` claims; prod fail-closed; and that
the static token cannot administer models.

## Web UI note

The SPA obtains a token via Auth0 (Authorization Code + PKCE) and sends it as
`Authorization: Bearer <token>` to the API. The **client secret is never in
frontend code** (PKCE public client). The current shipped `web/` page is an
unauthenticated local dev UI; wiring Auth0 login into it is the remaining web task
(see the plan's web section) and does not change the server-side enforcement, which
is already the authority.
