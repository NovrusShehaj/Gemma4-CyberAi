# Security & Safety

This is a cybersecurity AI product, so security is a product requirement, not an
afterthought. This document states what is implemented today and what is
deliberately deferred. Safe-by-default is the rule: every risky capability is off
until explicitly enabled.

## Product scope (safe by default)

The product is **defensive and authorized-use only**: education, defensive
security, authorized testing, CTF/lab environments, and owned systems. It is a
question-answering assistant. It does **not**, and by design must not:

- interact with live targets, scan, or exploit anything;
- execute commands, spawn tools, or run agents;
- fetch untrusted URLs or act autonomously.

There is no tool/agent layer. If one is ever added, it must ship with the
controls listed in "Future tool/agent controls" below — not before.

## Model-behaviour safety

- A safety-forward **system prompt** (shared by the CLI, API, and eval harness)
  instructs the model to reason from evidence, flag insufficient evidence rather
  than guess, and not fabricate CVEs/tool output/facts.
- The evaluation suite explicitly measures **hallucination resistance** and
  **insufficient-evidence recognition**, and a `factual` scorer hard-fails
  forbidden/wrong ATT&CK IDs. Model outputs are treated as potentially unreliable
  by design — the UI warns users to verify before acting.

## Transport / application controls (implemented)

| Control | Status | Notes |
|---|---|---|
| Input validation + size bounds | ✅ | pydantic; prompt ≤ 24k chars → 422 |
| Bearer-token auth | ✅ (opt-in) | `GEMMA_CYBER_API_TOKEN`; constant-time compare |
| Rate limiting | ✅ (opt-in) | in-process fixed window; per-token when authed |
| Security headers + CSP | ✅ | CSP, `X-Frame-Options: DENY`, nosniff, referrer, COOP |
| CORS allowlist | ✅ | empty = same-origin only |
| Structured errors (no stack leaks) | ✅ | typed → 401/422/429/503/504 |
| Request IDs | ✅ | `X-Request-ID` on every response |
| Runtime isolation | ✅ | Ollama not published; API binds localhost by default |
| Non-root container | ✅ | Dockerfile runs as `app` user |
| No secrets in repo | ✅ | all config via env; `.gitignore` blocks `.env`, tokens |

## Prompt-injection posture

The only untrusted input is the user's prompt, and the model has **no tools or
external actions** to hijack, so prompt injection cannot cause it to *do*
anything — the blast radius is limited to the text it returns. That property must
be preserved: do not add tools/RAG-over-untrusted-content without an injection
review and the boundary controls below.

## Secrets

- Never committed. `.gitignore` blocks `.env*`, `*.token`, `.netrc`, key files.
- Read from the environment at use time; the Settings `redacted()` view is the
  single chokepoint for anything logged.
- The user's email/identity is never sent to third-party services.

## Dependency & supply-chain

CI runs a dedicated **`security`** job on every push/PR (`.github/workflows/ci.yml`):

| Scan | Tool | Gate |
|---|---|---|
| Python static analysis | `bandit -r src -ll` (medium+) | fail on medium/high |
| Dependency CVEs | `pip-audit --skip-editable` | fail on any known vuln |
| Secret scanning (full history) | `gitleaks-action@v2` | fail on any finding |

The lint job also runs ruff + mypy + the test suite (incl. the auth negative tests,
which are the primary auth-control validation).

### Security baseline (2026-08-26)

- **bandit (medium+):** 0 findings. (3 low-severity `B101` "assert used" in eval
  scorers are intentional invariants; not run at the CI gate's `-ll` level.)
- **pip-audit:** no known vulnerabilities in dependencies.
- **gitleaks:** no committed secrets (the auth tests generate an RS256 keypair at
  runtime; none is stored).

**Policy:** do not blanket-suppress findings to make CI green. A justified exception
must be documented here with the finding id, reason, and review date. Image scanning
(Trivy/grype) is recommended before publishing a container image and is not yet wired.

**gitleaks note:** the action is free for personal GitHub accounts. Organization
accounts require a `GITLEAKS_LICENSE` secret; set it or swap for `trufflehog` if the
repo moves under an org.

## Logging & privacy

See `docs/operations.md`. In short: logs carry request ids, model tags, timings,
and error types — **not** prompt/response content by default. There is no user
account store yet, so there is no PII at rest.

## Future tool/agent controls (required if ever added — do not build prematurely)

Default-deny target allowlists · sandboxing · network isolation · human approval ·
credential isolation · immutable audit logging · kill switch · disposable
environments · rate limits · explicit written authorization. Until all of these
exist, the product stays a no-tools assistant.

## Reporting

For a suspected vulnerability, describe the class of issue privately to the
maintainer; do not file a public issue with a working exploit.
