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

The **hosted API, web UI, and `gemma-cyber` operator CLI** have no tool/agent
layer. That property is load-bearing: prompt injection on those surfaces can
only change text. Do not add tools to FastAPI “for convenience.”

A **local, opt-in** terminal agent (`gemma4`, extra `[agent]`) **is now
implemented** — see [`docs/agent.md`](agent.md) and
[`GEMMA4_TERMINAL_AGENT_IMPLEMENTATION_PLAN.md`](../GEMMA4_TERMINAL_AGENT_IMPLEMENTATION_PLAN.md).
It is the only surface in this repository that executes tools, and it changes
nothing about the hosted product:

- `gemma_cyber.api` has **zero** imports of `gemma_cyber.agent`, enforced by
  `tests/agent/test_import_boundary.py` (static scan **and** a clean-subprocess
  import probe).
- Installing the base package does not install the agent’s dependencies. A base
  install has a `gemma4` binary that prints how to install the extra and exits.
- The agent is local-only: no cloud sync, no telemetry, no network tools.
- Default mode is `read-only`: it cannot write a file or run a command.

Installing `[agent]` is the explicit decision that expands prompt-injection blast
radius from text to actions. Read `docs/agent.md` before doing it.

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
| Authentication (Auth0 JWT) | ✅ | RS256 + JWKS rotation, iss/aud/exp/claims; see `docs/auth.md` |
| Authorization (server-side) | ✅ | scopes from signed token; admin routes require `admin:models` |
| Prod fail-closed | ✅ | `GEMMA_CYBER_ENV=prod` with no auth configured refuses to start; hosted + writable registry also requires auth |
| Static dev token (fallback) | ✅ (opt-in) | `GEMMA_CYBER_API_TOKEN`; constant-time; never admin |
| Rate limiting | ✅ | hosted default 60/min (dev 0); 429 includes `Retry-After`; edge `limit_req` must cover 401s |
| Security headers + CSP | ✅ | CSP, `X-Frame-Options: DENY`, nosniff, referrer, COOP |
| CORS allowlist | ✅ | empty = same-origin only |
| Structured errors (no stack leaks) | ✅ | generate/SSE/ready/admin omit runtime URLs and filesystem paths |
| Request IDs | ✅ | `X-Request-ID` on every response |
| Runtime isolation | ✅ | Ollama not published; API binds localhost by default |
| Non-root container | ✅ | Dockerfile runs as `app` user |
| No secrets in repo | ✅ | all config via env; `.gitignore` blocks `.env`, tokens |

## Prompt-injection posture

The only untrusted input is the user's prompt, and the model has **no tools or
external actions** to hijack, so prompt injection cannot cause it to *do*
anything — the blast radius is limited to the text it returns. That property must
be preserved on the hosted path: do not add tools/RAG-over-untrusted-content to
the API or `gemma-cyber` without an injection review and the boundary controls
below. A future `gemma4` agent expands the blast radius to actions and is a
separate, fail-closed design (see the terminal-agent spec).

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

## Local agent controls (implemented — `gemma4` only)

Full detail in [`docs/agent.md`](agent.md). Summary of what is enforced in code,
with the tests that hold each one:

| Control | Where | Tests |
|---|---|---|
| Four permission modes, default `read-only` | `agent/permissions.py` | `test_permissions.py` (mode × side-effect matrix) |
| Workspace jail after `realpath`; symlink escape denied | `agent/workspace.py` | `test_workspace_jail.py` |
| Credential paths denied in **every** mode incl. `trusted` | `agent/permissions.py` | `test_permissions.py`, `adversarial/` |
| Destructive-command denylist with no approval path | `agent/permissions.py` | `adversarial/test_command_policy.py` |
| Child-process environment built from an allowlist | `agent/tools/shell.py` | `test_tools_shell.py` |
| Process-group kill on timeout/cancel | `agent/tools/shell.py` | `test_tools_shell.py` |
| ANSI/OSC stripped before the terminal **and** before re-injection | `agent/sanitize.py` | `test_sanitize.py`, `adversarial/` |
| Hash-checked edits; stale hash never writes | `agent/tools/fs.py` | `test_tools_fs.py` |
| Network denied by default; metadata/RFC1918 blocked | `agent/permissions.py` | `test_permissions.py` |
| Workspace content is data, never policy | `agent/context.py` | `adversarial/test_prompt_injection.py` |
| Project config cannot set `trusted` or hold credentials | `agent/config.py` | `test_config.py` |
| Audit log with no prompts, bodies, or secrets | `agent/audit.py` | `adversarial/test_command_policy.py` |
| Hosted API imports no agent code | — | `test_import_boundary.py` |

**What this does not solve.** Prompt injection is mitigated, not eliminated. The
layering (system policy, process-level mode, `<untrusted>` wrappers, schema
validation, path jail, ANSI stripping, default read-only, audit) means an
injected instruction cannot escalate privilege — but a model in `--mode agent`
can still be talked into a *permitted* action that is unwise. Default `read-only`
is the control that actually bounds this; treat `agent` and `trusted` as you
would treat handing someone your shell.

**`--mode trusted`** requires `--i-accept-risk` and a TTY, is refused with
`--json`, and cannot be set by any config file. It disables confirmation
prompts only; credential paths and destructive commands remain denied.

## Historic note: controls required before tools existed

Default-deny target allowlists · sandboxing · network isolation · human approval ·
credential isolation · audit logging · kill switch · rate limits · explicit
written authorization. These were the preconditions for building any tool layer.
They are now met for the **local** agent as tabulated above. The **hosted**
product remains a no-tools assistant and is not covered by, or eligible for,
this capability.

## Reporting

For a suspected vulnerability, describe the class of issue privately to the
maintainer; do not file a public issue with a working exploit.
