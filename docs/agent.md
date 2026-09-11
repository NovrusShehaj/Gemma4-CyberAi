# `gemma4` — the local terminal agent

`gemma4` is an **opt-in, local-only** coding and defensive-security agent. It is
the only part of this repository that executes tools. The hosted API, the web UI,
and the `gemma-cyber` operator CLI remain no-tools question-answering surfaces
and are unaffected by installing it.

Design contract: [`GEMMA4_TERMINAL_AGENT_IMPLEMENTATION_PLAN.md`](../GEMMA4_TERMINAL_AGENT_IMPLEMENTATION_PLAN.md).
Security posture: [`docs/security.md`](security.md).

---

## Honesty first

- **Answers are unverified.** The agent will state this on every start. Verify
  technique IDs, CVEs, and anything you are going to act on.
- **The default model is `gemma3:4b` — a general base model**, not a promoted
  cybersecurity model. `gemma3-cyber:v0.2` failed MODEL-GATE and remains
  experimental; building this agent does not change that. See
  [`docs/model-card.md`](model-card.md).
- **A known failure of this model family is mislabelling Kerberoasting as
  T1060** (it is T1558.003). The agent's system policy says so to the model.
- **A 4B model calls tools unreliably — measured, not assumed.** On a local
  Ollama daemon (2026-09-11):

  | Model | Task | Result |
  |---|---|---|
  | `gemma3:4b` | read a file and answer | worked (valid `<tool_call>`, 2 iterations) |
  | `gemma3:4b` | run `sec.secrets` and report | worked |
  | `gemma3:4b` | read then `fs.edit` a one-line bug | **failed** — emitted ```` ```tool_call> ```` instead of `<tool_call>`, twice, including after being handed the exact syntax |
  | `gemma4:12b` | read then `fs.edit` the same bug | worked (2 tool calls, 3 iterations, correct patch) |

  The codec refused the malformed envelope rather than guessing, which is the
  designed behaviour: a parser that executes things *resembling* a tool call is
  the entire attack surface. The framework is sound; the 4B model is the weak
  part. **For `--mode agent` or any editing work, use a larger local model**
  (`--model gemma4:12b`) or an OpenAI-compatible endpoint. `gemma4 models`
  reports what a provider actually claims rather than what would be convenient.

---

## Install

```bash
uv pip install -e '.[agent]'        # from a checkout
pipx install 'gemma-cyber[agent]'   # from a release
```

Installing the base package does **not** enable tools. A base install still has a
`gemma4` binary; it prints how to install the extra and exits non-zero.

Platforms: **macOS and Linux are supported and tested.** Windows is P1 and not
yet exercised — process-group termination and the `/bin/sh` path are POSIX
assumptions; the shell tests skip there.

Requires a running Ollama (`ollama serve`) with a pulled model, or any
OpenAI-compatible endpoint. Check with `gemma4 doctor`.

---

## Commands

```text
gemma4                        interactive agent (mode from config; default read-only)
gemma4 ask "..."              one-shot chat; no tools, no session
gemma4 run "..."              one-shot agent loop; --json / --jsonl for CI
gemma4 resume [id]            newest session for this workspace if id is omitted
gemma4 session ls|show|rm
gemma4 models                 providers and their reported capabilities
gemma4 tools                  tools and what the current mode allows
gemma4 doctor                 runtime, config, workspace, ignore files
gemma4 config path|init
```

Global flags: `--cwd`, `--mode`, `--provider`, `--model`, `--profile`,
`--allow-home`, `--allow-network`, `--allow-private-network`, `-y/--yes`,
`--json`, `--jsonl`, `--approval`, `--i-accept-risk`, `--debug`.

Slash commands in the REPL: `/help` `/status` `/mode` `/model` `/provider`
`/tools` `/permissions` `/files` `/context` `/diff` `/undo` `/compact`
`/session` `/clear` `/exit`. There is no `/yolo`.

Ctrl+C cancels the work in flight (including killing a running command's process
group). A second Ctrl+C, or Ctrl+D at an empty prompt, exits.

---

## Permission modes

| Mode | Reads | Workspace writes | Shell | Confirmations | How enabled |
|---|---|---|---|---|---|
| `read-only` (default) | yes | no | no | n/a | default |
| `workspace` | yes | yes | no | first write per file | `--mode` or config |
| `agent` | yes | yes | jailed to the workspace | each command unless allowlisted | `--mode` |
| `trusted` | yes | yes | yes | none | `--mode trusted --i-accept-risk` + TTY |

The mode is a **process-level** setting. The model cannot change it, and neither
can anything inside the workspace. A tool that the current mode forbids is not
offered to the model at all, and is refused if the model names it anyway.

`trusted` disables **confirmation prompts only**. It does not disable the
credential-path denylist or the destructive-command policy. It is refused with
`--json`, requires a TTY, and can never come from a config file.

`-y/--yes` is an *approver*, not a permission: it is consulted only after the
guard has already allowed a call, so it cannot escalate the mode.

---

## What is always denied, in every mode

**Credential paths** — `.ssh`, `.gnupg`, `.aws`, `.env*`, `*.pem`, `*.key`,
`id_rsa*`, `id_ed25519*`, `.netrc`, `credentials*`, `secrets*`,
`.git-credentials`, `.npmrc`, and the agent's own config file. A denied read
returns a structured `PermissionDenied`, never empty content.

**Destructive commands** — `rm -rf /`, `rm -rf /*`, `rm -rf ~`/`$HOME` in any
form, recursive deletes of top-level system directories, fork bombs, `mkfs`,
`dd if=`, block-device overwrites, `curl|sh` / `wget|sh`, `chmod -R 777 /`,
`diskutil erase`, `wipefs`. These have **no approval path**: a confirmation
prompt cannot promote them, an allowlist entry cannot whitelist them, and
`trusted` does not bypass them.

**The network** — denied for every tool by default. No tool in v1 declares
network access. `--allow-network` exists for future tools and still blocks cloud
metadata endpoints unconditionally and RFC1918 without `--allow-private-network`.

---

## The workspace jail

Root discovery: `--cwd` → nearest `.git` ancestor → `Path.cwd()`. `$HOME` and `/`
are refused as roots unless `--allow-home` (a jail containing every repo and
every credential on the machine is not a jail).

Every path is canonicalised with `realpath` **before** the containment check, so
`../../etc/passwd` and a symlink pointing at `/etc/shadow` are both refused.
Directory symlinks are never followed during enumeration. `.gitignore` and
`.gemma4ignore` are honoured; `.git/` is always skipped; binaries (NUL in the
first 8 KiB) and files over `max_file_bytes` are refused.

---

## Tools

| Tool | Side effect | Needs |
|---|---|---|
| `fs.read` | read | — |
| `fs.glob` | read | — |
| `fs.grep` | read | — (uses `rg` when present, Python otherwise) |
| `fs.edit` | workspace write | `workspace`+, approval, matching sha256 |
| `fs.write` | workspace write | `workspace`+, approval; **create-only** |
| `shell.exec` | process | `agent`+, approval unless allowlisted |
| `sec.secrets` | security-sensitive | — (read-only, offline) |
| `sec.deps` | security-sensitive | — (read-only, offline) |
| `sec.semgrep` | security-sensitive | — (read-only, local rules only) |

The belt is small on purpose: a 4B model given twenty overlapping tools picks
badly.

### Safe editing

`fs.edit` replaces a **unique** `old_string` and requires the `expected_sha256`
that `fs.read` returned for that file. If the file changed in between — because
you edited it in another window — the edit fails with `PatchConflict` and nothing
is written. That is the intended behaviour, not a bug: the model must re-read.

Writes are atomic (temp file in the same directory, then `os.replace`), preserve
the file's permission bits, and snapshot the original for `/undo`. Snapshots live
under the XDG data directory and expire after 24h. `/undo` restores from those
snapshots — never via `git reset`, which would discard your unrelated changes.

### Shell execution

`argv` lists are preferred; the shell-string form exists for pipes and needs the
same approval, and is disabled entirely in `--json` runs. There is one place in
the codebase where a string becomes a command (`/bin/sh -c` with an explicit
argv); `shell=True` appears nowhere.

The child environment is built from an **allowlist** (`PATH`, `HOME`, `USER`,
`TERM`, `LANG`, `LC_*`, `TMPDIR`, `VIRTUAL_ENV`, …) and re-checked against a
credential denylist, so `AWS_*`, `SSH_*`, `GPG_*`, `*_API_KEY`, `*_TOKEN` and
friends never reach a command. Commands run in a new session so a timeout or
Ctrl+C kills the whole process group, not just the direct child. There is no
PTY; interactive programs are refused rather than left hanging.

### Defensive security tools

Read-only and offline, always. `sec.secrets` reports locations, rule names,
fingerprints, and masked previews — never the credential it found. `sec.deps`
checks manifest hygiene (unpinned versions, URL/VCS deps, missing hashes,
non-default indexes) and explicitly does **not** claim to have checked CVEs,
because that needs network access it does not have. `sec.semgrep` runs only a
locally installed binary against the repository's **own committed ruleset**;
remote registry configs are never fetched.

There is no port scanner, no exploit runner, no credential-attack tool, and no
malware detonation. Those are out of scope permanently, not deferred.

---

## Untrusted content

`GEMMA4.md`, `.gemma4/instructions.md`, file contents, tool output, dependency
metadata, and search results are all **data**. They reach the model wrapped in
`<untrusted source="...">` fences (with any embedded closing tag neutralised),
and the permission guard never reads any of them.

Project instruction files can steer *what* you want done. They cannot enable
trusted mode, enable the network, widen the workspace, expose a credential, or
disable the jail — and there are tests that script the model into trying each.

Everything that crosses into the terminal or back into the model's context is
stripped of ANSI/OSC escape sequences and redacted for credential shapes. The
second half of that matters as much as the first: raw tool output re-injected
into a conversation would otherwise carry an escape channel into the next render.

---

## Configuration

Precedence, highest first:

```text
CLI flags > GEMMA4_* env > <project>/.gemma4/config.toml > ~/.config/gemma4/config.toml > defaults
```

`gemma4 config init` writes a commented starter file. Two rules are enforced in
code, not documented and hoped for:

- **No credentials in TOML.** Any `api_key` / `*_token` / `*_secret`-shaped key
  is dropped from either config file with a warning. Keys come from
  `GEMMA4_API_KEY` in the environment.
- **No trusted mode from a file**, project or user.

`GEMMA_CYBER_OLLAMA_HOST` is honoured as a fallback base URL, so one variable
configures both the existing generate path and the agent.

---

## Sessions, undo, and audit

```text
~/.local/share/gemma4/sessions/<id>/meta.json      provider, model, mode, usage, files
~/.local/share/gemma4/sessions/<id>/messages.jsonl one message per line
~/.local/share/gemma4/undo/<id>/                   pre-edit snapshots, 24h TTL
~/.local/share/gemma4/audit/YYYY-MM-DD.jsonl       authorisation decisions
~/.local/share/gemma4/history                      REPL history
```

Files are 0600, directories 0700, everything local. Messages are appended as they
happen, so an interrupted turn leaves a resumable session. `gemma4 resume` picks
up the newest session for the current workspace.

The audit log records session id, tool, redacted arguments, the allow/deny
decision and the rule that produced it, duration, and exit code. It records
**no** prompts, completions, file bodies, environment dumps, or secrets.

---

## CI usage

```bash
gemma4 --json run "check the auth tests and explain the failure"
```

No TTY prompts in machine mode. The noninteractive approval policy is `fail` by
default: a run that needed a human exits `6` rather than silently proceeding.
`--approval reject` instead declines the call and lets the model continue.

Exit codes: `0` ok · `1` error · `2` provider unreachable · `3` model
unavailable · `4` usage · `5` findings (`--fail-on-findings`) · `6` approval
required · `7` cancelled.

`--jsonl` streams one JSON object per event, then the result object.

---

## Profiles

`general` (default) · `coder` · `auditor`. Profiles are configuration over the
same runtime, not separate architectures. `auditor` is read-only and exposes only
the search and defensive-scanning tools — never `fs.edit` or `shell.exec`, even
though the runtime supports them.

```bash
gemma4 --profile auditor --json run "find committed credentials; do not modify anything"
```
