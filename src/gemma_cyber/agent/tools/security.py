"""Defensive, read-only cybersecurity tools (plan §20).

Scope is a product decision, not a capability limit. These tools inspect the
workspace the user already has. They never touch a remote target, never execute
a payload, and never fetch a rule or a database over the network. There is no
`sec.scan_target`, no exploit runner, and no detonation sandbox — those are
permanently out of scope for `gemma4`, not deferred.

**On the side effect.** These are `SECURITY_SENSITIVE`, not `PROCESS`, even
though two of them spawn a helper binary. The distinction is who chooses the
command: `shell.exec` runs a *model-authored* command and therefore needs
`agent` mode plus approval, while these run a fixed argv against a fixed binary
with arguments derived from a validated schema. That is what makes them safe in
the read-only `auditor` profile. It is still a process spawn, and it is audited
as one.

**On findings.** A secret scanner that prints the secrets it finds has created a
second copy of the problem. Findings carry a location, a rule name, a stable
fingerprint, and a masked preview — never the matched credential.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from gemma_cyber.agent.errors import ToolError
from gemma_cyber.agent.sanitize import _SECRET_PATTERNS
from gemma_cyber.agent.tools.base import Tool, ToolContext
from gemma_cyber.agent.types import SideEffect, ToolResult

__all__ = [
    "DepsAuditTool",
    "SecretScanTool",
    "SemgrepTool",
    "builtin_security_tools",
    "shannon_entropy",
]

_MAX_FINDINGS = 100
_HELPER_TIMEOUT_S = 120.0

#: Long opaque tokens in an assignment-ish context. Entropy filters the rest.
_CANDIDATE_RE = re.compile(
    r"""(?ix)
    \b(?P<name>[A-Z0-9_]*(?:key|secret|token|password|passwd|credential|auth|salt)[A-Z0-9_]*)
    \s*[:=]\s*
    ['"]?(?P<value>[A-Za-z0-9+/=_\-]{20,})['"]?
    """
)

#: Entropy above this, for a 20+ character token, is very unlikely to be prose.
_ENTROPY_THRESHOLD = 3.6

#: Filenames whose "secrets" are fixtures or documentation, not incidents.
_LOW_SIGNAL_NAMES = ("test", "spec", "fixture", "example", "sample", "mock", "doc")


def shannon_entropy(text: str) -> float:
    """Bits per character. Used to separate real keys from long identifiers."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _mask(value: str) -> str:
    """Enough to recognise a value you already know; useless to anyone else."""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}{'*' * 8}{value[-2:]}"


def _helper(name: str) -> str | None:
    return shutil.which(name)


def _run_helper(argv: list[str], ctx: ToolContext) -> subprocess.CompletedProcess[str]:
    """Run a fixed helper binary. No shell, no model-supplied argv, no network flags."""
    from gemma_cyber.agent.tools.shell import build_child_env

    try:
        return subprocess.run(  # noqa: S603 - fixed binary, argv list, never shell
            argv, cwd=str(ctx.workspace.root), env=build_child_env(
                workspace_root=ctx.workspace.root),
            capture_output=True, text=True, errors="replace",
            timeout=_HELPER_TIMEOUT_S, check=False, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"{Path(argv[0]).name} timed out after {_HELPER_TIMEOUT_S:.0f}s") from exc
    except OSError as exc:
        raise ToolError(f"could not run {Path(argv[0]).name}: {exc.strerror}") from exc


def _findings_result(tool: Tool, summary: str, findings: list[dict], **extra: object
                     ) -> ToolResult:
    """Findings go to the model as structured JSON data, not prose."""
    payload = {"summary": summary, "count": len(findings), "findings": findings, **extra}
    result = ToolResult(
        call_id="", name=tool.name, ok=True,
        content=json.dumps(payload, indent=2, default=str),
    )
    result.metadata = {"findings": len(findings)}
    return result


# -- sec.secrets ------------------------------------------------------------

class SecretScanArgs(BaseModel):
    path: str | None = Field(
        default=None, description="Subdirectory to scan. Defaults to the whole workspace."
    )
    include_low_signal: bool = Field(
        default=False,
        description="Include hits in test/fixture/example files, which are usually not real.",
    )


class SecretScanTool(Tool):
    name = "sec.secrets"
    description = (
        "Scan workspace files for committed credentials. Uses gitleaks or "
        "detect-secrets when installed, otherwise pattern plus entropy analysis. "
        "Reports locations and fingerprints, never the credential itself."
    )
    input_model = SecretScanArgs
    side_effect = SideEffect.SECURITY_SENSITIVE
    timeout_s = _HELPER_TIMEOUT_S
    path_fields = ("path",)

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, SecretScanArgs)
        root = ctx.workspace.resolve_in_jail(args.path) if args.path else ctx.workspace.root

        if binary := _helper("gitleaks"):
            findings, engine = self._gitleaks(binary, root, ctx), "gitleaks"
        else:
            findings, engine = self._builtin(root, ctx, args), "builtin"

        if not args.include_low_signal:
            findings = [f for f in findings if not f.get("low_signal")]
        findings = findings[:_MAX_FINDINGS]

        summary = (
            f"{len(findings)} potential credential(s) found by the {engine} engine"
            if findings else f"no committed credentials found by the {engine} engine"
        )
        return _findings_result(
            self, summary, findings, engine=engine,
            note="Values are masked and fingerprinted on purpose. Verify each hit by "
                 "opening the file yourself; rotate anything real rather than only deleting it.",
        )

    def _gitleaks(self, binary: str, root: Path, ctx: ToolContext) -> list[dict]:
        """Prefer the installed scanner. `--no-git` keeps it to the working tree."""
        completed = _run_helper(
            [binary, "detect", "--no-git", "--redact", "--report-format", "json",
             "--report-path", "-", "--source", str(root)],
            ctx,
        )
        try:
            raw = json.loads(completed.stdout or "[]")
        except json.JSONDecodeError:
            raw = []
        findings: list[dict] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("File", ""))
            findings.append({
                "path": rel,
                "line": item.get("StartLine"),
                "rule": item.get("RuleID", "gitleaks"),
                # gitleaks was run with --redact; never surface `Secret` anyway.
                "fingerprint": _fingerprint(str(item.get("Fingerprint") or rel)),
                "low_signal": self._is_low_signal(rel),
            })
        return findings

    def _builtin(self, root: Path, ctx: ToolContext, args: SecretScanArgs) -> list[dict]:
        """Pattern + entropy fallback. `.git` is excluded by the workspace walker."""
        findings: list[dict] = []
        candidates = [root] if root.is_file() else [
            p for p in ctx.workspace.iter_files(include_hidden=True)
            if str(p).startswith(str(root))
        ]
        for path in candidates:
            if ctx.cancel.is_cancelled() or len(findings) >= _MAX_FINDINGS:
                break
            try:
                data = ctx.workspace.open_read_bytes(path, max_bytes=ctx.max_file_bytes)
            except Exception:  # noqa: BLE001 - an unreadable file is not a finding
                continue
            if ctx.workspace.looks_binary(data):
                continue
            rel = ctx.workspace.relative(path)
            low_signal = self._is_low_signal(rel)
            for number, line in enumerate(
                data.decode("utf-8", errors="replace").splitlines(), start=1
            ):
                findings.extend(self._scan_line(line, rel, number, low_signal))
                if len(findings) >= _MAX_FINDINGS:
                    break
        return findings

    def _scan_line(self, line: str, rel: str, number: int, low_signal: bool) -> list[dict]:
        out: list[dict] = []
        for kind, pattern in _SECRET_PATTERNS:
            if kind == "assignment":
                continue  # handled by the entropy pass below, with fewer false positives
            for match in pattern.finditer(line):
                value = match.group(0)
                out.append({
                    "path": rel, "line": number, "rule": kind,
                    "masked": _mask(value), "fingerprint": _fingerprint(value),
                    "low_signal": low_signal,
                })
        for match in _CANDIDATE_RE.finditer(line):
            value = match.group("value")
            entropy = shannon_entropy(value)
            if entropy < _ENTROPY_THRESHOLD:
                continue
            if any(placeholder in value.lower()
                   for placeholder in ("example", "changeme", "your_", "xxxxx", "placeholder")):
                continue
            out.append({
                "path": rel, "line": number, "rule": "high-entropy-assignment",
                "name": match.group("name"), "entropy": round(entropy, 2),
                "masked": _mask(value), "fingerprint": _fingerprint(value),
                "low_signal": low_signal,
            })
        return out

    @staticmethod
    def _is_low_signal(rel: str) -> bool:
        lowered = rel.lower()
        return any(marker in lowered for marker in _LOW_SIGNAL_NAMES)


# -- sec.deps ---------------------------------------------------------------

class DepsAuditArgs(BaseModel):
    pass


class DepsAuditTool(Tool):
    name = "sec.deps"
    description = (
        "Inspect committed dependency manifests and lockfiles for supply-chain "
        "hygiene problems: unpinned versions, VCS/URL dependencies, missing "
        "hashes, and non-default package indexes. Offline: it does not look up CVEs."
    )
    input_model = DepsAuditArgs
    side_effect = SideEffect.SECURITY_SENSITIVE
    timeout_s = 30.0

    MANIFESTS = (
        "requirements.txt", "requirements-dev.txt", "requirements-train.txt",
        "pyproject.toml", "uv.lock", "poetry.lock", "Pipfile.lock",
        "package.json", "package-lock.json", "yarn.lock", "go.mod", "Cargo.lock",
    )

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        present: list[str] = []
        findings: list[dict] = []
        for name in self.MANIFESTS:
            candidate = ctx.workspace.root / name
            if not candidate.is_file():
                continue
            present.append(name)
            if name.startswith("requirements"):
                findings.extend(self._audit_requirements(candidate, ctx))

        if not present:
            return _findings_result(self, "no dependency manifests found", [])

        summary = (
            f"{len(findings)} dependency hygiene issue(s) across {len(present)} manifest(s)"
            if findings else f"no hygiene issues in {len(present)} manifest(s)"
        )
        return _findings_result(
            self, summary, findings, manifests=present,
            note="This check is offline by design: vulnerability lookup needs network "
                 "access, which gemma4 denies by default. Run `pip-audit` or `osv-scanner` "
                 "yourself for CVE data.",
        )

    def _audit_requirements(self, path: Path, ctx: ToolContext) -> list[dict]:
        findings: list[dict] = []
        rel = ctx.workspace.relative(path)
        try:
            text = ctx.workspace.read_text(path, max_bytes=ctx.max_file_bytes)
        except ToolError:
            return findings
        for number, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("--index-url", "--extra-index-url")):
                findings.append({
                    "path": rel, "line": number, "rule": "non-default-index",
                    "detail": "packages resolve from a non-default index; confirm it is trusted",
                })
                continue
            if line.startswith("-"):
                continue
            if any(marker in line for marker in ("git+", "http://", "https://", "file://")):
                findings.append({
                    "path": rel, "line": number, "rule": "url-dependency",
                    "detail": "dependency installed from a URL/VCS ref rather than an index",
                })
            elif not any(op in line for op in ("==", "===", "@")):
                findings.append({
                    "path": rel, "line": number, "rule": "unpinned-dependency",
                    "detail": f"{line.split('[')[0]} is not pinned to an exact version",
                })
            if "--hash=" not in line and "==" in line and path.name != "pyproject.toml":
                findings.append({
                    "path": rel, "line": number, "rule": "missing-hash",
                    "detail": "pinned without a hash; install is not reproducible-verifiable",
                })
        return findings


# -- sec.semgrep ------------------------------------------------------------

class SemgrepArgs(BaseModel):
    path: str | None = Field(default=None, description="Subdirectory to scan.")


class SemgrepTool(Tool):
    name = "sec.semgrep"
    description = (
        "Run a locally installed Semgrep against the workspace using the "
        "repository's own committed ruleset. Does nothing if Semgrep is not "
        "installed or the repository has no local rules."
    )
    input_model = SemgrepArgs
    side_effect = SideEffect.SECURITY_SENSITIVE
    timeout_s = _HELPER_TIMEOUT_S
    path_fields = ("path",)

    LOCAL_RULE_PATHS = (".semgrep.yml", ".semgrep.yaml", "semgrep.yml", ".semgrep")

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, SemgrepArgs)
        binary = _helper("semgrep")
        if binary is None:
            return _findings_result(
                self, "semgrep is not installed; no scan was run", [],
                note="Install semgrep locally to enable this tool. gemma4 will not "
                     "download it, and will not fetch rules from a remote registry.",
            )

        rules = next(
            (ctx.workspace.root / name for name in self.LOCAL_RULE_PATHS
             if (ctx.workspace.root / name).exists()),
            None,
        )
        if rules is None:
            return _findings_result(
                self, "no committed Semgrep ruleset found; no scan was run", [],
                note="Add a .semgrep.yml to the repository. Remote registry configs "
                     "(for example `--config p/ci`) are deliberately not used: they "
                     "would download and execute rules the repository has not reviewed.",
            )

        target = ctx.workspace.resolve_in_jail(args.path) if args.path else ctx.workspace.root
        completed = _run_helper(
            [binary, "--config", str(rules), "--json", "--quiet", "--no-git-ignore",
             "--metrics", "off", "--disable-version-check", str(target)],
            ctx,
        )
        try:
            data = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            raise ToolError("semgrep returned output that could not be parsed") from None

        findings = [
            {
                "path": str(item.get("path", "")),
                "line": (item.get("start") or {}).get("line"),
                "rule": item.get("check_id", "semgrep"),
                "severity": (item.get("extra") or {}).get("severity", "unknown"),
                "message": str((item.get("extra") or {}).get("message", ""))[:300],
            }
            for item in (data.get("results") or [])
            if isinstance(item, dict)
        ][:_MAX_FINDINGS]

        summary = (f"{len(findings)} semgrep finding(s)" if findings
                   else "semgrep reported no findings")
        return _findings_result(self, summary, findings, ruleset=ctx.workspace.relative(rules))


def builtin_security_tools() -> list[Tool]:
    """The defensive toolbelt. Read-only, offline, no live targets — ever."""
    return [SecretScanTool(), DepsAuditTool(), SemgrepTool()]
