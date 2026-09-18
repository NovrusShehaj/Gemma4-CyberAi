"""`PermissionGuard` — the only module that answers "may this happen" (plan §15).

Three rules define it:

1. **The model proposes; this code authorises.** A tool call reaches `run()`
   only after schema validation *and* :meth:`PermissionGuard.authorize`.
   "The model asked for it" is not an input to any decision here.
2. **Workspace content is never consulted.** This module does not read
   ``GEMMA4.md``, source files, or tool output. A repository cannot grant itself
   privileges, so a prompt injection that gets the model to *ask* for a shell in
   ``read-only`` still returns `PermissionDenied`.
3. **Some things are denied in every mode**, ``trusted`` included: credential
   paths and catastrophic commands. Trusted removes *confirmation prompts*, not
   the absolute controls.

The mode matrix (plan §15):

=================  =====  ================  =====================  ========
mode               read   workspace_write   process (shell)        confirms
=================  =====  ================  =====================  ========
read-only          yes    no                no                     n/a
workspace          yes    yes               no                     first write per file
agent              yes    yes               yes (jailed cwd)       each command unless allowlisted
trusted            yes    yes               yes                    none
=================  =====  ================  =====================  ========
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gemma_cyber.agent.errors import PermissionDenied, WorkspaceEscape
from gemma_cyber.agent.types import AgentState, PermissionMode, SideEffect
from gemma_cyber.agent.workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gemma_cyber.agent.tools.base import Tool

__all__ = [
    "DANGEROUS_COMMAND_PATTERNS",
    "INTERACTIVE_COMMANDS",
    "SENSITIVE_PATH_GLOBS",
    "SENSITIVE_PATH_PARTS",
    "Decision",
    "PermissionGuard",
    "is_blocked_network_host",
    "is_sensitive_path",
]


# -- always-deny paths (plan §15) -------------------------------------------

#: Directory names that are denied wherever they appear in a path.
SENSITIVE_PATH_PARTS = frozenset(
    {".ssh", ".gnupg", ".gpg", ".aws", ".azure", ".gcloud", ".kube", ".docker",
     ".config/gemma4", ".password-store", ".chef"}
)

#: Filename globs denied anywhere. Matched case-insensitively against each
#: component of the path, so `secrets/id_rsa.bak` is caught as readily as
#: `id_rsa`.
SENSITIVE_PATH_GLOBS = (
    ".env", ".env.*", "*.env",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "*.ppk",
    "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*",
    ".netrc", "_netrc", ".git-credentials", ".htpasswd", ".npmrc", ".pypirc",
    "credentials", "credentials.json", "credentials.yaml", "credentials.yml",
    "secrets.json", "secrets.yaml", "secrets.yml", ".secrets",
    "service-account*.json", "*.kdbx", "shadow", "master.key",
)

#: Agent/config files that may hold keys. Denied even though a key should never
#: be written there — defence in depth against a user who did it anyway.
SENSITIVE_RELATIVE_PATHS = (
    ".gemma4/config.toml",
)


def is_sensitive_path(path: Path | str, *, workspace_root: Path | None = None) -> str | None:
    """Return the matched rule if ``path`` is credential-bearing, else ``None``.

    Denial is by *rule name* so the audit log records which control fired, and
    the caller can tell the model "denied by policy" without echoing the path
    into a message that might later be rendered.
    """
    candidate = Path(path)
    parts = [p for p in candidate.parts if p not in ("/", "\\")]
    lowered = [p.lower() for p in parts]

    for index, part in enumerate(lowered):
        if part in SENSITIVE_PATH_PARTS:
            return f"sensitive-directory:{part}"
        # Two-segment rules like ".config/gemma4".
        if index + 1 < len(lowered) and f"{part}/{lowered[index + 1]}" in SENSITIVE_PATH_PARTS:
            return f"sensitive-directory:{part}/{lowered[index + 1]}"

    for part in lowered:
        for pattern in SENSITIVE_PATH_GLOBS:
            if fnmatch(part, pattern):
                return f"sensitive-file:{pattern}"

    if workspace_root is not None:
        try:
            rel = candidate.resolve(strict=False).relative_to(workspace_root).as_posix()
        except (ValueError, OSError):
            rel = ""
        for reserved in SENSITIVE_RELATIVE_PATHS:
            if rel == reserved:
                return f"agent-config:{reserved}"
    return None


# -- always-deny commands (plan §14) ----------------------------------------

#: Evaluated against the raw proposed command (argv joined, whitespace collapsed).
#: These are denied in every mode; a confirmation prompt cannot promote them.
DANGEROUS_COMMAND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("rm-rf-root", re.compile(r"\brm\b[^|;&]*\s-[a-zA-Z]*[rR][a-zA-Z]*f?[a-zA-Z]*\s+(/|/\*)(\s|$)")),
    ("rm-rf-root-alt", re.compile(r"\brm\b[^|;&]*\s-[a-zA-Z]*f[a-zA-Z]*[rR][a-zA-Z]*\s+(/|/\*)(\s|$)")),
    # `~` and `$HOME` in ANY form: `rm -rf ~`, `~/`, `~/Documents`, `$HOME/...`.
    # A workspace agent never has a legitimate reason to recursively delete
    # anything under the home directory; the earlier version of this rule
    # required a space or end-of-string after `~` and so let `rm -rf ~/` through.
    ("rm-rf-home", re.compile(
        r"\brm\b[^|;&]*\s-[a-zA-Z]*[rR][a-zA-Z]*\s+(~|\$HOME|\$\{HOME\})(/\S*)?(\s|$)")),
    # Recursive delete of a top-level system directory.
    ("rm-rf-system-dir", re.compile(
        r"\brm\b[^|;&]*\s-[a-zA-Z]*[rR][a-zA-Z]*\s+/(usr|etc|var|bin|sbin|lib|opt|boot"
        r"|System|Library|Applications|Users|home|root|private|dev|Volumes)(/\S*)?(\s|$)")),
    ("fork-bomb", re.compile(r":\s*\(\s*\)\s*\{|\.\s*\(\s*\)\s*\{\s*\.\|\.")),
    ("mkfs", re.compile(r"\bmkfs(\.[a-z0-9]+)?\b")),
    ("dd-to-device", re.compile(r"\bdd\b[^|;&]*\bif=")),
    ("disk-erase", re.compile(r"\bdiskutil\s+(erase|partitionDisk|reformat)")),
    ("overwrite-block-device", re.compile(r">\s*/dev/(sd[a-z]|nvme\d|disk\d|hd[a-z])")),
    ("curl-pipe-shell", re.compile(
        r"\b(curl|wget|fetch)\b[\s\S]*\|\s*(sudo\s+)?(ba|z|k|da|)sh\b")),
    ("remote-script-exec", re.compile(
        r"\b(ba|z|k|da|)sh\s+-c\s+[\"']?\s*\$\(\s*(curl|wget)\b")),
    ("chmod-777-root", re.compile(r"\bchmod\b[^|;&]*\s-R[^|;&]*\s777\s+/(\s|$)")),
    ("chown-root-recursive", re.compile(r"\bchown\b[^|;&]*\s-R[^|;&]*\s+/(\s|$)")),
    ("history-wipe", re.compile(r"\bshred\b[^|;&]*/dev/|\bwipefs\b")),
)

#: Programs that always need a terminal the MVP does not provide (no PTY —
#: plan §14). Refused regardless of arguments.
INTERACTIVE_COMMANDS = frozenset(
    {"vi", "vim", "nvim", "emacs", "nano", "pico", "less", "more", "top", "htop",
     "man", "tmux", "screen", "gdb", "lldb", "ftp", "telnet", "sudo", "su",
     "passwd", "vipw", "visudo", "crontab"}
)

#: Interpreters that are interactive only when given no program to run.
#: `python -c "..."` and `node script.js` are ordinary batch commands and must
#: keep working; a bare `python` would hang forever waiting on a closed stdin.
REPL_COMMANDS = frozenset(
    {"python", "python2", "python3", "node", "irb", "ruby", "php", "psql",
     "mysql", "sqlite3", "R", "julia", "ghci", "iex"}
)


def dangerous_command_rule(raw_command: str) -> str | None:
    """Return the deny-rule name that ``raw_command`` trips, or ``None``."""
    normalised = " ".join(raw_command.split())
    for name, pattern in DANGEROUS_COMMAND_PATTERNS:
        if pattern.search(normalised):
            return name
    return None


# -- network policy (plan §15) ----------------------------------------------

_METADATA_HOSTS = frozenset(
    {"169.254.169.254", "fd00:ec2::254", "metadata.google.internal",
     "metadata.goog", "100.100.100.200"}
)


def is_blocked_network_host(host: str, *, allow_private: bool = False) -> str | None:
    """Policy for any future network-enabled tool. No such tool exists in v1.

    Cloud metadata endpoints are blocked unconditionally — they are the standard
    SSRF credential-theft target and there is no legitimate agent use for them.
    RFC1918 / loopback / link-local require a second explicit opt-in.
    """
    target = host.strip().lower().strip("[]")
    if not target:
        return "empty-host"
    if target in _METADATA_HOSTS:
        return "cloud-metadata"
    try:
        address = ipaddress.ip_address(target)
    except ValueError:
        return None  # a name; DNS-time re-check is the caller's job
    if address.is_link_local:
        return "link-local"
    if not allow_private and (address.is_private or address.is_loopback or address.is_reserved):
        return "private-network"
    return None


# -- the guard --------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Decision:
    """Outcome of an authorisation. ``allowed=False`` is always raised, not returned."""

    allowed: bool
    requires_confirm: bool
    reason: str = ""
    #: Which rule permitted or required confirmation — recorded in the audit log.
    rule: str = ""


class PermissionGuard:
    """Authorises tool calls against the process mode. Stateless except for config."""

    def __init__(
        self,
        workspace: Workspace,
        mode: PermissionMode,
        *,
        shell_allowlist: tuple[str, ...] = (),
        allow_network: bool = False,
        allow_private_network: bool = False,
    ) -> None:
        self.workspace = workspace
        self.mode = mode
        self.shell_allowlist = tuple(a.strip() for a in shell_allowlist if a.strip())
        self.allow_network = allow_network
        self.allow_private_network = allow_private_network

    # -- visibility ---------------------------------------------------------

    def is_visible(self, tool: Tool) -> bool:
        """Whether a tool is offered to the model at all.

        Hiding `shell.exec` in ``read-only`` is not cosmetic: a 4B model handed a
        tool it cannot use burns iterations proposing denied calls (plan §9.3).
        The guard still denies it if the model names it anyway.
        """
        return self._mode_allows(tool.side_effect) and not (
            tool.network and not self.allow_network
        )

    def _mode_allows(self, side_effect: SideEffect) -> bool:
        if side_effect in (SideEffect.READ, SideEffect.SECURITY_SENSITIVE):
            return True
        if side_effect is SideEffect.WORKSPACE_WRITE:
            return self.mode.at_least(PermissionMode.WORKSPACE)
        if side_effect is SideEffect.PROCESS:
            return self.mode.at_least(PermissionMode.AGENT)
        if side_effect is SideEffect.NETWORK:
            return self.allow_network
        return False  # unknown side effect → deny (fail closed)

    # -- authorisation ------------------------------------------------------

    def authorize(self, tool: Tool, args: Any, state: AgentState) -> Decision:
        """Authorise one call. Raises :class:`PermissionDenied` when refused.

        ``args`` is the *validated* pydantic model, never the raw model JSON —
        validation happens first so this method reasons about typed fields.
        """
        # 1. Absolute controls first: they apply in every mode, trusted included.
        self._check_paths(tool, args)
        self._check_command(tool, args)
        self._check_network(tool, args)

        # 2. Mode ceiling.
        if not self._mode_allows(tool.side_effect):
            raise PermissionDenied(
                f"{tool.name} requires a higher permission mode "
                f"({self._required_mode(tool.side_effect)}); current mode is {self.mode.value}. "
                f"Restart with --mode {self._required_mode(tool.side_effect)} if that is intended."
            )

        # 3. Confirmation policy.
        return self._confirmation(tool, args, state)

    @staticmethod
    def _required_mode(side_effect: SideEffect) -> str:
        if side_effect is SideEffect.WORKSPACE_WRITE:
            return PermissionMode.WORKSPACE.value
        if side_effect is SideEffect.PROCESS:
            return PermissionMode.AGENT.value
        if side_effect is SideEffect.NETWORK:
            return "--allow-network"
        return PermissionMode.READ_ONLY.value

    def _check_paths(self, tool: Tool, args: Any) -> None:
        """Jail + sensitive-path check on every declared path argument.

        A denial returns a structured error; it never returns empty content
        pretending the file was blank (plan §15).
        """
        for field in tool.path_fields:
            raw = getattr(args, field, None)
            if raw in (None, ""):
                continue
            # Check the literal request first: `~/.ssh/id_rsa` must be reported
            # as a credential denial, not merely as a jail escape.
            rule = is_sensitive_path(Path(str(raw)).expanduser())
            if rule:
                raise PermissionDenied(
                    f"access to credential material is denied by policy [{rule}]. "
                    "This is denied in every mode, including trusted."
                )
            resolved = self.workspace.resolve_in_jail(str(raw))  # raises WorkspaceEscape
            rule = is_sensitive_path(resolved, workspace_root=self.workspace.root)
            if rule:
                raise PermissionDenied(
                    f"access to credential material is denied by policy [{rule}]. "
                    "This is denied in every mode, including trusted."
                )

    def _check_command(self, tool: Tool, args: Any) -> None:
        if not tool.command_field:
            return
        raw = getattr(args, tool.command_field, None)
        if raw is None:
            return
        command = " ".join(raw) if isinstance(raw, (list, tuple)) else str(raw)
        rule = dangerous_command_rule(command)
        if rule:
            raise PermissionDenied(
                f"command refused by the destructive-command policy [{rule}]. "
                "This rule has no confirmation path and is not disabled by trusted mode."
            )

    def _check_network(self, tool: Tool, args: Any) -> None:
        if not tool.network:
            return
        if not self.allow_network:
            raise PermissionDenied(
                f"{tool.name} needs network access, which is denied by default. "
                "Restart with --allow-network if that is intended."
            )
        host = getattr(args, "host", None) or getattr(args, "url", None)
        if not host:
            return
        target = str(host).split("//")[-1].split("/")[0].split(":")[0]
        blocked = is_blocked_network_host(target, allow_private=self.allow_private_network)
        if blocked:
            raise PermissionDenied(f"network destination refused by policy [{blocked}]")

    def _confirmation(self, tool: Tool, args: Any, state: AgentState) -> Decision:
        if self.mode is PermissionMode.TRUSTED:
            # Trusted skips prompts only. The absolute checks above already ran.
            return Decision(True, False, "trusted mode: confirmations disabled", "trusted")

        if tool.side_effect is SideEffect.WORKSPACE_WRITE:
            key = self.write_key(tool, args)
            if key and key in state.approved_writes:
                return Decision(True, False, "write already approved this session", "write-approved")
            return Decision(True, True, "first write to this path in this session", "write-confirm")

        if tool.side_effect is SideEffect.PROCESS:
            command = self._command_string(tool, args)
            match = self._allowlisted(command)
            if match:
                return Decision(True, False, f"allowlisted command prefix {match!r}", "shell-allowlist")
            return Decision(True, True, "shell commands require approval", "shell-confirm")

        if tool.requires_confirm:
            return Decision(True, True, f"{tool.name} always confirms", "tool-confirm")

        return Decision(True, False, "read-class operation", "read")

    def _allowlisted(self, command: str) -> str | None:
        """Exact prefix match against the configured allowlist.

        Prefix, not substring, and on a whitespace-normalised command, so
        ``git status`` does not also authorise ``echo x; git status``. Any shell
        metacharacter disqualifies the command from the allowlist entirely.
        """
        normalised = " ".join(command.split())
        if any(ch in normalised for ch in ";|&<>$`\n"):
            return None
        for prefix in self.shell_allowlist:
            wanted = " ".join(prefix.split())
            if normalised == wanted or normalised.startswith(wanted + " "):
                return prefix
        return None

    def _command_string(self, tool: Tool, args: Any) -> str:
        if not tool.command_field:
            return ""
        raw = getattr(args, tool.command_field, None)
        if raw is None:
            return ""
        return " ".join(raw) if isinstance(raw, (list, tuple)) else str(raw)

    def write_key(self, tool: Tool, args: Any) -> str | None:
        """Workspace-relative path a write-approval is remembered against."""
        for field in tool.path_fields:
            raw = getattr(args, field, None)
            if raw:
                try:
                    return self.workspace.relative(self.workspace.resolve_in_jail(str(raw)))
                except WorkspaceEscape:  # pragma: no cover - _check_paths ran first
                    return None
        return None
