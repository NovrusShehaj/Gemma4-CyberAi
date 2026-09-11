"""`shell.exec` — command execution inside the workspace jail (plan §14).

Deliberately built last of the MVP tools: it is only safe because
`PermissionGuard` (mode ceiling + destructive-command policy) and `Workspace`
(cwd jail) already exist and are tested. It is invisible below `--mode agent`.

Design choices that are security decisions, not style:

* **argv lists, never `shell=True`.** The shell-string path (when the user opts
  into it) runs an explicit ``/bin/sh -c`` argv, so there is exactly one place in
  the codebase where a string becomes a command, and it is auditable.
* **Environment is built from an allowlist**, not filtered from `os.environ`.
  A denylist misses the next credential variable someone invents.
* **New session / process group**, so a cancel or timeout kills the whole tree —
  ``make test`` that forked three children does not survive Ctrl+C.
* **No PTY.** A pty would add an escape-sequence channel and Windows complexity
  for no MVP benefit; interactive commands are refused instead.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from gemma_cyber.agent.errors import CommandTimeout, PermissionDenied, ToolError, UserCancelled
from gemma_cyber.agent.permissions import (
    INTERACTIVE_COMMANDS,
    REPL_COMMANDS,
    dangerous_command_rule,
)
from gemma_cyber.agent.tools.base import Tool, ToolContext
from gemma_cyber.agent.types import SideEffect, ToolResult

__all__ = ["ShellArgs", "ShellExecTool", "build_child_env"]

#: The only variables a child process inherits. Everything else is dropped,
#: including anything added to the parent environment in the future.
ENV_ALLOWLIST = frozenset(
    {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "TZ", "TMPDIR",
     "TEMP", "TMP", "VIRTUAL_ENV", "PWD", "COLUMNS", "LINES", "SYSTEMROOT",
     "COMSPEC", "PATHEXT"}
)

#: Prefixes/suffixes that must never reach a child even if something widens the
#: allowlist later. Defence in depth over the allowlist above.
ENV_DENY_SUBSTRINGS = (
    "AWS_", "SSH_", "GPG_", "GNUPG", "AZURE_", "GOOGLE_", "GCP_", "DOCKER_",
    "KUBE", "NPM_TOKEN", "API_KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD",
    "PASSWD", "CREDENTIAL", "SESSION_KEY", "PRIVATE_KEY",
)

_MAX_ARGV = 64
_MAX_COMMAND_CHARS = 4000


def build_child_env(parent: dict[str, str] | None = None, *, workspace_root: Path | None = None
                    ) -> dict[str, str]:
    """Construct the child environment from scratch (plan §14).

    Built by allowlist *and* re-checked against the denylist, so a future edit
    that adds a broad entry cannot silently start leaking credentials.
    """
    source = dict(os.environ if parent is None else parent)
    child: dict[str, str] = {}
    for key, value in source.items():
        if key in ENV_ALLOWLIST or key.startswith("LC_"):
            if any(marker in key.upper() for marker in ENV_DENY_SUBSTRINGS):
                continue
            child[key] = value
    child.setdefault("PATH", os.defpath)
    if workspace_root is not None:
        child["PWD"] = str(workspace_root)
    # Reduce the amount of escape-sequence output we have to strip afterwards.
    # The sanitiser still runs; this just lowers the noise.
    child["NO_COLOR"] = "1"
    child["CLICOLOR"] = "0"
    child["GEMMA4_AGENT"] = "1"
    return child


class ShellArgs(BaseModel):
    """Exactly one of ``argv`` or ``command``; ``argv`` is strongly preferred."""

    argv: list[str] | None = Field(
        default=None,
        description="Command and arguments as a list, e.g. [\"pytest\", \"-q\"]. Preferred.",
    )
    command: str | None = Field(
        default=None,
        description="A shell command string. Only use when you need a pipe or "
                    "redirect; it requires a stricter confirmation.",
    )
    cwd: str | None = Field(
        default=None, description="Working directory, relative to the workspace root."
    )
    timeout_s: float | None = Field(default=None, gt=0, le=600)

    @model_validator(mode="after")
    def _exactly_one(self) -> ShellArgs:
        if bool(self.argv) == bool(self.command):
            raise ValueError("provide exactly one of argv or command")
        if self.argv is not None and len(self.argv) > _MAX_ARGV:
            raise ValueError(f"argv has more than {_MAX_ARGV} elements")
        if self.argv is not None and not all(isinstance(a, str) and a for a in self.argv):
            raise ValueError("argv entries must be non-empty strings")
        if self.command is not None and len(self.command) > _MAX_COMMAND_CHARS:
            raise ValueError(f"command is longer than {_MAX_COMMAND_CHARS} characters")
        return self

    @property
    def raw_command(self) -> str:
        """The single string the destructive-command policy is evaluated against."""
        if self.command is not None:
            return self.command
        return shlex.join(self.argv or [])

    @property
    def uses_shell_string(self) -> bool:
        return self.command is not None


class ShellExecTool(Tool):
    name = "shell.exec"
    description = (
        "Run a command inside the workspace. Requires agent mode and, unless "
        "allowlisted, explicit approval. Not a terminal: no interactive programs, "
        "no input, output is captured and capped."
    )
    input_model = ShellArgs
    side_effect = SideEffect.PROCESS
    timeout_s = 60.0
    requires_confirm = True
    mutating = True
    path_fields = ("cwd",)
    command_field = "raw_command"

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, ShellArgs)

        # The guard already applied this policy. Re-checking here means a future
        # caller that forgets the guard still cannot run `rm -rf /`.
        if rule := dangerous_command_rule(args.raw_command):
            raise PermissionDenied(f"command refused by policy [{rule}]")

        if args.uses_shell_string and not ctx.extra.get("allow_shell_string", True):
            raise PermissionDenied(
                "shell-string commands are disabled in noninteractive mode. "
                "Pass an argv list instead."
            )

        argv = self._build_argv(args)
        self._reject_interactive(args)

        cwd = ctx.workspace.resolve_in_jail(args.cwd) if args.cwd else ctx.workspace.root
        if not cwd.is_dir():
            raise ToolError(f"cwd is not a directory: {args.cwd}")

        timeout = min(args.timeout_s or ctx.shell_timeout_s, ctx.shell_timeout_s)
        env = build_child_env(workspace_root=ctx.workspace.root)

        return self._spawn(argv, cwd=cwd, env=env, timeout=timeout, ctx=ctx, args=args)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _build_argv(args: ShellArgs) -> list[str]:
        if args.argv:
            return list(args.argv)
        # The one place a string becomes a command. Explicit `/bin/sh -c` rather
        # than `shell=True`: same semantics, one obvious audit point.
        shell = "/bin/sh" if os.name != "nt" else os.environ.get("COMSPEC", "cmd.exe")
        flag = "-c" if os.name != "nt" else "/c"
        return [shell, flag, args.command or ""]

    @staticmethod
    def _reject_interactive(args: ShellArgs) -> None:
        """Refuse what cannot work without a terminal (plan §14).

        Split deliberately: pagers and editors are always refused, but an
        interpreter is only interactive when it was given nothing to run —
        `python -c "..."` is a normal batch command an agent should be able to
        use, while a bare `python` would block forever on a closed stdin.
        """
        if args.argv:
            tokens = list(args.argv)
        else:
            try:
                tokens = shlex.split(args.command or "")
            except ValueError:
                return
        program = Path(tokens[0]).name.lower() if tokens else ""
        rest = tokens[1:]
        if program in REPL_COMMANDS and (not rest or "-i" in rest):
            raise ToolError(
                f"{program!r} with no program to run would wait for interactive input. "
                f"Give it something to execute (for example `{program} -c ...`)."
            )
        if program in INTERACTIVE_COMMANDS:
            raise ToolError(
                f"{program!r} needs an interactive terminal, which this tool does not "
                "provide. Use a non-interactive equivalent (for example `git log -n 20 "
                "--no-pager` instead of a pager)."
            )
        if program == "ssh" and "BatchMode=yes" not in (args.raw_command or ""):
            raise ToolError("ssh is refused unless it is non-interactive (BatchMode=yes)")

    def _spawn(
        self, argv: list[str], *, cwd: Path, env: dict[str, str], timeout: float,
        ctx: ToolContext, args: ShellArgs,
    ) -> ToolResult:
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":  # pragma: no cover - POSIX is the tested platform
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            # New session => its own process group => killpg reaches grandchildren.
            popen_kwargs["start_new_session"] = True

        started = time.monotonic()
        try:
            process = subprocess.Popen(  # noqa: S603 - argv list, never shell=True
                argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, errors="replace", **popen_kwargs,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"command not found: {argv[0]}") from exc
        except PermissionError as exc:
            raise ToolError(f"not executable: {argv[0]}") from exc
        except OSError as exc:
            raise ToolError(f"could not start command: {exc.strerror}") from exc

        deadline = started + timeout
        stdout = stderr = ""
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    if ctx.cancel.is_cancelled():
                        self._terminate(process)
                        raise UserCancelled("command cancelled") from None
                    if time.monotonic() >= deadline:
                        self._terminate(process)
                        raise CommandTimeout(
                            f"command exceeded {timeout:.0f}s and its process group was killed"
                        ) from None
        finally:
            if process.poll() is None:  # pragma: no cover - belt and braces
                self._terminate(process)

        duration_ms = int((time.monotonic() - started) * 1000)
        return self._format(process.returncode, stdout, stderr, args, ctx, duration_ms)

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        """SIGTERM the group, then SIGKILL what survives."""
        try:
            if os.name == "nt":  # pragma: no cover - POSIX is the tested platform
                process.kill()
                return
            group = os.getpgid(process.pid)
            os.killpg(group, signal.SIGTERM)
            for _ in range(20):
                if process.poll() is not None:
                    return
                time.sleep(0.05)
            os.killpg(group, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except OSError:  # pragma: no cover - already gone
                pass

    def _format(
        self, code: int | None, stdout: str, stderr: str, args: ShellArgs,
        ctx: ToolContext, duration_ms: int,
    ) -> ToolResult:
        """Streams stay labelled; the pipeline strips escapes and caps the total."""
        half = max(1024, ctx.max_output_bytes // 2)
        parts = [f"$ {args.raw_command}", f"exit code: {code}"]
        if stdout.strip():
            parts.append("--- stdout ---\n" + stdout[:half])
        if stderr.strip():
            parts.append("--- stderr ---\n" + stderr[:half])
        if not stdout.strip() and not stderr.strip():
            parts.append("(no output)")

        result = ToolResult(
            call_id="", name=self.name, ok=True, content="\n".join(parts),
            duration_ms=duration_ms,
        )
        result.metadata = {"exit_code": code, "command": args.raw_command}
        return result
