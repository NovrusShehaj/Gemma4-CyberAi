"""Terminal rendering (plan §8, §33).

Two rules govern this module:

1. **Nothing reaches the terminal unsanitised.** Model text and tool output are
   attacker-reachable (a file in the workspace can contain either), so every
   string goes through `sanitize_for_terminal` before Rich sees it.
2. **Rich markup is disabled for untrusted text.** ``[bold]`` in a file would
   otherwise be interpreted as formatting; worse, malformed markup raises inside
   the renderer. Untrusted content is printed with ``markup=False``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from gemma_cyber.agent.sanitize import sanitize_for_terminal
from gemma_cyber.agent.types import PermissionMode, Usage

__all__ = ["Renderer"]

_MODE_STYLE = {
    PermissionMode.READ_ONLY: "green",
    PermissionMode.WORKSPACE: "yellow",
    PermissionMode.AGENT: "dark_orange",
    PermissionMode.TRUSTED: "bold red",
}


class Renderer:
    """All user-facing output. Diagnostics go to stderr so `--json` stays clean."""

    def __init__(self, *, console: Console | None = None, quiet: bool = False) -> None:
        self.console = console or Console(soft_wrap=True)
        self.err = Console(stderr=True, soft_wrap=True)
        self.quiet = quiet
        self._streaming = False

    # -- model output -------------------------------------------------------

    def stream(self, delta: str) -> None:
        """Write a streamed token. Plain text: partial markdown cannot be parsed."""
        if self.quiet or not delta:
            return
        if not self._streaming:
            self.console.print("[dim]gemma4[/dim] ", end="")
            self._streaming = True
        self.console.print(sanitize_for_terminal(delta), end="", markup=False, highlight=False)

    def end_stream(self) -> None:
        if self._streaming:
            self.console.print()
            self._streaming = False

    # -- structured lines ---------------------------------------------------

    def tool_start(self, name: str, summary: str) -> None:
        if self.quiet:
            return
        self.end_stream()
        self.console.print(
            Text("→ ", style="cyan") + Text(sanitize_for_terminal(summary), style="cyan")
        )

    def tool_result(self, name: str, ok: bool, detail: str) -> None:
        if self.quiet:
            return
        self.end_stream()
        style = "dim" if ok else "red"
        marker = "←" if ok else "✗"
        self.console.print(
            Text(f"{marker} ", style=style) + Text(sanitize_for_terminal(detail), style=style)
        )

    def error(self, message: str, *, code: str = "") -> None:
        self.end_stream()
        prefix = f"[{code}] " if code else ""
        self.err.print(
            Text("error ", style="bold red")
            + Text(prefix + sanitize_for_terminal(message), style="red")
        )

    def warn(self, message: str) -> None:
        self.err.print(Text("warning ", style="yellow")
                       + Text(sanitize_for_terminal(message), style="yellow"))

    def info(self, message: str) -> None:
        if not self.quiet:
            self.console.print(Text(sanitize_for_terminal(message), style="dim"))

    def plain(self, message: str) -> None:
        """Untrusted content, printed verbatim with markup disabled."""
        self.console.print(sanitize_for_terminal(message), markup=False, highlight=False)

    def rule(self, title: str = "") -> None:
        if not self.quiet:
            self.console.rule(Text(sanitize_for_terminal(title), style="dim") if title else "")

    # -- rich blocks --------------------------------------------------------

    def diff(self, patch: str) -> None:
        if self.quiet or not patch.strip():
            return
        self.console.print(
            Syntax(sanitize_for_terminal(patch), "diff", theme="ansi_dark",
                   word_wrap=True, background_color="default")
        )

    def table(
        self, title: str, rows: Sequence[Sequence[str]], headers: Sequence[str]
    ) -> None:
        from rich.table import Table

        table = Table(title=title, header_style="bold", box=None, pad_edge=False)
        for header in headers:
            table.add_column(header)
        for row in rows:
            table.add_row(*[sanitize_for_terminal(str(cell)) for cell in row])
        self.console.print(table)

    # -- session chrome -----------------------------------------------------

    def banner(
        self, *, provider: str, model: str, mode: PermissionMode, workspace: str, version: str
    ) -> None:
        """Startup banner. States the honest model position every time (plan §8)."""
        style = _MODE_STYLE.get(mode, "white")
        self.console.print(
            Text("gemma4 ", style="bold")
            + Text(f"{version}  ", style="dim")
            + Text(f"{provider}://{model}", style="cyan")
            + Text("  mode=", style="dim") + Text(mode.value, style=style)
            + Text(f"  workspace={workspace}", style="dim")
        )
        self.console.print(
            Text("Answers are unverified. This is a general base model, not a promoted "
                 "cybersecurity model; verify technique IDs and CVEs before acting.",
                 style="dim italic")
        )
        if mode is PermissionMode.TRUSTED:
            self.console.print(
                Text("TRUSTED MODE: confirmations are disabled for this session.",
                     style="bold red")
            )

    def status(
        self, *, provider: str, model: str, mode: PermissionMode, workspace: str,
        session_id: str, iteration: int, usage: Usage,
    ) -> None:
        rows = [
            ("provider", provider), ("model", model), ("mode", mode.value),
            ("workspace", workspace), ("session", session_id), ("iteration", str(iteration)),
        ]
        if usage.total:
            rows.append(("tokens", f"in={usage.input_tokens} out={usage.output_tokens}"))
        self.table("status", rows, ("field", "value"))

    def usage_line(self, usage: Usage) -> None:
        if self.quiet or not usage.total:
            return
        self.console.print(
            Text(f"  tokens in={usage.input_tokens} out={usage.output_tokens}", style="dim")
        )

    @staticmethod
    def is_tty() -> bool:
        return sys.stdin.isatty() and sys.stdout.isatty()

    def print_json(self, payload: Any) -> None:
        """Machine output goes to stdout raw — never through Rich styling."""
        import json

        sys.stdout.write(json.dumps(payload, default=str) + "\n")
        sys.stdout.flush()
