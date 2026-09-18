"""The interactive REPL: Prompt Toolkit input + Rich output (plan §8).

Hybrid, not a full-screen TUI. Streaming output scrolls in the normal terminal
and stays greppable; a Textual app would buy a dashboard nobody needs for a
single sequential tool loop.

Ctrl+C semantics (plan §8): the first one cancels the work in flight; a second
one, or Ctrl+D at an empty prompt, exits. Cancelling sets the shared
`CancelToken`, which the provider thread and any running subprocess both poll —
so a cancel actually stops a `pytest` run rather than orphaning it.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import signal
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout

from gemma_cyber.agent.config import history_path
from gemma_cyber.agent.errors import AgentError, SessionError
from gemma_cyber.agent.events import EventKind
from gemma_cyber.agent.sessions import UndoStore
from gemma_cyber.agent.types import Message, PermissionMode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gemma_cyber.agent.cli import AgentApp

__all__ = ["Repl", "run_repl"]

SLASH_HELP = [
    ("/help", "show this list"),
    ("/status", "provider, model, mode, workspace, session, usage"),
    ("/mode [read-only|workspace|agent]", "change the permission ceiling"),
    ("/model [tag]", "show or switch the model tag"),
    ("/provider [name]", "show or switch the configured provider"),
    ("/tools", "tools available in the current mode"),
    ("/permissions", "what the current mode allows"),
    ("/files", "files read or modified this session"),
    ("/context", "what the next request would contain"),
    ("/diff", "changes this session made to the workspace"),
    ("/undo [path]", "restore the last snapshot of a file this session changed"),
    ("/compact", "summarise the conversation to free context"),
    ("/session [ls|show|rm <id>]", "list, inspect, or delete sessions"),
    ("/clear", "forget the conversation (the session file is kept)"),
    ("/exit", "quit"),
]


class Repl:
    """Owns the input loop and the slash commands. The agent loop lives in runtime."""

    def __init__(self, app: AgentApp) -> None:
        self.app = app
        self.renderer = app.renderer
        self._pending_exit = False
        self._session: PromptSession[str] = PromptSession(history=self._history())
        app.bus.subscribe(self._on_event)

    # -- wiring -------------------------------------------------------------

    @staticmethod
    def _history() -> FileHistory | InMemoryHistory:
        try:
            path = history_path()
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.touch(exist_ok=True)
            path.chmod(0o600)
            return FileHistory(str(path))
        except OSError:
            # A read-only home should degrade, not prevent the agent from running.
            return InMemoryHistory()

    def _on_event(self, event) -> None:
        if event.kind == EventKind.MODEL_TOKEN:
            self.renderer.stream(event.get("text", ""))
        elif event.kind == EventKind.TOOL_STARTED:
            self.renderer.tool_start(event.get("tool", ""), event.get("tool", ""))
        elif event.kind == EventKind.TOOL_COMPLETED:
            ok = bool(event.get("ok", True))
            self.renderer.tool_result(
                event.get("tool", ""), ok, f"{event.get('tool', '')} {'ok' if ok else 'failed'}"
            )

    # -- main loop ----------------------------------------------------------

    async def run(self) -> int:
        app = self.app
        self.renderer.banner(
            provider=app.provider_name, model=app.provider.model, mode=app.state.mode,
            workspace=str(app.workspace.root), version=app.version,
        )
        for warning in app.config_warnings:
            self.renderer.warn(warning)
        self.renderer.info("Type /help for commands, /exit to quit.")

        while True:
            try:
                with patch_stdout():
                    line = await self._session.prompt_async("you> ")
            except KeyboardInterrupt:
                if self._pending_exit:
                    break
                self._pending_exit = True
                self.renderer.info("(press Ctrl+C again or Ctrl+D to exit)")
                continue
            except EOFError:
                break

            self._pending_exit = False
            text = line.strip()
            if not text:
                continue
            if text.startswith("/"):
                if await self._slash(text) is False:
                    break
                continue

            await self._turn(text)

        self.renderer.info(f"session {app.state.session_id} saved")
        return 0

    async def _turn(self, text: str) -> None:
        app = self.app
        app.state.cancel.reset()
        loop = asyncio.get_running_loop()
        installed = False

        def on_sigint() -> None:
            app.state.cancel.cancel()
            self.renderer.warn("cancelling…")

        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signal.SIGINT, on_sigint)
            installed = True

        try:
            result = await app.runtime.run_turn(app.state, text)
        except KeyboardInterrupt:  # pragma: no cover - platform fallback
            app.state.cancel.cancel()
            self.renderer.warn("cancelled")
            return
        finally:
            if installed:
                with contextlib.suppress(NotImplementedError, RuntimeError):
                    loop.remove_signal_handler(signal.SIGINT)
            self.renderer.end_stream()

        if not result.ok and result.error:
            self.renderer.error(result.error, code=result.error_code or "")
        self.renderer.usage_line(result.usage)
        app.persist_meta()

    # -- slash commands -----------------------------------------------------

    async def _slash(self, text: str) -> bool | None:
        parts = text.split()
        command, args = parts[0].lower(), parts[1:]
        handlers = {
            "/help": self._help, "/status": self._status, "/tools": self._tools,
            "/permissions": self._permissions, "/files": self._files,
            "/context": self._context, "/diff": self._diff, "/clear": self._clear,
        }
        if command in ("/exit", "/quit"):
            return False
        if command in handlers:
            handlers[command]()
            return None
        if command == "/mode":
            self._mode(args)
        elif command == "/model":
            self._model(args)
        elif command == "/provider":
            self._provider(args)
        elif command == "/undo":
            self._undo(args)
        elif command == "/session":
            self._session_cmd(args)
        elif command == "/compact":
            await self._compact()
        else:
            self.renderer.warn(f"unknown command {command}; try /help")
        return None

    def _help(self) -> None:
        self.renderer.table("commands", [(c, d) for c, d in SLASH_HELP], ("command", "what"))

    def _status(self) -> None:
        app = self.app
        self.renderer.status(
            provider=app.provider_name, model=app.provider.model, mode=app.state.mode,
            workspace=str(app.workspace.root), session_id=app.state.session_id,
            iteration=app.state.iteration, usage=app.state.usage,
        )

    def _tools(self) -> None:
        rows = [
            (t.name, t.side_effect.value, "yes" if t.requires_confirm else "no", t.description)
            for t in self.app.registry.visible(self.app.guard)
        ]
        hidden = len(self.app.registry) - len(rows)
        self.renderer.table("tools", rows, ("tool", "side effect", "confirms", "what"))
        if hidden:
            self.renderer.info(f"{hidden} tool(s) hidden by mode={self.app.state.mode.value}")

    def _permissions(self) -> None:
        from gemma_cyber.agent.context import mode_banner

        self.renderer.plain(mode_banner(self.app.state.mode))
        self.renderer.info(
            "Credential paths and destructive commands are refused in every mode."
        )

    def _files(self) -> None:
        state = self.app.state
        rows = [(path, "read") for path in sorted(state.file_hashes)]
        touched = {str(p) for p in state.files_touched}
        rows += [(self.app.workspace.relative(Path(p)), "modified") for p in sorted(touched)]
        if not rows:
            self.renderer.info("no files touched yet")
            return
        self.renderer.table("files", rows, ("path", "how"))

    def _context(self) -> None:
        app = self.app
        try:
            packed = app.context.build(
                app.state, app.provider.capabilities(), model=app.provider.model,
                tools=app.registry.specs(app.guard),
            )
        except AgentError as exc:
            self.renderer.error(str(exc))
            return
        self.renderer.table(
            "context",
            [("messages", str(len(packed.request.messages))),
             ("estimated tokens", str(packed.estimated_tokens)),
             ("dropped messages", str(packed.dropped_messages)),
             ("dropped blocks", ", ".join(packed.dropped_blocks) or "none")],
            ("field", "value"),
        )

    def _diff(self) -> None:
        app = self.app
        store = UndoStore(app.undo_dir, app.state.session_id)
        originals: dict[str, bytes] = {}
        for record in store.entries():
            originals.setdefault(str(record.get("path")),
                                 (store.directory / str(record.get("snapshot"))).read_bytes()
                                 if (store.directory / str(record.get("snapshot"))).is_file()
                                 else b"")
        if not originals:
            self.renderer.info("no workspace changes recorded this session")
            return
        for rel, before in originals.items():
            try:
                after = (app.workspace.root / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            patch = "".join(difflib.unified_diff(
                before.decode("utf-8", errors="replace").splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{rel}", tofile=f"b/{rel}",
            ))
            if patch:
                self.renderer.diff(patch)

    def _undo(self, args: list[str]) -> None:
        app = self.app
        store = UndoStore(app.undo_dir, app.state.session_id)
        popped = store.pop_latest(args[0] if args else None)
        if popped is None:
            self.renderer.info("nothing to undo in this session")
            return
        rel, data = popped
        try:
            target = app.workspace.resolve_in_jail(rel)
            target.write_bytes(data)
        except (AgentError, OSError) as exc:
            self.renderer.error(f"could not restore {rel}: {exc}")
            return
        self.renderer.info(f"restored {rel}")

    def _mode(self, args: list[str]) -> None:
        app = self.app
        if not args:
            self.renderer.info(f"mode={app.state.mode.value}")
            return
        try:
            wanted = PermissionMode(args[0])
        except ValueError:
            self.renderer.error(f"unknown mode {args[0]!r}")
            return
        if wanted is PermissionMode.TRUSTED:
            # Startup-only, by design: trusted is a risk decision that needs an
            # explicit flag and a TTY, not a mid-session keystroke (plan §8).
            self.renderer.error(
                "trusted mode cannot be entered at runtime. Restart with "
                "`gemma4 --mode trusted --i-accept-risk`."
            )
            return
        app.set_mode(wanted)
        self.renderer.info(f"mode={wanted.value}")

    def _model(self, args: list[str]) -> None:
        if not args:
            self.renderer.info(f"model={self.app.provider.model}")
            return
        self.app.provider.model = args[0]
        self.renderer.info(f"model={args[0]} (unverified; capabilities may differ)")

    def _provider(self, args: list[str]) -> None:
        if not args:
            self.renderer.info(f"provider={self.app.provider_name}")
            return
        try:
            self.app.switch_provider(args[0])
        except AgentError as exc:
            self.renderer.error(str(exc))
            return
        self.renderer.info(f"provider={self.app.provider_name} model={self.app.provider.model}")

    def _session_cmd(self, args: list[str]) -> None:
        app = self.app
        sub = args[0] if args else "show"
        if sub == "ls":
            rows = [(m.id, m.model, m.mode, str(len(m.files_touched)))
                    for m in app.sessions.list(workspace=app.workspace.root)]
            self.renderer.table("sessions", rows, ("id", "model", "mode", "files"))
        elif sub == "rm" and len(args) > 1:
            try:
                app.sessions.delete(args[1])
            except SessionError as exc:
                self.renderer.error(str(exc))
                return
            self.renderer.info(f"deleted session {args[1]}")
        else:
            self.renderer.info(f"session {app.state.session_id}")

    async def _compact(self) -> None:
        summary = await self.app.runtime.compact(self.app.state)
        self.renderer.info("conversation compacted")
        if summary:
            self.renderer.plain(summary)

    def _clear(self) -> None:
        self.app.state.messages.clear()
        self.app.state.call_counts.clear()
        self.app.sessions.append(
            self.app.state.session_id,
            Message(role="tool", name="session", content="[conversation cleared]"),
        )
        self.renderer.info("conversation cleared (session file kept)")


async def run_repl(app: AgentApp) -> int:
    return await Repl(app).run()
