"""`gemma4` command surface and composition root (plan §8, §45).

This module parses flags, resolves configuration, constructs every dependency,
and hands control to the runtime or the REPL. It contains no agent loop, no
permission logic, and no tool behaviour — if a decision lives here that a test
should be able to make without a terminal, it is in the wrong place.

`gemma-cyber` is untouched. This is a second console script behind the `[agent]`
extra; a base install gets a `gemma4` that explains what to install and exits.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The Typer application, or None when the [agent] extra is not installed.
#: Declared up front so `main()` can be defined once, outside the import guard.
app: Any = None

_MISSING_EXTRA = (
    "gemma4 needs the optional [agent] extra.\n"
    "  uv pip install -e '.[agent]'      (from a checkout)\n"
    "  pipx install 'gemma-cyber[agent]' (from a release)\n"
)

try:
    import typer
except ImportError:  # pragma: no cover - exercised only in a base install
    # Deliberately NOT rebinding `typer` to a sentinel here: `app is None` is
    # already the "extra missing" signal, and rebinding the module name would
    # defeat static analysis of `typer.Option(...)` defaults below.
    pass
else:
    from gemma_cyber.agent import AGENT_VERSION
    from gemma_cyber.agent.audit import AuditLog
    from gemma_cyber.agent.config import (
        AgentConfig,
        LoadedConfig,
        ProviderConfig,
        audit_dir,
        data_dir,
        load_config,
        project_config_path,
        sessions_dir,
        undo_dir,
        user_config_path,
        write_default_config,
    )
    from gemma_cyber.agent.context import ContextManager
    from gemma_cyber.agent.errors import (
        AgentError,
        ConfigurationError,
        ProviderError,
        SessionError,
    )
    from gemma_cyber.agent.events import EventBus, EventKind
    from gemma_cyber.agent.permissions import PermissionGuard
    from gemma_cyber.agent.providers.base import ChatProvider
    from gemma_cyber.agent.providers.fake import FakeProvider, ScriptedTurn
    from gemma_cyber.agent.providers.ollama_chat import OllamaChatProvider
    from gemma_cyber.agent.runtime import AgentRuntime, TurnResult
    from gemma_cyber.agent.sessions import SessionStore, UndoStore
    from gemma_cyber.agent.tools import builtin_tools
    from gemma_cyber.agent.tools.base import (
        Approver,
        AutoApprover,
        PolicyApprover,
        ToolContext,
        ToolRegistry,
        ToolRuntime,
    )
    from gemma_cyber.agent.types import AgentState, PermissionMode
    from gemma_cyber.agent.ui.approvals import TtyApprover
    from gemma_cyber.agent.ui.render import Renderer
    from gemma_cyber.agent.workspace import Workspace, WorkspaceLimits, discover_root

    # -- exit codes (stable; scripts depend on them) ------------------------

    EXIT_OK = 0
    EXIT_ERROR = 1
    EXIT_PROVIDER_DOWN = 2
    EXIT_MODEL_UNAVAILABLE = 3
    EXIT_USAGE = 4
    EXIT_FINDINGS = 5
    EXIT_APPROVAL_REQUIRED = 6
    EXIT_CANCELLED = 7

    OUTPUT_SCHEMA_VERSION = 1

    app = typer.Typer(
        name="gemma4",
        help="Local, opt-in terminal agent for the Gemma4-CyberAI stack. "
             "Answers are unverified; the hosted API remains a no-tools product.",
        add_completion=False,
        no_args_is_help=False,
        pretty_exceptions_enable=False,
    )
    session_app = typer.Typer(help="Inspect and delete local sessions.")
    config_app = typer.Typer(help="Show or create the user configuration file.")
    app.add_typer(session_app, name="session")
    app.add_typer(config_app, name="config")

    # -- global options ------------------------------------------------------

    @dataclass(slots=True)
    class GlobalOptions:
        cwd: Path | None = None
        mode: str | None = None
        provider: str | None = None
        model: str | None = None
        profile: str | None = None
        allow_home: bool = False
        allow_network: bool = False
        allow_private_network: bool = False
        accept_risk: bool = False
        yes: bool = False
        json_out: bool = False
        jsonl_out: bool = False
        approval: str = "fail"
        debug: bool = False

    _OPTIONS = GlobalOptions()

    # -- the assembled application ------------------------------------------

    @dataclass
    class AgentApp:
        """Everything wired together. Built by :func:`build_app`, used by the REPL."""

        config: AgentConfig
        config_warnings: tuple[str, ...]
        workspace: Workspace
        guard: PermissionGuard
        registry: ToolRegistry
        provider: ChatProvider
        provider_name: str
        runtime: AgentRuntime
        context: ContextManager
        sessions: SessionStore
        state: AgentState
        renderer: Renderer
        bus: EventBus
        undo_dir: Path
        approver: Approver
        version: str = AGENT_VERSION
        session_meta: Any = None
        options: GlobalOptions = field(default_factory=GlobalOptions)

        def set_mode(self, mode: PermissionMode) -> None:
            """Change the ceiling at runtime. Trusted is refused by the caller."""
            self.state.mode = mode
            self.guard.mode = mode
            if self.session_meta is not None:
                self.session_meta.mode = mode.value

        def switch_provider(self, name: str) -> None:
            provider_config = self.config.provider(name)
            self.provider.close()
            self.provider = _make_provider(name, provider_config)
            self.provider_name = name

        def persist_meta(self) -> None:
            if self.session_meta is None:
                return
            self.session_meta.usage = self.state.usage
            self.session_meta.files_touched = [
                self.workspace.relative(p) for p in self.state.files_touched
            ]
            try:
                self.sessions.update_meta(self.session_meta)
            except SessionError:
                pass

    # -- construction --------------------------------------------------------

    def _cli_overrides(options: GlobalOptions) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        permissions: dict[str, Any] = {}
        if options.mode:
            permissions["mode"] = options.mode
        if options.allow_network:
            permissions["allow_network"] = True
        if options.allow_private_network:
            permissions["allow_private_network"] = True
        if options.allow_home:
            permissions["allow_home"] = True
        if permissions:
            overrides["permissions"] = permissions
        if options.provider:
            overrides["default_provider"] = options.provider
        if options.profile:
            overrides["profile"] = options.profile
        if options.model:
            target = options.provider or "__selected__"
            overrides.setdefault("providers", {})[target] = {"model": options.model}
        return overrides

    def _resolve_model_override(overrides: dict[str, Any], config_default: str) -> None:
        providers = overrides.get("providers")
        if isinstance(providers, dict) and "__selected__" in providers:
            providers[config_default] = providers.pop("__selected__")

    def _make_provider(name: str, provider_config: ProviderConfig) -> ChatProvider:
        if provider_config.type == "ollama":
            return OllamaChatProvider(
                model=provider_config.model, base_url=provider_config.base_url,
                timeout_s=provider_config.timeout_s,
                tools_native=provider_config.tools_native,
                context_tokens=provider_config.context_tokens, name=name,
            )
        if provider_config.type == "openai-compatible":
            from gemma_cyber.agent.providers.openai_compat import OpenAICompatProvider

            return OpenAICompatProvider(
                model=provider_config.model, base_url=provider_config.base_url,
                api_key=provider_config.api_key(), timeout_s=provider_config.timeout_s,
                tools_native=provider_config.tools_native,
                context_tokens=provider_config.context_tokens, name=name,
            )
        if provider_config.type == "fake":
            # Only reachable from an explicit config entry; used by smoke tests.
            return FakeProvider([ScriptedTurn(text="(fake provider)")], name=name,
                                model=provider_config.model)
        raise ConfigurationError(f"unsupported provider type: {provider_config.type}")

    def _resolve_mode(loaded: LoadedConfig, options: GlobalOptions, renderer: Renderer
                      ) -> PermissionMode:
        """Apply the trusted-mode gate (plan §15). Fails closed on every branch."""
        mode = loaded.config.permissions.mode
        if mode is not PermissionMode.TRUSTED:
            return mode
        if options.json_out or options.jsonl_out:
            raise ConfigurationError(
                "--mode trusted cannot be combined with machine output; "
                "trusted mode exists for interactive use only"
            )
        if not options.accept_risk:
            raise ConfigurationError(
                "--mode trusted requires --i-accept-risk. It disables every "
                "confirmation prompt for this session."
            )
        if not Renderer.is_tty():
            raise ConfigurationError(
                "--mode trusted requires an interactive terminal"
            )
        renderer.warn("trusted mode: confirmations are disabled for this session")
        return mode

    def build_app(
        options: GlobalOptions | None = None,
        *,
        provider: ChatProvider | None = None,
        renderer: Renderer | None = None,
        resume_session: str | None = None,
        quiet: bool = False,
    ) -> AgentApp:
        """The composition root. Importable so tests can wire an agent without a TTY."""
        options = options or _OPTIONS
        renderer = renderer or Renderer(quiet=quiet)

        root = discover_root(options.cwd, allow_home=options.allow_home)
        probe = load_config(workspace_root=root, cli_overrides=_cli_overrides(options))
        overrides = _cli_overrides(options)
        _resolve_model_override(overrides, probe.config.default_provider)
        loaded = load_config(workspace_root=root, cli_overrides=overrides)
        config = loaded.config

        if config.permissions.allow_home and options.cwd is None:
            root = discover_root(None, allow_home=True)

        mode = _resolve_mode(loaded, options, renderer)
        workspace = Workspace(root, limits=WorkspaceLimits(
            max_file_bytes=config.context.max_file_bytes,
            tree_max_entries=config.context.tree_max_entries,
            instructions_max_bytes=config.context.instructions_max_bytes,
        ))

        guard = PermissionGuard(
            workspace, mode,
            shell_allowlist=config.permissions.shell_allowlist,
            allow_network=config.permissions.allow_network,
            allow_private_network=config.permissions.allow_private_network,
        )
        registry = ToolRegistry(builtin_tools(config.profile))
        provider_name = config.default_provider
        chat_provider = provider or _make_provider(provider_name, config.provider())

        bus = EventBus()
        sessions = SessionStore(sessions_dir())
        undo_root = undo_dir()
        UndoStore.cleanup_expired(undo_root)

        if resume_session:
            loaded_session = sessions.load(resume_session)
            meta = loaded_session.meta
            state = AgentState(session_id=meta.id, mode=mode, messages=loaded_session.messages)
        else:
            created = sessions.create(
                workspace=str(workspace.root), provider=provider_name,
                model=chat_provider.model, mode=mode.value, profile=config.profile,
            )
            meta = created.meta
            state = AgentState(session_id=meta.id, mode=mode)

        approver = _make_approver(options, renderer)

        def context_factory(active: AgentState) -> ToolContext:
            return ToolContext(
                workspace=workspace, mode=active.mode, cancel=active.cancel,
                session_id=active.session_id, state=active,
                max_file_bytes=config.context.max_file_bytes,
                max_output_bytes=config.context.tool_output_max_bytes,
                shell_timeout_s=config.tools.shell_timeout_s,
                grep_max_results=config.tools.grep_max_results,
                glob_max_results=config.tools.glob_max_results,
                undo_dir=undo_root,
                extra={"allow_shell_string": not (options.json_out or options.jsonl_out)},
            )

        tool_runtime = ToolRuntime(
            registry, guard, context_factory=context_factory,
            audit=AuditLog(audit_dir()), approver=approver,
            max_output_bytes=config.context.tool_output_max_bytes,
            on_event=lambda kind, fields: bus.emit(kind, **fields),
        )
        context = ContextManager(
            workspace_stub=workspace.tree_stub(),
            project_hints=tuple(workspace.project_hints()),
            instruction_files=tuple(workspace.instruction_files()),
            tool_result_max_chars=config.context.tool_result_max_chars,
        )
        runtime = AgentRuntime(
            provider=chat_provider, registry=registry, tool_runtime=tool_runtime,
            context=context, sessions=sessions, bus=bus, guard=guard,
            max_iterations=config.context.max_iterations,
            turn_timeout_s=config.context.turn_timeout_s,
            loop_repeat_limit=config.context.loop_repeat_limit,
        )
        return AgentApp(
            config=config, config_warnings=loaded.warnings, workspace=workspace,
            guard=guard, registry=registry, provider=chat_provider,
            provider_name=provider_name, runtime=runtime, context=context,
            sessions=sessions, state=state, renderer=renderer, bus=bus,
            undo_dir=undo_root, approver=approver, session_meta=meta, options=options,
        )

    def _make_approver(options: GlobalOptions, renderer: Renderer) -> Approver:
        machine = options.json_out or options.jsonl_out
        if options.yes:
            # -y approves only what the mode already permits: the guard has
            # already run by the time an approver is consulted.
            return AutoApprover()
        if machine or not Renderer.is_tty():
            return PolicyApprover(options.approval)
        return TtyApprover(renderer)

    # -- global callback -----------------------------------------------------

    @app.callback(invoke_without_command=True)
    def main_callback(  # noqa: PLR0913 - a CLI surface is a wide function
        ctx: typer.Context,
        cwd: Path | None = typer.Option(None, "--cwd", help="Workspace root override."),
        mode: str | None = typer.Option(
            None, "--mode", help="read-only | workspace | agent | trusted"),
        provider: str | None = typer.Option(None, "--provider", help="Configured provider name."),
        model: str | None = typer.Option(None, "--model", help="Model tag."),
        profile: str | None = typer.Option(
            None, "--profile", help="general | coder | auditor"),
        allow_home: bool = typer.Option(False, "--allow-home",
                                        help="Permit $HOME as the workspace root."),
        allow_network: bool = typer.Option(False, "--allow-network",
                                           help="Permit network-capable tools."),
        allow_private_network: bool = typer.Option(
            False, "--allow-private-network", help="Also permit RFC1918 destinations."),
        accept_risk: bool = typer.Option(False, "--i-accept-risk",
                                         help="Required with --mode trusted."),
        yes: bool = typer.Option(False, "-y", "--yes",
                                 help="Auto-approve operations the mode already allows."),
        json_out: bool = typer.Option(False, "--json", help="Machine output (run)."),
        jsonl_out: bool = typer.Option(False, "--jsonl", help="Streaming machine output (run)."),
        approval: str = typer.Option(
            "fail", "--approval", help="Noninteractive policy: reject | fail | allowlist."),
        debug: bool = typer.Option(False, "--debug", help="Structured logs on stderr."),
        version: bool = typer.Option(False, "--version", help="Print the version and exit."),
    ) -> None:
        global _OPTIONS
        _OPTIONS = GlobalOptions(
            cwd=cwd, mode=mode, provider=provider, model=model, profile=profile,
            allow_home=allow_home, allow_network=allow_network,
            allow_private_network=allow_private_network, accept_risk=accept_risk,
            yes=yes, json_out=json_out, jsonl_out=jsonl_out, approval=approval, debug=debug,
        )
        if debug:
            import logging

            logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
        if version:
            typer.echo(f"gemma4 {AGENT_VERSION}")
            raise typer.Exit(EXIT_OK)
        if ctx.invoked_subcommand is None:
            raise typer.Exit(_run_interactive())

    # -- commands ------------------------------------------------------------

    def _run_interactive(resume: str | None = None) -> int:
        from gemma_cyber.agent.ui.repl import run_repl

        try:
            agent = build_app(_OPTIONS, resume_session=resume)
        except AgentError as exc:
            Renderer().error(str(exc), code=exc.code)
            return _exit_code_for(exc)
        try:
            return asyncio.run(run_repl(agent))
        except KeyboardInterrupt:  # pragma: no cover - interactive path
            return EXIT_CANCELLED
        finally:
            agent.persist_meta()
            agent.provider.close()

    @app.command("ask")
    def cmd_ask(prompt: str = typer.Argument(..., help="One-shot question. No tools are used.")
                ) -> None:
        """Ask one question with no tools and no session (plan §8)."""
        renderer = Renderer()
        try:
            agent = build_app(_OPTIONS, renderer=renderer)
        except AgentError as exc:
            renderer.error(str(exc), code=exc.code)
            raise typer.Exit(_exit_code_for(exc)) from None
        # No tools: an `ask` must not be able to touch the filesystem at all.
        agent.registry = ToolRegistry([])
        agent.runtime.registry = agent.registry
        agent.bus.subscribe(
            lambda e: renderer.stream(e.get("text", ""))
            if e.kind == EventKind.MODEL_TOKEN else None
        )
        try:
            result = asyncio.run(agent.runtime.run_turn(agent.state, prompt))
        finally:
            renderer.end_stream()
            agent.provider.close()
        if not result.ok:
            renderer.error(result.error or "failed", code=result.error_code or "")
            raise typer.Exit(EXIT_ERROR)

    @app.command("run")
    def cmd_run(
        prompt: str = typer.Argument(..., help="One-shot agent task."),
        fail_on_findings: bool = typer.Option(
            False, "--fail-on-findings", help="Exit 5 if the run reported findings."),
    ) -> None:
        """Run one agent turn noninteractively; `--json` makes it CI-shaped."""
        options = _OPTIONS
        machine = options.json_out or options.jsonl_out
        renderer = Renderer(quiet=machine)
        try:
            agent = build_app(options, renderer=renderer, quiet=machine)
        except AgentError as exc:
            if machine:
                renderer.print_json({"schema": OUTPUT_SCHEMA_VERSION, "ok": False,
                                     "error": str(exc), "error_code": exc.code})
            else:
                renderer.error(str(exc), code=exc.code)
            raise typer.Exit(_exit_code_for(exc)) from None

        if options.jsonl_out:
            agent.bus.subscribe(lambda e: renderer.print_json(
                {"schema": OUTPUT_SCHEMA_VERSION, "event": e.kind, **e.fields}
            ))
        elif not machine:
            agent.bus.subscribe(
                lambda e: renderer.stream(e.get("text", ""))
                if e.kind == EventKind.MODEL_TOKEN else None
            )

        try:
            result = asyncio.run(agent.runtime.run_turn(agent.state, prompt))
        finally:
            renderer.end_stream()
            agent.persist_meta()
            agent.provider.close()

        code = _run_exit_code(agent, result, fail_on_findings=fail_on_findings)
        if machine:
            renderer.print_json(_run_payload(agent, result, code))
        elif not result.ok and result.error:
            renderer.error(result.error, code=result.error_code or "")
        raise typer.Exit(code)

    def _run_payload(agent: AgentApp, result: TurnResult, code: int) -> dict[str, Any]:
        return {
            "schema": OUTPUT_SCHEMA_VERSION,
            "ok": result.ok,
            "exit_code": code,
            "session_id": agent.state.session_id,
            "stop_reason": result.stop_reason,
            "final_text": result.final_text,
            "iterations": result.iterations,
            "tool_calls": result.tool_calls,
            "files_touched": result.files_touched,
            "usage": result.usage.to_dict(),
            "mode": agent.state.mode.value,
            "provider": agent.provider_name,
            "model": agent.provider.model,
            "error": result.error,
            "error_code": result.error_code,
            "approval_blocked": bool(getattr(agent.approver, "blocked", False)),
        }

    def _run_exit_code(agent: AgentApp, result: TurnResult, *, fail_on_findings: bool) -> int:
        if getattr(agent.approver, "blocked", False):
            return EXIT_APPROVAL_REQUIRED
        if result.error_code == "cancelled":
            return EXIT_CANCELLED
        if not result.ok:
            return EXIT_ERROR
        if fail_on_findings and _looks_like_findings(result.final_text):
            return EXIT_FINDINGS
        return EXIT_OK

    def _looks_like_findings(text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in ("finding", "vulnerab", "cve-", "secret found"))

    @app.command("resume")
    def cmd_resume(session_id: str | None = typer.Argument(None)) -> None:
        """Resume a session; with no id, the newest one for this workspace."""
        target = session_id
        if target is None:
            store = SessionStore(sessions_dir())
            try:
                root = discover_root(_OPTIONS.cwd, allow_home=_OPTIONS.allow_home)
            except AgentError as exc:
                Renderer().error(str(exc), code=exc.code)
                raise typer.Exit(_exit_code_for(exc)) from None
            latest = store.latest(workspace=root)
            if latest is None:
                Renderer().error("no previous session for this workspace")
                raise typer.Exit(EXIT_ERROR)
            target = latest.id
        raise typer.Exit(_run_interactive(resume=target))

    @session_app.command("ls")
    def cmd_session_ls(
        all_workspaces: bool = typer.Option(False, "--all", help="Include other workspaces.")
    ) -> None:
        renderer = Renderer()
        store = SessionStore(sessions_dir())
        workspace = None
        if not all_workspaces:
            workspace = discover_root(_OPTIONS.cwd, allow_home=True)
        rows = [
            (m.id, m.model, m.mode, m.profile, m.workspace)
            for m in store.list(workspace=workspace)
        ]
        if not rows:
            renderer.info("no sessions")
            return
        renderer.table("sessions", rows, ("id", "model", "mode", "profile", "workspace"))

    @session_app.command("show")
    def cmd_session_show(session_id: str) -> None:
        renderer = Renderer()
        try:
            session = SessionStore(sessions_dir()).load(session_id)
        except SessionError as exc:
            renderer.error(str(exc), code=exc.code)
            raise typer.Exit(EXIT_ERROR) from None
        renderer.table(
            f"session {session.meta.id}",
            [("workspace", session.meta.workspace), ("model", session.meta.model),
             ("mode", session.meta.mode), ("messages", str(len(session.messages)))],
            ("field", "value"),
        )
        for message in session.messages:
            renderer.plain(f"[{message.role}] {message.content[:2000]}")

    @session_app.command("rm")
    def cmd_session_rm(session_id: str) -> None:
        renderer = Renderer()
        try:
            SessionStore(sessions_dir()).delete(session_id)
        except SessionError as exc:
            renderer.error(str(exc), code=exc.code)
            raise typer.Exit(EXIT_ERROR) from None
        renderer.info(f"deleted session {session_id}")

    @app.command("tools")
    def cmd_tools() -> None:
        """List tools and what each would be allowed to do in the current mode."""
        renderer = Renderer()
        try:
            root = discover_root(_OPTIONS.cwd, allow_home=_OPTIONS.allow_home)
            loaded = load_config(workspace_root=root, cli_overrides=_cli_overrides(_OPTIONS))
        except AgentError as exc:
            renderer.error(str(exc), code=exc.code)
            raise typer.Exit(_exit_code_for(exc)) from None
        guard = PermissionGuard(Workspace(root), loaded.config.permissions.mode)
        rows = [
            (t.name, t.side_effect.value,
             "visible" if guard.is_visible(t) else f"hidden in {guard.mode.value}",
             t.description[:70])
            for t in ToolRegistry(builtin_tools(loaded.config.profile)).all()
        ]
        renderer.table("tools", rows, ("tool", "side effect", "availability", "what"))

    @app.command("models")
    def cmd_models() -> None:
        """Show provider capabilities honestly — including what is unknown."""
        renderer = Renderer()
        try:
            root = discover_root(_OPTIONS.cwd, allow_home=True)
            loaded = load_config(workspace_root=root, cli_overrides=_cli_overrides(_OPTIONS))
        except AgentError as exc:
            renderer.error(str(exc), code=exc.code)
            raise typer.Exit(_exit_code_for(exc)) from None

        rows: list[tuple[str, ...]] = []
        for name, provider_config in sorted(loaded.config.providers.items()):
            try:
                provider = _make_provider(name, provider_config)
            except AgentError as exc:
                rows.append((name, provider_config.type, provider_config.model, str(exc)))
                continue
            caps = provider.capabilities()
            protocol = "native" if caps.tools_native else "xml-codec (emulated)"
            context = str(caps.context_tokens) if caps.context_tokens else "unknown"
            rows.append((name, provider_config.type, provider.model, protocol, context))
            provider.close()
        renderer.table(
            "providers", rows, ("name", "type", "model", "tool protocol", "context tokens")
        )
        renderer.info(
            "Native tool calling is reported, not assumed. A small local model will "
            "call tools unreliably; a larger local model is recommended for --mode agent."
        )

    @app.command("doctor")
    def cmd_doctor() -> None:
        """Check the local setup. Reports facts; never prints a credential."""
        renderer = Renderer()
        checks: list[tuple[str, str, str]] = []
        ok = True

        checks.append(("agent extra", "ok", f"gemma4 {AGENT_VERSION}"))
        checks.append(("data dir", "ok", str(data_dir())))
        user_path = user_config_path()
        checks.append(("user config", "ok" if user_path.is_file() else "absent", str(user_path)))

        try:
            root = discover_root(_OPTIONS.cwd, allow_home=_OPTIONS.allow_home)
            checks.append(("workspace", "ok", str(root)))
        except AgentError as exc:
            checks.append(("workspace", "FAIL", str(exc)))
            renderer.table("doctor", checks, ("check", "status", "detail"))
            raise typer.Exit(EXIT_ERROR) from None

        project_path = project_config_path(root)
        checks.append(("project config",
                       "ok" if project_path.is_file() else "absent", str(project_path)))

        try:
            loaded = load_config(workspace_root=root, cli_overrides=_cli_overrides(_OPTIONS))
        except ConfigurationError as exc:
            checks.append(("config", "FAIL", str(exc)))
            renderer.table("doctor", checks, ("check", "status", "detail"))
            raise typer.Exit(EXIT_ERROR) from None

        for warning in loaded.warnings:
            ok = False
            checks.append(("config warning", "WARN", warning))
        checks.append(("permission mode", "ok", loaded.config.permissions.mode.value))
        checks.append(("profile", "ok", loaded.config.profile))

        workspace = Workspace(root)
        ignore_files = [
            name for name in (".gitignore", ".gemma4ignore") if (root / name).is_file()
        ]
        checks.append(("ignore files", "ok" if ignore_files else "absent",
                       ", ".join(ignore_files) or "none"))
        instructions = [label for label, _ in workspace.instruction_files()]
        checks.append(("project instructions", "ok" if instructions else "absent",
                       ", ".join(instructions) or "none (untrusted when present)"))

        provider_config = loaded.config.provider()
        checks.append(("provider", "ok",
                       f"{provider_config.type} {provider_config.base_url}"))
        try:
            provider = _make_provider(loaded.config.default_provider, provider_config)
        except AgentError as exc:
            checks.append(("provider build", "FAIL", str(exc)))
            provider = None
            ok = False
        if provider is not None:
            lister = getattr(provider, "list_models", None)
            if lister is None:
                checks.append(("model runtime", "skip", "provider cannot list models"))
            else:
                try:
                    tags = lister()
                except ProviderError as exc:
                    ok = False
                    checks.append(("model runtime", "FAIL", str(exc)))
                else:
                    checks.append(("model runtime", "ok", f"{len(tags)} model(s) available"))
                    present = provider_config.model in tags
                    if not present:
                        ok = False
                    checks.append((
                        "configured model", "ok" if present else "FAIL",
                        provider_config.model if present
                        else f"{provider_config.model} not pulled (`ollama pull "
                             f"{provider_config.model}`)",
                    ))
            provider.close()

        renderer.table("doctor", checks, ("check", "status", "detail"))
        # Table cells are width-limited, and a truncated failure message is
        # useless. Repeat every non-ok detail in full underneath.
        for name, status, detail in checks:
            if status in ("FAIL", "WARN"):
                renderer.plain(f"{status}: {name} — {detail}")
        renderer.info(
            "gemma4 is a local, opt-in agent. The hosted API and `gemma-cyber` remain "
            "no-tools surfaces. Answers are unverified."
        )
        raise typer.Exit(EXIT_OK if ok else EXIT_ERROR)

    @config_app.command("path")
    def cmd_config_path() -> None:
        renderer = Renderer()
        rows = [("user config", str(user_config_path())), ("data dir", str(data_dir()))]
        try:
            root = discover_root(_OPTIONS.cwd, allow_home=True)
            rows.append(("project config", str(project_config_path(root))))
        except AgentError:
            pass
        rows += [("sessions", str(sessions_dir())), ("undo", str(undo_dir())),
                 ("audit", str(audit_dir()))]
        renderer.table("paths", rows, ("what", "path"))

    @config_app.command("init")
    def cmd_config_init(
        force: bool = typer.Option(False, "--force", help="Overwrite an existing file.")
    ) -> None:
        renderer = Renderer()
        path = user_config_path()
        if path.exists() and not force:
            renderer.error(f"{path} already exists; pass --force to overwrite")
            raise typer.Exit(EXIT_USAGE)
        write_default_config(path)
        renderer.info(f"wrote {path}")

    # -- helpers -------------------------------------------------------------

    def _exit_code_for(exc: AgentError) -> int:
        if isinstance(exc, ConfigurationError):
            return EXIT_USAGE
        if isinstance(exc, ProviderError):
            return EXIT_PROVIDER_DOWN
        return EXIT_ERROR


def main() -> int:
    """Console-script entry point.

    A base install has a working `gemma4` binary that explains what to install
    rather than an ImportError traceback — tools never appear by accident, and
    their absence is never a crash.
    """
    if app is None:  # pragma: no cover - exercised only in a base install
        sys.stderr.write(_MISSING_EXTRA)
        return 4
    try:
        app()
    except SystemExit as exc:  # typer.Exit propagates as SystemExit
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
