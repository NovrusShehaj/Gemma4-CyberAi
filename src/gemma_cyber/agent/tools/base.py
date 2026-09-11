"""Tool contract, registry, and the one execution pipeline (plan §11).

`ToolRuntime.execute` is the only way a tool ever runs. Every step in it exists
because skipping it has a specific failure mode:

===  ==========================  ====================================================
#    step                        what it prevents
===  ==========================  ====================================================
1    lookup                      executing a hallucinated tool name
2    pydantic validation         a wrong-typed argument reaching filesystem code
3    `PermissionGuard.authorize` the model authorising itself
4    approval                    a silent side effect the user did not agree to
5    timeout                     a hung command holding the turn forever
6    truncate + digest           an unbounded file eating the context window
7    strip ANSI                  terminal injection on display *and* on re-injection
8    redact                      a credential persisting into `messages.jsonl`
9    audit                       an undecidable "what did it do last Tuesday"
===  ==========================  ====================================================

A tool that wants to skip a step does not get to; the pipeline owns all of them.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from gemma_cyber.agent.audit import AuditLog, AuditRecord, agent_logger
from gemma_cyber.agent.errors import (
    AgentError,
    CommandTimeout,
    PermissionDenied,
    ToolError,
    ToolNotFound,
    UserCancelled,
    safe_message,
)
from gemma_cyber.agent.permissions import Decision, PermissionGuard
from gemma_cyber.agent.sanitize import redact_secrets, strip_ansi, truncate_with_digest
from gemma_cyber.agent.types import (
    AgentState,
    CancelToken,
    PermissionMode,
    SideEffect,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from gemma_cyber.agent.workspace import Workspace

__all__ = [
    "ApprovalRequest",
    "Approver",
    "AutoApprover",
    "BuiltinToolSource",
    "DenyingApprover",
    "PolicyApprover",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolRuntime",
    "ToolSource",
]


# -- context ----------------------------------------------------------------

@dataclass(slots=True)
class ToolContext:
    """Everything a tool may touch. Note what is absent: no CLI, no UI, no provider."""

    workspace: Workspace
    mode: PermissionMode
    cancel: CancelToken
    session_id: str
    state: AgentState
    #: Budgets from `ContextConfig`/`ToolsConfig`, passed as plain values so tools
    #: do not depend on the config module's shape.
    max_file_bytes: int = 1_048_576
    max_output_bytes: int = 32_768
    shell_timeout_s: float = 60.0
    grep_max_results: int = 200
    glob_max_results: int = 500
    #: Where `fs.edit` writes undo snapshots. None disables snapshotting.
    undo_dir: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


# -- tool contract ----------------------------------------------------------

class Tool(ABC):
    """Base class for every tool.

    Subclasses set the class attributes and implement :meth:`run`. The metadata
    is not documentation — `PermissionGuard` authorises on ``side_effect``,
    ``path_fields`` and ``command_field``, so an omission is a security hole, not
    a cosmetic lapse.
    """

    name: str = ""
    description: str = ""
    input_model: type[BaseModel]
    side_effect: SideEffect = SideEffect.READ
    timeout_s: float = 30.0
    requires_confirm: bool = False
    network: bool = False
    mutating: bool = False
    #: Argument names holding workspace paths. The guard jails and denylist-checks
    #: each of these before the tool sees them.
    path_fields: tuple[str, ...] = ()
    #: Argument holding the proposed command, for the destructive-command policy.
    command_field: str | None = None

    @abstractmethod
    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        """Execute. Called on a worker thread, after validation and authorisation.

        Raise :class:`ToolError` (or a subclass) for a recoverable failure; the
        pipeline converts it into a structured result the model can act on.
        """

    def to_spec(self) -> ToolSpec:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=schema,
            side_effect=self.side_effect,
        )

    def ok(self, content: str, **metadata: Any) -> ToolResult:
        """Convenience constructor; truncation/sanitising happens in the pipeline."""
        return ToolResult(call_id="", name=self.name, ok=True, content=content,
                          metadata=metadata)


# -- registry ---------------------------------------------------------------

class ToolSource(Protocol):
    """Where tools come from. The only source in v1 is `BuiltinToolSource`.

    The port exists now so that a future source (setuptools entry points, an MCP
    client — plan §21) plugs in *behind* `ToolRegistry.add_source`, which forces
    every discovered tool through the same metadata validation and therefore the
    same `PermissionGuard`. A plugin can add a tool; it can never add a tool that
    skips authorisation, change the process mode, or disable the jail.
    """

    #: Identifies the source in `gemma4 tools` and in the audit trail.
    name: str

    def provide(self) -> list[Tool]: ...


class BuiltinToolSource:
    """The in-code toolbelt. Trusted because it ships with the package."""

    name = "builtin"

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools = list(tools)

    def provide(self) -> list[Tool]:
        return list(self._tools)


class ToolRegistry:
    """In-process registry. Every tool enters through `register`, whatever its source."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self._origin: dict[str, str] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool, *, source: str = "builtin") -> None:
        """Validate and admit one tool.

        The metadata checks are not cosmetic: `PermissionGuard` authorises on
        `side_effect`, and a tool arriving from an untrusted source with a
        missing or invented side effect must be rejected here rather than
        default into something permissive.
        """
        if not tool.name:
            raise ValueError("a tool must have a name")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        if not isinstance(tool.side_effect, SideEffect):
            raise ValueError(
                f"tool {tool.name!r} from source {source!r} declares an unknown "
                f"side effect {tool.side_effect!r}; refusing to register it"
            )
        if not isinstance(getattr(tool, "input_model", None), type):
            raise ValueError(f"tool {tool.name!r} has no pydantic input_model")
        self._tools[tool.name] = tool
        self._origin[tool.name] = source

    def add_source(self, source: ToolSource) -> None:
        """Admit every tool a source provides, under the same validation."""
        for tool in source.provide():
            self.register(tool, source=source.name)

    def origin(self, name: str) -> str:
        """Which source provided ``name``. Recorded for review, not for policy."""
        return self._origin.get(name, "unknown")

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(
                f"no tool named {name!r}. Available: {', '.join(sorted(self._tools)) or 'none'}"
            ) from None

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in sorted(self._tools)]

    def visible(self, guard: PermissionGuard) -> list[Tool]:
        """Tools the current mode can actually run — the only ones the model sees."""
        return [tool for tool in self.all() if guard.is_visible(tool)]

    def specs(self, guard: PermissionGuard) -> list[ToolSpec]:
        return [tool.to_spec() for tool in self.visible(guard)]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# -- approvals --------------------------------------------------------------

@dataclass(slots=True)
class ApprovalRequest:
    """What the user is being asked to allow."""

    tool: str
    summary: str
    decision: Decision
    arguments: dict[str, Any]
    #: Rendered unified diff, when the tool can show one (fs.edit / fs.write).
    preview: str = ""
    side_effect: SideEffect = SideEffect.READ


class Approver(Protocol):
    """Decides confirmations. TTY prompt, `-y`, or a noninteractive policy."""

    async def approve(self, request: ApprovalRequest) -> bool: ...


class AutoApprover:
    """`-y/--yes`: approve anything the mode already permits.

    It cannot escalate: it is consulted *after* `PermissionGuard.authorize` has
    already allowed the call, so a denied operation never reaches it.
    """

    def __init__(self, *, record: list[ApprovalRequest] | None = None) -> None:
        self.seen: list[ApprovalRequest] = record if record is not None else []

    async def approve(self, request: ApprovalRequest) -> bool:
        self.seen.append(request)
        return True


class PolicyApprover:
    """Noninteractive approval policy for `gemma4 run` (plan §10).

    Three behaviours, none of which is "allow":

    * ``reject`` — decline the call and let the model try something else;
    * ``fail`` — decline and mark the run as blocked, so the CLI exits nonzero
      (the default: a CI job that silently skipped its edits is worse than one
      that failed);
    * ``allowlist`` — approve only calls the guard already marked as needing no
      confirmation for an allowlisted reason.
    """

    def __init__(self, policy: str = "fail") -> None:
        if policy not in ("reject", "fail", "allowlist"):
            raise ValueError(f"unknown approval policy: {policy}")
        self.policy = policy
        self.blocked = False
        self.seen: list[ApprovalRequest] = []

    async def approve(self, request: ApprovalRequest) -> bool:
        self.seen.append(request)
        if self.policy == "allowlist" and not request.decision.requires_confirm:
            return True  # pragma: no cover - the guard would not have asked
        if self.policy == "fail":
            self.blocked = True
        return False


class DenyingApprover:
    """Noninteractive default: refuse anything that needs a human (plan §10).

    `gemma4 run --json` in CI has no TTY. Implicit approval there would mean a
    scheduled job silently rewriting files, so the default is refusal.
    """

    def __init__(self, *, reason: str = "approval required but no TTY is available") -> None:
        self.reason = reason
        self.seen: list[ApprovalRequest] = []

    async def approve(self, request: ApprovalRequest) -> bool:
        self.seen.append(request)
        return False


# -- the pipeline -----------------------------------------------------------

class ToolRuntime:
    """Owns the nine-step execution pipeline. Tools never bypass it."""

    def __init__(
        self,
        registry: ToolRegistry,
        guard: PermissionGuard,
        *,
        context_factory,
        audit: AuditLog,
        approver: Approver,
        max_output_bytes: int = 32_768,
        on_event=None,
    ) -> None:
        self._registry = registry
        self._guard = guard
        self._context_factory = context_factory
        self._audit = audit
        self._approver = approver
        self._max_output_bytes = max_output_bytes
        self._on_event = on_event

    async def execute(self, call: ToolCall, state: AgentState) -> ToolResult:
        started = time.monotonic()
        workspace_root = self._guard.workspace.root

        try:
            tool = self._registry.get(call.name)
        except ToolNotFound as exc:
            return self._record_failure(call, state, exc, started, tool_known=False)

        if not self._guard.is_visible(tool):
            # Hidden by mode: report it as unavailable rather than advertising
            # that a privileged tool exists behind a different flag.
            hidden = ToolNotFound(
                f"tool {call.name!r} is not available in {state.mode.value} mode"
            )
            return self._record_failure(call, state, hidden, started)

        try:
            args = tool.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            return self._record_failure(
                call, state, ToolError(_format_validation(exc)), started
            )

        try:
            decision = self._guard.authorize(tool, args, state)
        except PermissionDenied as exc:
            return self._record_failure(call, state, exc, started, decision="deny")

        if decision.requires_confirm:
            request = ApprovalRequest(
                tool=tool.name,
                summary=self._summarise(tool, args),
                decision=decision,
                arguments=_dump(args),
                preview=self._preview(tool, args, state),
                side_effect=tool.side_effect,
            )
            self._emit("tool_approval_requested", tool=tool.name)
            approved = await self._approver.approve(request)
            if not approved:
                rejected = PermissionDenied(
                    f"{tool.name} was not approved by the user. Do not retry it; "
                    "ask what to do differently instead."
                )
                return self._record_failure(call, state, rejected, started, decision="rejected")
            if tool.side_effect is SideEffect.WORKSPACE_WRITE:
                key = self._guard.write_key(tool, args)
                if key:
                    state.approved_writes.add(key)

        ctx = self._context_factory(state)
        self._emit("tool_started", tool=tool.name)

        try:
            result = await self._run_with_timeout(tool, args, ctx)
        except UserCancelled as exc:
            return self._record_failure(call, state, exc, started, decision="cancelled")
        except AgentError as exc:
            return self._record_failure(call, state, exc, started)
        except Exception as exc:  # noqa: BLE001 - a tool bug must not end the session
            agent_logger().exception("tool crashed", extra={"tool": tool.name})
            wrapped = ToolError(
                f"{tool.name} failed unexpectedly ({exc.__class__.__name__})"
            )
            return self._record_failure(call, state, wrapped, started)

        result = self._finalise(result, call, tool, started, workspace_root)
        self._audit.write(AuditRecord(
            session_id=state.session_id, tool=tool.name, decision="allow",
            rule=decision.rule, args=_dump(args), mode=state.mode.value, ok=result.ok,
            duration_ms=result.duration_ms, truncated=result.truncated,
            exit_code=result.metadata.get("exit_code"),
        ))
        self._emit("tool_completed", tool=tool.name, ok=result.ok)
        return result

    # -- internals ----------------------------------------------------------

    async def _run_with_timeout(self, tool: Tool, args: BaseModel, ctx: ToolContext) -> ToolResult:
        """Run on a worker thread under a wall-clock backstop.

        A tool that owns a subprocess enforces its own timeout first (and kills
        the process group); this backstop adds a grace window so a tool that
        hangs *outside* its own timeout still cannot hold the turn open.
        """
        budget = tool.timeout_s + 2.0
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(tool.run, args, ctx), timeout=budget
            )
        except TimeoutError as exc:
            ctx.cancel.cancel()
            raise CommandTimeout(
                f"{tool.name} exceeded its {tool.timeout_s:.0f}s budget and was abandoned"
            ) from exc

    def _finalise(
        self, result: ToolResult, call: ToolCall, tool: Tool, started: float, root
    ) -> ToolResult:
        """Steps 6–8: bound, strip, redact. Applied to every result, always."""
        result.call_id = call.id
        result.name = tool.name
        result.duration_ms = int((time.monotonic() - started) * 1000)

        truncation = truncate_with_digest(result.content, self._max_output_bytes)
        cleaned = redact_secrets(strip_ansi(truncation.text))
        result.content = cleaned
        if truncation.truncated:
            result.truncated = True
            result.remainder_sha256 = truncation.remainder_sha256
        if result.error:
            result.error = redact_secrets(strip_ansi(safe_message(Exception(result.error),
                                                                 workspace_root=root)))
        return result

    def _record_failure(
        self,
        call: ToolCall,
        state: AgentState,
        exc: AgentError,
        started: float,
        *,
        decision: str = "deny",
        tool_known: bool = True,
    ) -> ToolResult:
        """Turn a failure into a structured result the model can read and recover from."""
        message = safe_message(exc, workspace_root=self._guard.workspace.root)
        message = redact_secrets(strip_ansi(message))
        duration = int((time.monotonic() - started) * 1000)
        self._audit.write(AuditRecord(
            session_id=state.session_id,
            tool=call.name if tool_known else f"{call.name} (unknown)",
            decision=decision, args=call.arguments, mode=state.mode.value,
            ok=False, error_code=exc.code, duration_ms=duration,
        ))
        self._emit("tool_completed", tool=call.name, ok=False)
        return ToolResult(
            call_id=call.id, name=call.name, ok=False, error=message,
            error_code=exc.code, duration_ms=duration,
        )

    @staticmethod
    def _summarise(tool: Tool, args: BaseModel) -> str:
        dumped = _dump(args)
        if tool.command_field and (raw := dumped.get(tool.command_field)):
            command = " ".join(raw) if isinstance(raw, list) else str(raw)
            return f"{tool.name}: {command}"
        for field_name in tool.path_fields:
            if value := dumped.get(field_name):
                return f"{tool.name}: {value}"
        return tool.name

    @staticmethod
    def _preview(tool: Tool, args: BaseModel, state: AgentState) -> str:
        builder = getattr(tool, "preview", None)
        if builder is None:
            return ""
        try:
            return str(builder(args, state))
        except Exception:  # noqa: BLE001 - a preview must never block a decision
            return ""

    def _emit(self, kind: str, **fields: Any) -> None:
        if self._on_event is not None:
            self._on_event(kind, fields)


def _dump(args: BaseModel) -> dict[str, Any]:
    try:
        return args.model_dump(mode="json")
    except Exception:  # pragma: no cover - defensive
        return {}


def _format_validation(exc: ValidationError) -> str:
    """A small model corrects itself from a precise message, not a stack trace."""
    parts: list[str] = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"{loc or 'arguments'}: {err.get('msg', 'invalid')}")
    return "invalid arguments — " + "; ".join(parts) + ". Fix them and call again."
