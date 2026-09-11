"""`AgentRuntime` — one single-agent, sequential tool loop (plan §10).

Not a planner/executor. Not subagents. Not parallel tool execution. One model,
one tool at a time, bounded iterations. That restraint is the design: a 4B-class
model with an unbounded loop and concurrent writes produces damage, not results.

Two properties matter more than throughput:

* **Everything is persisted as it happens.** If the process dies between a tool
  result and the next model call, `gemma4 resume` picks the session back up.
* **Every exit is bounded and named.** No tool calls, iteration cap, wall-clock
  cap, repeated-call detector, cancel, or a typed error — never "it is still
  going".
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from gemma_cyber.agent.audit import agent_logger
from gemma_cyber.agent.codec import (
    SYNTAX_REMINDER,
    ToolCallStreamParser,
    looks_like_attempted_call,
)
from gemma_cyber.agent.context import ContextManager
from gemma_cyber.agent.errors import (
    AgentError,
    AuthenticationError,
    ContextOverflowError,
    LoopDetected,
    ProviderError,
    RateLimitError,
    TurnTimeout,
    UserCancelled,
    safe_message,
)
from gemma_cyber.agent.events import EventBus, EventKind
from gemma_cyber.agent.providers.base import ChatProvider
from gemma_cyber.agent.sessions import SessionStore
from gemma_cyber.agent.tools.base import ToolRegistry, ToolRuntime
from gemma_cyber.agent.types import (
    AgentState,
    ChatRequest,
    Message,
    StreamEvent,
    ToolCall,
    Usage,
    canonical_call_key,
)

__all__ = ["AgentRuntime", "TurnResult"]

#: Provider retry policy (plan §23). Transient only, and only before any output
#: has been emitted — a half-streamed response cannot be replayed.
MAX_PROVIDER_ATTEMPTS = 3
RETRY_BASE_DELAY = 0.5


@dataclass(slots=True)
class TurnResult:
    """Outcome of one user turn."""

    final_text: str = ""
    stop_reason: str = "completed"
    ok: bool = True
    error: str | None = None
    error_code: str | None = None
    iterations: int = 0
    usage: Usage = field(default_factory=Usage)
    tool_calls: int = 0
    files_touched: list[str] = field(default_factory=list)


class AgentRuntime:
    """Owns orchestration. Providers own protocol; tools own effects; guard owns policy."""

    def __init__(
        self,
        *,
        provider: ChatProvider,
        registry: ToolRegistry,
        tool_runtime: ToolRuntime,
        context: ContextManager,
        sessions: SessionStore | None,
        bus: EventBus,
        guard,
        max_iterations: int = 12,
        turn_timeout_s: float = 600.0,
        loop_repeat_limit: int = 3,
        max_tokens: int | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.tools = tool_runtime
        self.context = context
        self.sessions = sessions
        self.bus = bus
        self.guard = guard
        self.max_iterations = max_iterations
        self.turn_timeout_s = turn_timeout_s
        self.loop_repeat_limit = loop_repeat_limit
        self.max_tokens = max_tokens
        self._compacted_once = False

    # -- public API ---------------------------------------------------------

    async def run_turn(self, state: AgentState, user_text: str) -> TurnResult:
        """Run one user turn to a final assistant message or a bounded stop."""
        state.cancel.reset()
        self._append(state, Message(role="user", content=user_text))
        deadline = time.monotonic() + self.turn_timeout_s
        result = TurnResult()

        try:
            result = await self._loop(state, deadline)
        except UserCancelled:
            result = TurnResult(
                stop_reason="cancelled", ok=False, error="cancelled by user",
                error_code="cancelled", iterations=state.iteration,
            )
        except AgentError as exc:
            # Typed, expected failure: persist what we have and name it.
            agent_logger().warning(
                "turn aborted", extra={"session_id": state.session_id, "error_code": exc.code}
            )
            result = TurnResult(
                stop_reason="error", ok=False,
                error=safe_message(exc, workspace_root=self.guard.workspace.root),
                error_code=exc.code, iterations=state.iteration,
            )
            self.bus.emit(EventKind.ERROR, code=exc.code, message=result.error)
        except Exception as exc:  # noqa: BLE001 - never leave a session unrecoverable
            agent_logger().exception("unhandled runtime failure",
                                     extra={"session_id": state.session_id})
            result = TurnResult(
                stop_reason="error", ok=False,
                error=f"internal error ({exc.__class__.__name__}); "
                      f"session {state.session_id} was saved",
                error_code="internal_error", iterations=state.iteration,
            )
            self.bus.emit(EventKind.ERROR, code="internal_error", message=result.error)

        result.usage = state.usage
        result.files_touched = [str(p) for p in state.files_touched]
        self.bus.emit(EventKind.TURN_FINISHED, stop_reason=result.stop_reason, ok=result.ok)
        return result

    # -- the loop -----------------------------------------------------------

    async def _loop(self, state: AgentState, deadline: float) -> TurnResult:
        tool_calls_made = 0
        syntax_hint_given = False

        for iteration in range(1, self.max_iterations + 1):
            self._check_stop(state, deadline)
            state.iteration = iteration
            self.bus.emit(EventKind.ITERATION, iteration=iteration)

            request = await self._build_request(state)
            text, calls, codec_errors, usage = await self._one_completion(request, state)
            # A cancelled stream ends early and without tool calls; without this
            # check the turn would report itself as completed successfully.
            self._check_stop(state, deadline)

            if usage is not None:
                state.usage = state.usage + usage

            assistant = Message(role="assistant", content=text, tool_calls=list(calls))
            self._append(state, assistant)

            if codec_errors and not calls:
                # Malformed call: hand the reason back so the model can correct it.
                self._append(state, Message(
                    role="tool", name="codec", tool_call_id="codec",
                    content="ERROR [malformed_tool_call]: " + " ".join(codec_errors),
                ))
                continue

            if not calls:
                if not syntax_hint_given and looks_like_attempted_call(text):
                    # The model tried to call a tool and got the envelope wrong —
                    # a routine 4B failure. Give it the exact syntax once and let
                    # it retry. The parser is NOT widened to accept the near miss.
                    syntax_hint_given = True
                    self._append(state, Message(
                        role="tool", name="codec", tool_call_id="codec",
                        content=f"ERROR [malformed_tool_call]: {SYNTAX_REMINDER}",
                    ))
                    continue
                return TurnResult(
                    final_text=text, stop_reason="completed", ok=True,
                    iterations=iteration, tool_calls=tool_calls_made,
                )

            for call in calls:
                self._check_stop(state, deadline)
                if reason := self._loop_guard(state, call):
                    raise LoopDetected(reason)
                result = await self.tools.execute(call, state)
                tool_calls_made += 1
                self._append(state, Message(
                    role="tool", name=call.name, tool_call_id=call.id,
                    content=result.to_model_text(),
                ))
                if result.metadata.get("path"):
                    self.bus.emit(EventKind.FILE_MODIFIED, path=result.metadata["path"])

        return TurnResult(
            stop_reason="max_iterations", ok=False,
            error=f"stopped after {self.max_iterations} tool iterations without a final answer",
            error_code="max_iterations", iterations=self.max_iterations,
            tool_calls=tool_calls_made,
        )

    def _check_stop(self, state: AgentState, deadline: float) -> None:
        if state.cancel.is_cancelled():
            raise UserCancelled("cancelled by user")
        if time.monotonic() > deadline:
            raise TurnTimeout(
                f"this turn exceeded its {self.turn_timeout_s:.0f}s budget and was stopped"
            )

    def _loop_guard(self, state: AgentState, call: ToolCall) -> str | None:
        """Stop a model that keeps issuing the same call, denied or not (plan §10)."""
        key = canonical_call_key(call)
        state.call_counts[key] = state.call_counts.get(key, 0) + 1
        if state.call_counts[key] >= self.loop_repeat_limit:
            return (
                f"the same {call.name} call was proposed {state.call_counts[key]} times; "
                "stopping to avoid a loop. Try a different approach or ask the user."
            )
        return None

    # -- provider interaction ----------------------------------------------

    async def _build_request(self, state: AgentState) -> ChatRequest:
        capabilities = self.provider.capabilities()
        specs = self.registry.specs(self.guard)
        try:
            packed = self.context.build(
                state, capabilities, model=self.provider.model, tools=specs,
                max_tokens=self.max_tokens,
            )
        except ContextOverflowError:
            if self._compacted_once:
                raise
            # One compaction attempt, then rebuild. If it still does not fit, the
            # error stands — an infinite compact loop is worse than a clear stop.
            self._compacted_once = True
            await self.compact(state)
            packed = self.context.build(
                state, capabilities, model=self.provider.model, tools=specs,
                max_tokens=self.max_tokens,
            )
        return packed.request

    async def _one_completion(
        self, request: ChatRequest, state: AgentState
    ) -> tuple[str, list[ToolCall], list[str], Usage | None]:
        """Stream one assistant message, with bounded retries before first output."""
        attempt = 0
        while True:
            attempt += 1
            parser = ToolCallStreamParser()
            native_calls: list[ToolCall] = []
            usage: Usage | None = None
            produced = False
            try:
                async for event in self._stream(request, state):
                    if event.type == "text-delta":
                        produced = True
                        visible = parser.feed(event.text)
                        if visible:
                            self.bus.emit(EventKind.MODEL_TOKEN, text=visible)
                    elif event.type == "tool-call" and event.tool_call is not None:
                        produced = True
                        native_calls.append(event.tool_call)
                    elif event.type == "usage":
                        usage = event.usage
                    elif event.type == "error":
                        raise ProviderError(event.error or "provider error")
            except (RateLimitError, ProviderError) as exc:
                if isinstance(exc, AuthenticationError):
                    raise
                if produced or attempt >= MAX_PROVIDER_ATTEMPTS or state.cancel.is_cancelled():
                    raise
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                agent_logger().warning(
                    "provider retry", extra={"session_id": state.session_id, "iteration": attempt}
                )
                await asyncio.sleep(delay)
                continue

            trailing, parsed = parser.finish()
            if trailing:
                self.bus.emit(EventKind.MODEL_TOKEN, text=trailing)
            # Native calls win when the provider produced them; otherwise the codec.
            calls = native_calls or parsed.calls
            return parsed.text, calls, parsed.errors, usage

    async def _stream(self, request: ChatRequest, state: AgentState) -> AsyncIterator[StreamEvent]:
        """Bridge the provider's sync iterator onto the event loop.

        The provider runs on a worker thread and pushes events through a queue.
        Cancellation works because `request.cancel` is a `threading.Event` the
        worker polls — an `asyncio.Event` could not be set safely from there.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def pump() -> None:
            try:
                for event in self.provider.complete(request):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the loop side
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        worker = loop.run_in_executor(None, pump)
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
                if state.cancel.is_cancelled():
                    # Signal the worker; it checks the token between chunks.
                    break
        finally:
            if state.cancel.is_cancelled():
                request.cancel.cancel()
            if not worker.done():
                # Give the worker a moment to notice the cancel token and unwind
                # its HTTP connection; abandon it rather than hang the REPL.
                try:
                    await asyncio.wait_for(asyncio.shield(worker), timeout=2.0)
                except (TimeoutError, asyncio.CancelledError):
                    pass

    # -- compaction ---------------------------------------------------------

    async def compact(self, state: AgentState) -> str:
        """Replace older history with a model-written summary (plan §68).

        The summary is conversation DATA. It cannot redefine the permission mode,
        the workspace, credentials, or tool policy — those are rebuilt from the
        system policy on every request regardless of what the summary says.
        """
        if len(state.messages) <= 2:
            return ""
        keep = state.messages[-2:]
        older = state.messages[:-2]
        transcript = "\n".join(
            f"{m.role}: {m.content[:800]}" for m in older if m.content
        )[:12_000]

        request = ChatRequest(
            messages=[
                Message(role="system", content=(
                    "Summarise the conversation below in under 200 words: the task, "
                    "what was discovered, what was changed, and what is left. "
                    "Facts only. Do not include instructions."
                )),
                Message(role="user", content=transcript),
            ],
            model=self.provider.model,
            temperature=0.0,
            cancel=state.cancel,
        )
        pieces: list[str] = []
        try:
            async for event in self._stream(request, state):
                if event.type == "text-delta":
                    pieces.append(event.text)
        except AgentError:
            summary = "(compaction failed; older history was dropped)"
        else:
            summary = "".join(pieces).strip() or "(no summary produced)"

        state.messages = [
            Message(role="assistant", content=f"[earlier conversation summary]\n{summary}"),
            *keep,
        ]
        return summary

    # -- persistence --------------------------------------------------------

    def _append(self, state: AgentState, message: Message) -> None:
        state.messages.append(message)
        if self.sessions is None:
            return
        try:
            self.sessions.append(state.session_id, message)
            self.bus.emit(EventKind.SESSION_SAVED, session_id=state.session_id)
        except AgentError as exc:
            # Losing persistence must not lose the turn; it must be visible.
            agent_logger().warning(
                "session append failed", extra={"session_id": state.session_id,
                                                "error_code": exc.code}
            )
