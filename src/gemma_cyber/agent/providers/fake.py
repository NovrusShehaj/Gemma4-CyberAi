"""`FakeProvider` — a first-class test adapter, not throwaway scaffolding.

CI must never need a live model (plan §24), and every runtime behaviour worth
asserting — a tool round-trip, a loop, a cancellation, a provider error, a
retry — is easier to pin down deterministically than to provoke from a 4B model.

Scripted turns are consumed in order. A turn's text may contain `<tool_call>`
envelopes (exercising the real codec path) or carry native calls directly, so
the same script can drive both tool protocols.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from gemma_cyber.agent.types import (
    ChatRequest,
    ModelCapabilities,
    StreamEvent,
    ToolCall,
    Usage,
)

__all__ = ["FakeProvider", "ScriptedTurn"]


@dataclass(slots=True)
class ScriptedTurn:
    """One provider response.

    ``text`` is streamed in ``chunk_size`` pieces. ``native_calls`` emits
    ``tool-call`` events directly (native protocol); putting `<tool_call>` blocks
    in ``text`` instead exercises `XmlToolCodec`.
    """

    text: str = ""
    native_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    error: str | None = None
    error_code: str | None = None
    #: Total seconds to spread the stream over — used by cancellation tests.
    delay_s: float = 0.0
    chunk_size: int = 8
    #: Raise instead of yielding an error event, for abort-path tests.
    raises: Exception | None = None


class FakeProvider:
    """Deterministic `ChatProvider` driven by a list of scripted turns."""

    def __init__(
        self,
        turns: list[ScriptedTurn] | list[str] | None = None,
        *,
        name: str = "fake",
        model: str = "fake-model",
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._turns: list[ScriptedTurn] = [
            ScriptedTurn(text=t) if isinstance(t, str) else t for t in (turns or [])
        ]
        self._capabilities = capabilities or ModelCapabilities(
            streaming=True, tools_native=False, tools_emulated=True, context_tokens=8192
        )
        #: Every request the runtime made, for assertions about context packing.
        self.requests: list[ChatRequest] = []
        self.closed = False
        self._index = 0

    # -- ChatProvider -------------------------------------------------------

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def complete(self, request: ChatRequest) -> Iterator[StreamEvent]:
        self.requests.append(request)
        turn = self._next_turn()

        if turn.raises is not None:
            raise turn.raises

        if turn.error is not None:
            yield StreamEvent(type="error", error=turn.error, error_code=turn.error_code)
            return

        chunks = self._chunks(turn)
        per_chunk = (turn.delay_s / len(chunks)) if (turn.delay_s and chunks) else 0.0
        for chunk in chunks:
            if request.cancel.is_cancelled():
                return
            if per_chunk:
                # Sleep in slices so a cancel lands promptly rather than after
                # the whole delay — the behaviour Ctrl+C tests depend on.
                self._interruptible_sleep(per_chunk, request)
                if request.cancel.is_cancelled():
                    return
            yield StreamEvent.delta(chunk)

        for call in turn.native_calls:
            if request.cancel.is_cancelled():
                return
            yield StreamEvent.call(call)

        if turn.usage is not None:
            yield StreamEvent(type="usage", usage=turn.usage)
        yield StreamEvent.done()

    def close(self) -> None:
        self.closed = True

    # -- helpers ------------------------------------------------------------

    def _next_turn(self) -> ScriptedTurn:
        if self._index < len(self._turns):
            turn = self._turns[self._index]
            self._index += 1
            return turn
        # Running off the end means the runtime looped further than the script
        # expected. Return a terminal answer rather than hanging the test.
        return ScriptedTurn(text="(fake provider: no further scripted turns)")

    @staticmethod
    def _chunks(turn: ScriptedTurn) -> list[str]:
        if not turn.text:
            return []
        size = max(1, turn.chunk_size)
        return [turn.text[i:i + size] for i in range(0, len(turn.text), size)]

    @staticmethod
    def _interruptible_sleep(seconds: float, request: ChatRequest) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if request.cancel.wait(min(0.01, seconds)):
                return

    @property
    def turns_consumed(self) -> int:
        return self._index
