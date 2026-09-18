"""A minimal in-process event bus (plan §27).

Its whole job is to keep the runtime from importing the UI. The runtime emits;
the REPL, the JSON writer, or nothing at all subscribes. Synchronous, not
persisted, no queue, no broker — an enterprise bus here would be architecture
for its own sake.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AgentEvent", "EventBus", "EventKind"]


class EventKind:
    """Event names. A class of constants keeps typos out of subscriber code."""

    MODEL_TOKEN = "model_token_received"
    TOOL_APPROVAL_REQUESTED = "tool_approval_requested"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    FILE_MODIFIED = "file_modified"
    ERROR = "error_occurred"
    SESSION_SAVED = "session_saved"
    ITERATION = "iteration_started"
    TURN_FINISHED = "turn_finished"


@dataclass(slots=True)
class AgentEvent:
    kind: str
    fields: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)


Handler = Callable[[AgentEvent], None]


class EventBus:
    """Synchronous fan-out. A failing subscriber never breaks the runtime."""

    def __init__(self) -> None:
        self._handlers: list[Handler] = []
        #: Captured events, for tests that assert on runtime behaviour.
        self.history: list[AgentEvent] = []
        self.record_history = False

    def subscribe(self, handler: Handler) -> Callable[[], None]:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    def emit(self, kind: str, **fields: Any) -> None:
        event = AgentEvent(kind=kind, fields=fields)
        if self.record_history:
            self.history.append(event)
        for handler in list(self._handlers):
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - a renderer bug must not abort a turn
                continue

    def kinds(self) -> list[str]:
        return [e.kind for e in self.history]
