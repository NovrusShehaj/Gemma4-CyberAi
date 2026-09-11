"""Core domain model for the `gemma4` terminal agent (plan §7).

These types are the vocabulary every other agent module speaks. They are
deliberately separate from `gemma_cyber.clients.ollama_client.GenerationResult`
and `InferenceEngine`: that pair is the single-turn *generate* contract the
hosted API and the evaluation harness depend on, and overloading it with tool
calls would break scorecards (plan §9). Chat+tools is a different contract.

Dataclasses are used for transport/runtime state (cheap, no validation cost on
every streamed token). Pydantic is reserved for the places where validation *is*
a security boundary: tool arguments (`tools/base.py`) and config (`config.py`).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "AgentState",
    "CancelToken",
    "ChatRequest",
    "Message",
    "ModelCapabilities",
    "PermissionMode",
    "Role",
    "SideEffect",
    "StreamEvent",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "canonical_call_key",
    "new_id",
]


def new_id(prefix: str = "") -> str:
    """Short, collision-safe identifier used for sessions and tool calls."""
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}{raw}" if prefix else raw


# -- permission vocabulary --------------------------------------------------

class PermissionMode(StrEnum):
    """Process-wide capability ceiling. Set at startup; the model cannot change it.

    Ordering matters: :meth:`at_least` is used by the guard and by tool
    visibility, so the enum is paired with an explicit rank rather than relying
    on declaration order.
    """

    READ_ONLY = "read-only"
    WORKSPACE = "workspace"
    AGENT = "agent"
    TRUSTED = "trusted"

    @property
    def rank(self) -> int:
        return _MODE_RANK[self]

    def at_least(self, other: PermissionMode) -> bool:
        return self.rank >= other.rank


_MODE_RANK: dict[PermissionMode, int] = {
    PermissionMode.READ_ONLY: 0,
    PermissionMode.WORKSPACE: 1,
    PermissionMode.AGENT: 2,
    PermissionMode.TRUSTED: 3,
}


class SideEffect(StrEnum):
    """What a tool does to the world. The guard authorises on this, not on name."""

    READ = "read"
    WORKSPACE_WRITE = "workspace_write"
    PROCESS = "process"
    NETWORK = "network"
    SECURITY_SENSITIVE = "security_sensitive"


# -- cancellation -----------------------------------------------------------

class CancelToken:
    """Thread-safe cancellation signal shared by the runtime, providers, and tools.

    Deviation from the plan's ``asyncio.Event`` (§10) with cause: providers stream
    on a worker thread and tools run in ``asyncio.to_thread``, and
    ``asyncio.Event`` is not thread-safe — setting it from a tool thread is a
    data race. ``threading.Event`` is checkable from both worlds and is what
    ``os.killpg`` paths in `shell.exec` need.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


# -- conversation -----------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class Message:
    """One conversation entry. Persisted verbatim as a JSONL line (plan §18)."""

    role: Role
    content: str
    #: Set on ``role="tool"`` messages so a result can be matched to its call.
    tool_call_id: str | None = None
    #: Set on ``role="tool"`` messages: which tool produced this content.
    name: str | None = None
    #: Tool calls the assistant emitted with this message (native or codec).
    tool_calls: list[ToolCall] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content, "ts": self.ts}
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.tool_calls:
            data["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Rebuild from a persisted record, ignoring unknown keys (plan §18)."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", "") or "",
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            tool_calls=[ToolCall.from_dict(c) for c in data.get("tool_calls", []) or []],
            ts=float(data.get("ts", 0.0) or 0.0),
        )


# -- tools ------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ToolSpec:
    """The provider-facing description of a tool: name, docs, JSON schema.

    ``side_effect`` travels with the spec so that any future tool source (entry
    points, MCP — plan §21) still carries the metadata the guard authorises on.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    side_effect: SideEffect

    def to_openai_schema(self) -> dict[str, Any]:
        """Native tool-calling wire format (Ollama and OpenAI share this shape)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolCall:
    """A model-proposed invocation. Untrusted until validated *and* authorised."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("call_"))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        args = data.get("arguments")
        return cls(
            name=str(data.get("name", "")),
            arguments=args if isinstance(args, dict) else {},
            id=str(data.get("id") or new_id("call_")),
        )


@dataclass(slots=True)
class ToolResult:
    """The sanitised, bounded outcome handed back to the model.

    ``content`` has already been truncated, ANSI-stripped and redacted by
    `ToolRuntime` before it reaches this object — nothing downstream re-derives
    those guarantees.
    """

    call_id: str
    name: str
    ok: bool
    content: str = ""
    error: str | None = None
    error_code: str | None = None
    truncated: bool = False
    #: SHA-256 of the omitted remainder, so truncation is auditable (plan §11.6).
    remainder_sha256: str | None = None
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_model_text(self) -> str:
        """Render for the conversation. Wrapping as untrusted is `context.py`'s job."""
        if not self.ok:
            return f"ERROR [{self.error_code or 'tool_error'}]: {self.error or 'tool failed'}"
        body = self.content
        if self.truncated:
            digest = self.remainder_sha256 or "unknown"
            body += (
                f"\n\n[output truncated; sha256 of omitted remainder: {digest[:16]}]"
            )
        return body


def canonical_call_key(call: ToolCall) -> str:
    """Stable hash of ``tool + canonical args`` used by the loop detector (§10)."""
    try:
        payload = json.dumps(call.arguments, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        payload = repr(sorted(call.arguments.items()))
    return hashlib.sha256(f"{call.name}\x00{payload}".encode()).hexdigest()


# -- provider contract ------------------------------------------------------

@dataclass(slots=True)
class Usage:
    """Token accounting when the provider reports it. Omitted from UI when zero."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
        }


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """What a provider/model actually supports — reported, never assumed (§9).

    ``tools_native`` defaults to False: the runtime falls back to `XmlToolCodec`
    unless a provider positively asserts reliable native tool calling. Claiming
    support we have not contract-tested is how a 4B model silently no-ops.
    """

    streaming: bool = True
    tools_native: bool = False
    tools_emulated: bool = True
    json_schema: bool = False
    vision: bool = False
    reasoning: bool = False
    #: Context budget in tokens; None = unknown (ContextManager uses a floor).
    context_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "streaming": self.streaming,
            "tools_native": self.tools_native,
            "tools_emulated": self.tools_emulated,
            "json_schema": self.json_schema,
            "vision": self.vision,
            "reasoning": self.reasoning,
            "context_tokens": self.context_tokens,
        }


@dataclass(slots=True)
class ChatRequest:
    """One provider call. ``tools`` is empty when the codec path is in use."""

    messages: list[Message]
    model: str
    tools: list[ToolSpec] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int | None = None
    seed: int | None = None
    cancel: CancelToken = field(default_factory=CancelToken)
    #: Sequences that end generation early (the codec closes a call tag).
    stop: list[str] = field(default_factory=list)


StreamEventType = Literal["text-delta", "tool-call", "usage", "error", "done"]


@dataclass(slots=True)
class StreamEvent:
    """One item of a provider stream. Exactly one payload field is meaningful."""

    type: StreamEventType
    text: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    error: str = ""
    #: Populated on "error" so the runtime's retry table can discriminate.
    error_code: str | None = None

    @classmethod
    def delta(cls, text: str) -> StreamEvent:
        return cls(type="text-delta", text=text)

    @classmethod
    def call(cls, tool_call: ToolCall) -> StreamEvent:
        return cls(type="tool-call", tool_call=tool_call)

    @classmethod
    def done(cls) -> StreamEvent:
        return cls(type="done")


# -- agent state ------------------------------------------------------------

@dataclass(slots=True)
class AgentState:
    """Mutable state of one agent session (plan §10)."""

    session_id: str
    mode: PermissionMode
    messages: list[Message] = field(default_factory=list)
    iteration: int = 0
    pending_calls: list[ToolCall] = field(default_factory=list)
    files_touched: list[Path] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cancel: CancelToken = field(default_factory=CancelToken)
    #: SHA-256 of file contents as last read, keyed by workspace-relative path.
    #: `fs.edit` requires the model to echo one of these back (plan §13).
    file_hashes: dict[str, str] = field(default_factory=dict)
    #: Paths already approved for writing in this session (plan §15).
    approved_writes: set[str] = field(default_factory=set)
    #: Tool-call keys already seen, for the loop detector.
    call_counts: dict[str, int] = field(default_factory=dict)

    def record_file_hash(self, rel_path: str, digest: str) -> None:
        self.file_hashes[rel_path] = digest
