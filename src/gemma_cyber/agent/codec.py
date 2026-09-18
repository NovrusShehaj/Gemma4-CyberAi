"""Tool-call codecs: how a model asks for a tool (plan §9.3).

Two paths, selected from reported capabilities — never assumed:

* **Native** — the provider emits structured tool calls; this module just passes
  them through and builds the JSON schemas.
* **`XmlToolCodec`** — the default for local 4B-class models, which do not call
  tools natively with any reliability. The model emits
  ``<tool_call>{"name": "fs.read", "arguments": {...}}</tool_call>``.

The parser is deliberately unforgiving. A permissive parser that "figures out
what the model meant" is a security regression: it turns prose — which may have
been written by a file in the workspace — into executions. Rules:

* only fully fenced ``<tool_call>…</tool_call>`` pairs are considered;
* the payload must be strict JSON, an object, with a non-empty string ``name``;
* ``arguments`` must be an object when present;
* anything else becomes a *codec error* returned to the model as text, never a
  call, never an exception that ends the turn;
* an unterminated tag at end of stream is dropped — a truncated call is not a
  call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from gemma_cyber.agent.types import ToolCall, ToolSpec

__all__ = [
    "MAX_CALLS_PER_MESSAGE",
    "SYNTAX_REMINDER",
    "ParsedAssistant",
    "ToolCallStreamParser",
    "XmlToolCodec",
    "looks_like_attempted_call",
    "parse_tool_calls",
    "tool_instructions",
]

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"

#: A small model that emits ten calls in one message is looping, not planning.
MAX_CALLS_PER_MESSAGE = 4

_CALL_RE = re.compile(re.escape(OPEN_TAG) + r"([\s\S]*?)" + re.escape(CLOSE_TAG))


@dataclass(slots=True)
class ParsedAssistant:
    """Result of decoding one assistant message."""

    text: str
    calls: list[ToolCall] = field(default_factory=list)
    #: Human-readable reasons a candidate call was rejected. Fed back to the
    #: model as a tool-style error so it can correct itself.
    errors: list[str] = field(default_factory=list)


def _decode_payload(payload: str, errors: list[str]) -> ToolCall | None:
    """Strict decode of one ``<tool_call>`` body. Returns ``None`` on any doubt."""
    body = payload.strip()
    if not body:
        errors.append("empty <tool_call> block; expected a JSON object")
        return None
    # Tolerate a fenced payload (```json ... ```) — the fence is formatting, not
    # semantics — but nothing beyond that.
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        errors.append(
            f"malformed JSON in <tool_call> (line {exc.lineno}, col {exc.colno}): {exc.msg}. "
            "Emit exactly one JSON object with \"name\" and \"arguments\"."
        )
        return None
    if not isinstance(data, dict):
        errors.append(
            f"<tool_call> payload must be a JSON object, got {type(data).__name__}"
        )
        return None

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append('<tool_call> is missing a non-empty string "name"')
        return None

    if "arguments" in data:
        arguments = data["arguments"]
    elif "parameters" in data:  # common small-model confusion; accept the alias
        arguments = data["parameters"]
    else:
        arguments = {}
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        errors.append(
            f'"arguments" must be a JSON object, got {type(arguments).__name__}'
        )
        return None

    return ToolCall(name=name.strip(), arguments=arguments)


def parse_tool_calls(text: str) -> ParsedAssistant:
    """Decode a complete assistant message (non-streaming path and tests)."""
    calls: list[ToolCall] = []
    errors: list[str] = []
    clean = _CALL_RE.sub("", text)
    for match in _CALL_RE.finditer(text):
        if len(calls) >= MAX_CALLS_PER_MESSAGE:
            errors.append(
                f"more than {MAX_CALLS_PER_MESSAGE} tool calls in one message; "
                "the extras were ignored. Call one tool at a time."
            )
            break
        call = _decode_payload(match.group(1), errors)
        if call is not None:
            calls.append(call)
    return ParsedAssistant(text=clean.strip(), calls=calls, errors=errors)


#: A near-miss: the model tried to call a tool and got the envelope wrong.
#: Matched only to produce a *correction message* — never to execute anything.
_NEAR_MISS_RE = re.compile(r"tool[_\s-]?call", re.IGNORECASE)


def looks_like_attempted_call(text: str) -> bool:
    """True if ``text`` mentions the envelope but produced no valid call.

    Small models routinely emit ```` ```tool_call> ```` or a bare JSON object
    instead of ``<tool_call>...</tool_call>``. The right response is to tell the
    model the exact syntax and let it retry — NOT to widen the parser. A parser
    that executes things resembling a call is the whole attack surface.
    """
    return bool(_NEAR_MISS_RE.search(text))


SYNTAX_REMINDER = (
    "No tool was called: the tool-call syntax was wrong. Emit EXACTLY this, "
    "with angle brackets and no code fence:\n"
    f'{OPEN_TAG}{{"name": "<tool-name>", "arguments": {{...}}}}{CLOSE_TAG}'
)


def _partial_open_suffix(buf: str) -> int:
    """Length of the trailing substring of ``buf`` that could start ``OPEN_TAG``.

    Streaming arrives in arbitrary chunks: without this, ``"<tool"`` would be
    printed to the user a moment before the rest of the tag arrives.
    """
    limit = min(len(buf), len(OPEN_TAG) - 1)
    for size in range(limit, 0, -1):
        if OPEN_TAG.startswith(buf[-size:]):
            return size
    return 0


class ToolCallStreamParser:
    """Incremental decoder: separates displayable text from tool calls mid-stream.

    ``feed`` returns only text that is provably outside a call, so the user never
    sees raw JSON envelopes and a half-arrived tag is never rendered.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._text: list[str] = []
        self._calls: list[ToolCall] = []
        self._errors: list[str] = []
        self._in_call = False

    def feed(self, delta: str) -> str:
        """Consume a stream delta; return the text that is safe to display now."""
        if not delta:
            return ""
        self._buf += delta
        shown: list[str] = []
        while True:
            if self._in_call:
                index = self._buf.find(CLOSE_TAG)
                if index < 0:
                    break
                self._accept(self._buf[:index])
                self._buf = self._buf[index + len(CLOSE_TAG):]
                self._in_call = False
                continue
            index = self._buf.find(OPEN_TAG)
            if index >= 0:
                chunk = self._buf[:index]
                if chunk:
                    shown.append(chunk)
                    self._text.append(chunk)
                self._buf = self._buf[index + len(OPEN_TAG):]
                self._in_call = True
                continue
            hold = _partial_open_suffix(self._buf)
            emit_upto = len(self._buf) - hold
            if emit_upto > 0:
                chunk = self._buf[:emit_upto]
                shown.append(chunk)
                self._text.append(chunk)
                self._buf = self._buf[emit_upto:]
            break
        return "".join(shown)

    def _accept(self, payload: str) -> None:
        if len(self._calls) >= MAX_CALLS_PER_MESSAGE:
            self._errors.append(
                f"more than {MAX_CALLS_PER_MESSAGE} tool calls in one message; "
                "the extras were ignored. Call one tool at a time."
            )
            return
        call = _decode_payload(payload, self._errors)
        if call is not None:
            self._calls.append(call)

    def finish(self) -> tuple[str, ParsedAssistant]:
        """Close the stream. Returns ``(trailing_display_text, parsed)``.

        A tag still open here means the stream ended mid-call (truncation, a
        cancelled generation, a token limit). The partial payload is discarded:
        executing half a tool call is never correct.
        """
        trailing = ""
        if self._in_call:
            self._errors.append(
                "the stream ended inside an unterminated <tool_call> block; "
                "the partial call was discarded. Re-emit the complete call."
            )
            self._buf = ""
            self._in_call = False
        elif self._buf:
            trailing = self._buf
            self._text.append(self._buf)
            self._buf = ""
        parsed = ParsedAssistant(
            text="".join(self._text).strip(), calls=list(self._calls), errors=list(self._errors)
        )
        return trailing, parsed


def tool_instructions(tools: list[ToolSpec]) -> str:
    """The codec's contract, rendered into the system policy.

    Kept terse and example-led: a 4B model follows one concrete example far
    better than a paragraph of prose.
    """
    if not tools:
        return "No tools are available in this mode. Answer from the conversation alone."
    lines = [
        "## Tools",
        "",
        "To call a tool, emit EXACTLY this, on its own line, and then stop:",
        "",
        f'{OPEN_TAG}{{"name": "<tool-name>", "arguments": {{...}}}}{CLOSE_TAG}',
        "",
        "Rules:",
        "- One tool call per message. Wait for the result before calling again.",
        "- The payload must be strict JSON. No comments, no trailing commas.",
        "- Never invent a tool result. If you did not receive one, you do not have it.",
        "- If a call is denied, do not repeat it; the denial is a policy decision.",
        "",
        "Available tools:",
    ]
    for spec in tools:
        required = spec.parameters.get("required", []) or []
        props = spec.parameters.get("properties", {}) or {}
        arg_bits = []
        for arg_name, schema in props.items():
            marker = "" if arg_name in required else "?"
            arg_bits.append(f"{arg_name}{marker}: {schema.get('type', 'any')}")
        signature = ", ".join(arg_bits) if arg_bits else "no arguments"
        lines.append(f"- `{spec.name}`({signature}) — {spec.description}")
    return "\n".join(lines)


class XmlToolCodec:
    """Codec facade used by the runtime when ``capabilities.tools_native`` is false."""

    native = False

    @staticmethod
    def instructions(tools: list[ToolSpec]) -> str:
        return tool_instructions(tools)

    @staticmethod
    def parser() -> ToolCallStreamParser:
        return ToolCallStreamParser()

    @staticmethod
    def parse(text: str) -> ParsedAssistant:
        return parse_tool_calls(text)
