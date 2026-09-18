"""Provider contract + Ollama-specific quirks. No live daemon is required.

The contract block at the bottom runs against every adapter: a healthy stream
emits text and/or tool events and ends with exactly one `done`. That is the
shared shape the runtime is allowed to rely on.
"""

from __future__ import annotations

import json

import httpx
import pytest

from gemma_cyber.agent.errors import AuthenticationError, ProviderError, RateLimitError
from gemma_cyber.agent.providers.fake import FakeProvider, ScriptedTurn
from gemma_cyber.agent.providers.ollama_chat import OllamaChatProvider
from gemma_cyber.agent.types import (
    CancelToken,
    ChatRequest,
    Message,
    SideEffect,
    ToolCall,
    ToolSpec,
)


def _ndjson(*frames: dict) -> bytes:
    return b"".join(json.dumps(f).encode() + b"\n" for f in frames)


def _provider(handler, **kwargs) -> OllamaChatProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaChatProvider(model="gemma3:4b", client=client, **kwargs)


def _request(**kwargs) -> ChatRequest:
    kwargs.setdefault("messages", [Message(role="user", content="hi")])
    kwargs.setdefault("model", "gemma3:4b")
    return ChatRequest(**kwargs)


# -- happy path -------------------------------------------------------------

def test_stream_emits_text_then_done() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        return httpx.Response(200, content=_ndjson(
            {"message": {"role": "assistant", "content": "Hel"}, "done": False},
            {"message": {"role": "assistant", "content": "lo"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True,
             "prompt_eval_count": 11, "eval_count": 4},
        ))

    events = list(_provider(handler).complete(_request()))
    assert [e.type for e in events] == ["text-delta", "text-delta", "usage", "done"]
    assert "".join(e.text for e in events) == "Hello"
    assert events[2].usage is not None and events[2].usage.input_tokens == 11


def test_trailing_slash_in_host_is_normalised() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=_ndjson({"done": True}))

    provider = _provider(handler)
    provider.base_url = "http://127.0.0.1:11434/"
    OllamaChatProvider(model="m", base_url="http://127.0.0.1:11434/")  # constructor path
    provider.base_url = provider.base_url.rstrip("/")
    list(provider.complete(_request()))
    assert seen == ["http://127.0.0.1:11434/api/chat"]


def test_constructor_strips_trailing_slashes() -> None:
    assert OllamaChatProvider(model="m", base_url="http://h:1/").base_url == "http://h:1"


# -- quirks -----------------------------------------------------------------

def test_thinking_field_is_not_streamed_as_answer_text() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_ndjson(
            {"message": {"role": "assistant", "thinking": "hidden reasoning",
                         "content": ""}, "done": False},
            {"message": {"role": "assistant", "content": "answer"}, "done": True},
        ))

    events = list(_provider(handler).complete(_request()))
    text = "".join(e.text for e in events)
    assert text == "answer"
    assert "hidden" not in text


def test_blank_and_malformed_frames_are_tolerated() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        payload = (b"\n" + b"not json\n" + b"[1,2]\n"
                   + _ndjson({"message": {"content": "ok"}, "done": True}))
        return httpx.Response(200, content=payload)

    events = list(_provider(handler).complete(_request()))
    assert [e.type for e in events] == ["text-delta", "done"]


def test_a_flood_of_malformed_frames_fails_closed() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"garbage\n" * 50)

    with pytest.raises(ProviderError, match="unparseable"):
        list(_provider(handler).complete(_request()))


def test_stream_without_done_reports_truncation() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_ndjson({"message": {"content": "part"}}))

    events = list(_provider(handler).complete(_request()))
    assert [e.type for e in events] == ["text-delta", "error", "done"]
    assert "completion marker" in events[1].error


def test_error_frame_becomes_an_error_event() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_ndjson({"error": "model is loading"}))

    events = list(_provider(handler).complete(_request()))
    assert events[0].type == "error" and "loading" in events[0].error


# -- native tool calls ------------------------------------------------------

def test_native_tool_calls_are_decoded_when_enabled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "tools" in json.loads(request.content)
        return httpx.Response(200, content=_ndjson(
            {"message": {"content": "", "tool_calls": [
                {"function": {"name": "fs.read", "arguments": {"path": "a.py"}}},
                {"function": {"name": "bad"}},               # no arguments -> skipped
                {"function": {"name": "", "arguments": {}}},  # empty name -> skipped
                {"nope": 1},                                  # malformed -> skipped
            ]}, "done": True},
        ))

    spec = ToolSpec(name="fs.read", description="d", parameters={"type": "object"},
                    side_effect=SideEffect.READ)
    provider = _provider(handler, tools_native=True)
    events = list(provider.complete(_request(tools=[spec])))
    calls = [e.tool_call for e in events if e.type == "tool-call" and e.tool_call]
    assert len(calls) == 1 and calls[0].name == "fs.read"


def test_tool_schemas_are_withheld_when_native_tools_are_off() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "tools" not in json.loads(request.content)
        return httpx.Response(200, content=_ndjson({"done": True}))

    spec = ToolSpec(name="fs.read", description="d", parameters={"type": "object"},
                    side_effect=SideEffect.READ)
    list(_provider(handler).complete(_request(tools=[spec])))


def test_string_encoded_native_arguments_are_parsed() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_ndjson(
            {"message": {"content": "", "tool_calls": [
                {"function": {"name": "fs.read", "arguments": '{"path": "a"}'}}]},
             "done": True},
        ))

    events = list(_provider(handler, tools_native=True).complete(_request()))
    calls = [e.tool_call for e in events if e.type == "tool-call" and e.tool_call]
    assert calls[0].arguments == {"path": "a"}


# -- transport errors -------------------------------------------------------

@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, RateLimitError), (401, AuthenticationError), (403, AuthenticationError),
     (404, ProviderError), (500, ProviderError), (503, ProviderError)],
)
def test_status_codes_map_to_typed_errors(status: int, expected: type[Exception]) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="upstream said no")

    with pytest.raises(expected):
        list(_provider(handler).complete(_request()))


def test_connection_refused_is_a_provider_error_with_a_usable_hint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderError, match="ollama serve"):
        list(_provider(handler).complete(_request()))


def test_read_timeout_is_a_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError, match="stopped responding"):
        list(_provider(handler).complete(_request()))


# -- cancellation -----------------------------------------------------------

def test_cancel_stops_the_stream_early() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_ndjson(
            *[{"message": {"content": f"{i}"}, "done": False} for i in range(50)],
            {"done": True},
        ))

    cancel = CancelToken()
    stream = _provider(handler).complete(_request(cancel=cancel))
    first = next(stream)
    assert first.type == "text-delta"
    cancel.cancel()
    assert list(stream) == []


# -- model listing ----------------------------------------------------------

def test_list_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "gemma3:4b"}, {"x": 1}]})

    assert _provider(handler).list_models() == ["gemma3:4b"]


def test_list_models_unreachable_is_a_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=request)

    with pytest.raises(ProviderError):
        _provider(handler).list_models()


# -- shared contract --------------------------------------------------------

def _ollama_contract_provider() -> OllamaChatProvider:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_ndjson(
            {"message": {"content": "hello"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ))

    return _provider(handler)


@pytest.mark.parametrize(
    "make_provider",
    [lambda: FakeProvider([ScriptedTurn(text="hello")]), _ollama_contract_provider],
    ids=["fake", "ollama-mock"],
)
def test_contract_text_then_exactly_one_done(make_provider) -> None:
    provider = make_provider()
    events = list(provider.complete(_request()))
    assert events[-1].type == "done"
    assert sum(1 for e in events if e.type == "done") == 1
    assert "".join(e.text for e in events if e.type == "text-delta") == "hello"
    assert provider.capabilities().streaming is True
    provider.close()


@pytest.mark.parametrize(
    "make_provider",
    [lambda: FakeProvider([ScriptedTurn(native_calls=[ToolCall(name="fs.read",
                                                              arguments={"path": "a"})])]),
     lambda: _provider(
         lambda _: httpx.Response(200, content=_ndjson(
             {"message": {"content": "", "tool_calls": [
                 {"function": {"name": "fs.read", "arguments": {"path": "a"}}}]},
              "done": True})),
         tools_native=True)],
    ids=["fake", "ollama-mock"],
)
def test_contract_tool_call_then_done(make_provider) -> None:
    events = list(make_provider().complete(_request()))
    calls = [e.tool_call for e in events if e.type == "tool-call" and e.tool_call]
    assert len(calls) == 1 and calls[0].name == "fs.read"
    assert events[-1].type == "done"


# -- the tool-role quirk ----------------------------------------------------

def test_tool_results_are_delivered_as_user_messages_on_the_emulated_path() -> None:
    """Measured quirk: gemma3's template has no `tool` role.

    A ``role="tool"`` message is dropped by the daemon and the model then
    fabricates the file contents it never received. On the codec path the
    adapter therefore relabels it; the `<untrusted>` wrapper is unaffected.
    """
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=_ndjson({"done": True}))

    messages = [
        Message(role="user", content="read it"),
        Message(role="assistant", content="ok"),
        Message(role="tool", name="fs.read",
                content='<untrusted source="tool:fs.read">print(1)</untrusted>'),
    ]
    list(_provider(handler).complete(_request(messages=messages)))
    wire = captured[0]["messages"]
    assert wire[2]["role"] == "user"
    assert "Result of tool `fs.read`" in wire[2]["content"]
    assert "<untrusted" in wire[2]["content"]


def test_native_path_keeps_the_tool_role() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=_ndjson({"done": True}))

    messages = [Message(role="tool", name="fs.read", content="print(1)")]
    list(_provider(handler, tools_native=True).complete(_request(messages=messages)))
    assert captured[0]["messages"][0]["role"] == "tool"
