"""Runtime loop behaviour, driven entirely by FakeProvider. No live model."""

from __future__ import annotations

import asyncio
import threading

import pytest

from gemma_cyber.agent.errors import ProviderError, RateLimitError
from gemma_cyber.agent.events import EventKind
from gemma_cyber.agent.providers.fake import FakeProvider, ScriptedTurn
from gemma_cyber.agent.types import ModelCapabilities, PermissionMode, ToolCall, Usage


def _call(name: str, **arguments: object) -> str:
    import json
    return f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'


# -- the vertical slice -----------------------------------------------------

async def test_scripted_read_then_answer(make_agent) -> None:
    provider = FakeProvider([
        ScriptedTurn(text="Let me look. " + _call("fs.read", path="src/app.py")),
        ScriptedTurn(text="It returns 42."),
    ])
    agent = make_agent(provider)
    result = await agent.ask("What does app.py return?")

    assert result.ok and result.stop_reason == "completed"
    assert result.final_text == "It returns 42."
    assert result.tool_calls == 1

    roles = [m.role for m in agent.state.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_message = agent.state.messages[2]
    assert "def main" in tool_message.content
    assert tool_message.name == "fs.read"

    # The second request carries the tool result back to the model.
    assert len(provider.requests) == 2
    assert any("def main" in m.content for m in provider.requests[1].messages)


async def test_native_tool_calls_drive_the_same_loop(make_agent) -> None:
    provider = FakeProvider(
        [ScriptedTurn(native_calls=[ToolCall(name="fs.read", arguments={"path": "README.md"})]),
         ScriptedTurn(text="It is a test project.")],
        capabilities=ModelCapabilities(tools_native=True, context_tokens=8192),
    )
    agent = make_agent(provider)
    result = await agent.ask("Summarise the readme")
    assert result.ok and result.tool_calls == 1
    assert "# Project" in agent.state.messages[2].content


async def test_plain_answer_makes_no_tool_calls(make_agent) -> None:
    agent = make_agent(FakeProvider([ScriptedTurn(text="Hello.")]))
    result = await agent.ask("hi")
    assert result.final_text == "Hello." and result.tool_calls == 0


# -- persistence ------------------------------------------------------------

async def test_every_message_is_persisted_as_it_happens(make_agent) -> None:
    provider = FakeProvider([
        ScriptedTurn(text=_call("fs.read", path="README.md")),
        ScriptedTurn(text="done"),
    ])
    agent = make_agent(provider)
    await agent.ask("read it")
    reloaded = agent.sessions.load(agent.session.meta.id)
    assert [m.role for m in reloaded.messages] == ["user", "assistant", "tool", "assistant"]


# -- stop conditions --------------------------------------------------------

async def test_max_iterations_stops_the_loop(make_agent) -> None:
    # A model that calls a different path forever: never finishes, never repeats.
    provider = FakeProvider([
        ScriptedTurn(text=_call("fs.glob", pattern=f"*.{i}")) for i in range(20)
    ])
    agent = make_agent(provider, max_iterations=4)
    result = await agent.ask("loop forever")
    assert not result.ok
    assert result.stop_reason == "max_iterations" and result.iterations == 4


async def test_identical_repeated_calls_trip_the_loop_detector(make_agent) -> None:
    provider = FakeProvider([ScriptedTurn(text=_call("fs.read", path="README.md"))] * 10)
    agent = make_agent(provider)
    result = await agent.ask("read it over and over")
    assert not result.ok
    assert result.error_code == "loop_detected"
    assert "loop" in (result.error or "")


async def test_repeated_denied_calls_do_not_spin_forever(make_agent) -> None:
    # read-only mode: every fs.write is denied, and the model keeps asking.
    provider = FakeProvider(
        [ScriptedTurn(text=_call("fs.write", path="new.txt", content="x"))] * 10
    )
    agent = make_agent(provider, mode=PermissionMode.READ_ONLY)
    result = await agent.ask("write a file")
    assert not result.ok and result.error_code == "loop_detected"
    denials = [m for m in agent.state.messages if m.role == "tool" and "ERROR" in m.content]
    assert denials and "not available in read-only mode" in denials[0].content


async def test_turn_timeout_is_enforced(make_agent) -> None:
    provider = FakeProvider([ScriptedTurn(text=_call("fs.glob", pattern=f"*.{i}"))
                             for i in range(10)])
    agent = make_agent(provider)
    agent.runtime.turn_timeout_s = -1.0  # already expired
    result = await agent.ask("anything")
    assert result.error_code == "turn_timeout"


# -- malformed calls --------------------------------------------------------

async def test_malformed_tool_call_is_returned_to_the_model(make_agent) -> None:
    provider = FakeProvider([
        ScriptedTurn(text='<tool_call>{"name": "fs.read", "arguments": {</tool_call>'),
        ScriptedTurn(text="Sorry, corrected."),
    ])
    agent = make_agent(provider)
    result = await agent.ask("read something")
    assert result.ok
    codec_errors = [m for m in agent.state.messages if m.name == "codec"]
    assert codec_errors and "malformed JSON" in codec_errors[0].content


async def test_unknown_tool_is_a_recoverable_error(make_agent) -> None:
    provider = FakeProvider([
        ScriptedTurn(text=_call("fs.destroy", path="x")),
        ScriptedTurn(text="Understood."),
    ])
    agent = make_agent(provider)
    result = await agent.ask("destroy it")
    assert result.ok
    assert "no tool named" in agent.state.messages[2].content


async def test_invalid_arguments_are_a_recoverable_error(make_agent) -> None:
    provider = FakeProvider([
        ScriptedTurn(text=_call("fs.read", path=123)),
        ScriptedTurn(text="ok"),
    ])
    agent = make_agent(provider)
    await agent.ask("read")
    assert "invalid arguments" in agent.state.messages[2].content


# -- provider errors and retries -------------------------------------------

async def test_transient_provider_error_is_retried_then_succeeds(make_agent, monkeypatch) -> None:
    monkeypatch.setattr("gemma_cyber.agent.runtime.RETRY_BASE_DELAY", 0.0)
    provider = FakeProvider([
        ScriptedTurn(raises=RateLimitError("429")),
        ScriptedTurn(text="recovered"),
    ])
    agent = make_agent(provider)
    result = await agent.ask("hi")
    assert result.ok and result.final_text == "recovered"


async def test_authentication_error_is_not_retried(make_agent) -> None:
    from gemma_cyber.agent.errors import AuthenticationError

    provider = FakeProvider([ScriptedTurn(raises=AuthenticationError("bad key"))] * 5)
    agent = make_agent(provider)
    result = await agent.ask("hi")
    assert not result.ok and result.error_code == "authentication_error"
    assert provider.turns_consumed == 1


async def test_retries_are_capped(make_agent, monkeypatch) -> None:
    monkeypatch.setattr("gemma_cyber.agent.runtime.RETRY_BASE_DELAY", 0.0)
    provider = FakeProvider([ScriptedTurn(raises=ProviderError("500"))] * 10)
    agent = make_agent(provider)
    result = await agent.ask("hi")
    assert not result.ok and result.error_code == "provider_error"
    assert provider.turns_consumed == 3


async def test_error_event_in_stream_becomes_a_turn_error(make_agent, monkeypatch) -> None:
    monkeypatch.setattr("gemma_cyber.agent.runtime.RETRY_BASE_DELAY", 0.0)
    provider = FakeProvider([ScriptedTurn(error="model is loading")] * 5)
    agent = make_agent(provider)
    result = await agent.ask("hi")
    assert not result.ok and "loading" in (result.error or "")


# -- cancellation -----------------------------------------------------------

async def test_cancel_stops_an_in_flight_stream(make_agent) -> None:
    provider = FakeProvider([ScriptedTurn(text="x" * 400, delay_s=5.0, chunk_size=1)])
    agent = make_agent(provider)

    async def cancel_soon() -> None:
        await asyncio.sleep(0.05)
        agent.state.cancel.cancel()

    task = asyncio.create_task(agent.ask("slow please"))
    asyncio.create_task(cancel_soon())
    result = await asyncio.wait_for(task, timeout=5.0)
    assert result.stop_reason == "cancelled" and result.error_code == "cancelled"


async def test_cancel_token_is_thread_safe_from_a_worker(make_agent) -> None:
    # The provider runs on a worker thread; the token must be settable from
    # anywhere. (An asyncio.Event here would be a data race.)
    provider = FakeProvider([ScriptedTurn(text="y" * 200, delay_s=3.0, chunk_size=1)])
    agent = make_agent(provider)
    threading.Timer(0.05, agent.state.cancel.cancel).start()
    result = await asyncio.wait_for(agent.ask("slow"), timeout=5.0)
    assert result.stop_reason == "cancelled"


# -- events and usage -------------------------------------------------------

async def test_events_are_emitted_for_the_ui(make_agent) -> None:
    provider = FakeProvider([
        ScriptedTurn(text="thinking " + _call("fs.read", path="README.md")),
        ScriptedTurn(text="done", usage=Usage(input_tokens=10, output_tokens=3)),
    ])
    agent = make_agent(provider)
    await agent.ask("go")
    kinds = agent.bus.kinds()
    assert EventKind.MODEL_TOKEN in kinds
    assert EventKind.TOOL_STARTED in kinds and EventKind.TOOL_COMPLETED in kinds
    assert EventKind.SESSION_SAVED in kinds and EventKind.TURN_FINISHED in kinds
    # The raw envelope is never emitted to the UI.
    tokens = "".join(e.get("text", "") for e in agent.bus.history
                     if e.kind == EventKind.MODEL_TOKEN)
    assert "<tool_call>" not in tokens


async def test_usage_accumulates(make_agent) -> None:
    provider = FakeProvider([ScriptedTurn(text="hi", usage=Usage(input_tokens=7, output_tokens=2))])
    agent = make_agent(provider)
    result = await agent.ask("hi")
    assert result.usage.input_tokens == 7 and result.usage.output_tokens == 2


# -- compaction -------------------------------------------------------------

async def test_compact_replaces_history_with_a_summary(make_agent) -> None:
    provider = FakeProvider([
        ScriptedTurn(text="a"), ScriptedTurn(text="b"),
        ScriptedTurn(text="SUMMARY: we read a file."),
    ])
    agent = make_agent(provider)
    await agent.ask("one")
    await agent.ask("two")
    before = len(agent.state.messages)
    summary = await agent.runtime.compact(agent.state)
    assert "SUMMARY" in summary
    assert len(agent.state.messages) < before
    assert "[earlier conversation summary]" in agent.state.messages[0].content


async def test_context_overflow_triggers_one_compaction_then_stops(make_agent) -> None:
    provider = FakeProvider(
        [ScriptedTurn(text="summary"), ScriptedTurn(text="still too big")],
        capabilities=ModelCapabilities(context_tokens=64),  # smaller than the policy
    )
    agent = make_agent(provider)
    result = await agent.ask("hello")
    assert not result.ok and result.error_code == "context_overflow"


@pytest.mark.parametrize("mode", list(PermissionMode))
async def test_every_mode_can_complete_a_plain_turn(make_agent, mode) -> None:
    agent = make_agent(FakeProvider([ScriptedTurn(text="ok")]), mode=mode)
    result = await agent.ask("hi")
    assert result.ok


# -- small-model ergonomics -------------------------------------------------

async def test_a_malformed_envelope_gets_one_syntax_correction(make_agent) -> None:
    """The 4B failure seen against a live model: ```tool_call> instead of <tool_call>."""
    provider = FakeProvider([
        ScriptedTurn(text='Let me look.\n```tool_call>{"name": "fs.read", '
                          '"arguments": {"path": "README.md"}}```'),
        ScriptedTurn(text=_call("fs.read", path="README.md")),
        ScriptedTurn(text="It is a test project."),
    ])
    agent = make_agent(provider)
    result = await agent.ask("read the readme")

    assert result.ok and result.tool_calls == 1
    hint = [m for m in agent.state.messages if m.name == "codec"][0]
    assert "<tool_call>" in hint.content
    assert result.final_text == "It is a test project."


async def test_the_syntax_correction_is_given_only_once(make_agent) -> None:
    """A model that keeps getting it wrong must not loop on the hint."""
    provider = FakeProvider([ScriptedTurn(text="```tool_call> nope ```")] * 6)
    agent = make_agent(provider)
    result = await agent.ask("read something")
    assert result.ok  # it terminates as an ordinary answer, not a loop
    hints = [m for m in agent.state.messages if m.name == "codec"]
    assert len(hints) == 1


async def test_a_plain_answer_is_never_mistaken_for_an_attempted_call(make_agent) -> None:
    agent = make_agent(FakeProvider([ScriptedTurn(text="The answer is 42.")]))
    result = await agent.ask("what is the answer")
    assert result.final_text == "The answer is 42."
    assert not [m for m in agent.state.messages if m.name == "codec"]
