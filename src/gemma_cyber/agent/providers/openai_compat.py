"""`OpenAICompatProvider` — one adapter for every OpenAI-shaped endpoint (plan §9).

llama.cpp, vLLM, LM Studio, OpenRouter, a future Gemma4-CyberAI
`/v1/chat/completions`, and OpenAI itself all speak the same wire format. One
HTTP adapter covers them; N vendor SDKs would be N dependencies, N auth models,
and N places for a credential to leak. The official OpenAI SDK is deliberately
not a dependency.

The API key comes from the environment (`GEMMA4_API_KEY` by default) and is read
at request time. It is never stored in a config file, never logged, and never
included in an error message.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from gemma_cyber.agent.errors import AuthenticationError, ProviderError, RateLimitError
from gemma_cyber.agent.types import (
    ChatRequest,
    Message,
    ModelCapabilities,
    StreamEvent,
    ToolCall,
    Usage,
)

__all__ = ["OpenAICompatProvider"]

_DONE_SENTINEL = "[DONE]"


class OpenAICompatProvider:
    """Streaming `/chat/completions` over Server-Sent Events."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 180.0,
        tools_native: bool = True,
        context_tokens: int | None = None,
        client: httpx.Client | None = None,
        name: str = "openai-compatible",
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout_s = timeout_s
        self._tools_native = tools_native
        self._context_tokens = context_tokens
        self._client = client
        self._owns_client = client is None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=10.0))
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def capabilities(self) -> ModelCapabilities:
        """OpenAI-shaped endpoints usually do tools; `tools_native` stays configurable
        because "OpenAI-compatible" is a claim, not a guarantee."""
        return ModelCapabilities(
            streaming=True,
            tools_native=self._tools_native,
            tools_emulated=True,
            json_schema=self._tools_native,
            context_tokens=self._context_tokens,
        )

    # -- streaming ----------------------------------------------------------

    def complete(self, request: ChatRequest) -> Iterator[StreamEvent]:
        payload = self._build_payload(request)
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        partial_calls: dict[int, dict[str, Any]] = {}
        usage: Usage | None = None
        try:
            with self.client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
            ) as response:
                self._raise_for_status(response)
                for line in response.iter_lines():
                    if request.cancel.is_cancelled():
                        return
                    raw = line.strip()
                    if not raw or not raw.startswith("data:"):
                        continue
                    data = raw[len("data:"):].strip()
                    if data == _DONE_SENTINEL:
                        break
                    try:
                        frame = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(frame, dict):
                        continue
                    if error := frame.get("error"):
                        yield StreamEvent(type="error", error=str(error),
                                          error_code="provider_error")
                        continue
                    usage = self._merge_usage(usage, frame.get("usage"))
                    yield from self._choice_events(frame, partial_calls)
        except httpx.HTTPStatusError:
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ProviderError(f"cannot reach the endpoint at {self.base_url}") from exc
        except httpx.ReadTimeout as exc:
            raise ProviderError(
                f"the endpoint stopped responding after {self.timeout_s:.0f}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"transport error talking to {self.base_url}: {exc!r}") from exc

        for call in self._finish_calls(partial_calls):
            yield StreamEvent.call(call)
        if usage is not None:
            yield StreamEvent(type="usage", usage=usage)
        yield StreamEvent.done()

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [self._wire_message(m) for m in request.messages],
            "stream": True,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.tools and self._tools_native:
            payload["tools"] = [spec.to_openai_schema() for spec in request.tools]
        return payload

    @staticmethod
    def _wire_message(message: Message) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "tool",
                "content": message.content,
                "tool_call_id": message.tool_call_id or "unknown",
            }
        wire: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            wire["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in message.tool_calls
            ]
        return wire

    def _choice_events(
        self, frame: dict[str, Any], partial: dict[int, dict[str, Any]]
    ) -> Iterator[StreamEvent]:
        choices = frame.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                yield StreamEvent.delta(content)
            # Tool calls stream as fragments keyed by index; accumulate and emit
            # them once the stream is finished, never half-built.
            for fragment in delta.get("tool_calls") or []:
                if not isinstance(fragment, dict):
                    continue
                index = int(fragment.get("index", 0) or 0)
                slot = partial.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if identifier := fragment.get("id"):
                    slot["id"] = str(identifier)
                function = fragment.get("function")
                if isinstance(function, dict):
                    if name := function.get("name"):
                        slot["name"] += str(name)
                    if arguments := function.get("arguments"):
                        slot["arguments"] += str(arguments)

    @staticmethod
    def _finish_calls(partial: dict[int, dict[str, Any]]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for _, slot in sorted(partial.items()):
            name = str(slot.get("name", "")).strip()
            if not name:
                continue
            raw = slot.get("arguments") or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                continue  # a truncated argument blob is not a call
            if not isinstance(arguments, dict):
                continue
            call = ToolCall(name=name, arguments=arguments)
            if slot.get("id"):
                call.id = str(slot["id"])
            calls.append(call)
        return calls

    @staticmethod
    def _merge_usage(current: Usage | None, raw: Any) -> Usage | None:
        if not isinstance(raw, dict):
            return current
        return Usage(
            input_tokens=int(raw.get("prompt_tokens", 0) or 0),
            output_tokens=int(raw.get("completion_tokens", 0) or 0),
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        try:
            response.read()
            detail = response.text.strip()[:300]
        except Exception:  # pragma: no cover - defensive
            detail = ""
        if status == 429:
            raise RateLimitError(f"endpoint is rate limiting (429): {detail}")
        if status in (401, 403):
            # Never echo the key or the header back.
            raise AuthenticationError(
                f"endpoint rejected the credential ({status}); check GEMMA4_API_KEY"
            )
        raise ProviderError(f"endpoint returned HTTP {status}: {detail}")

    def list_models(self) -> list[str]:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            response = self.client.get(f"{self.base_url}/models", headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(f"cannot list models at {self.base_url}: {exc!r}") from exc
        entries = data.get("data") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            return []
        return [str(e.get("id")) for e in entries if isinstance(e, dict) and e.get("id")]
