"""`OllamaChatProvider` — streaming `/api/chat` for the agent (plan §9).

This is a *new* client, not a refactor of `gemma_cyber.clients.OllamaClient`.
That one owns `/api/generate` for the hosted API and the evaluation harness and
keeps its `requests` dependency; rewriting it to share HTTP code would put the
eval path at risk for no benefit. Two clients, two contracts, zero coupling.

Everything Ollama-specific is isolated here: trailing-slash hosts, the
`message.thinking` field on reasoning models, tool-call shapes, and NDJSON
frames that may be blank, partial, or unparseable.
"""

from __future__ import annotations

import json
import logging
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

__all__ = ["OllamaChatProvider"]

logger = logging.getLogger("gemma_cyber.agent")

DEFAULT_BASE_URL = "http://127.0.0.1:11434"

#: Frames without content are normal in Ollama's stream (keep-alives, the final
#: done frame). Malformed ones are counted and tolerated up to this many before
#: the stream is declared broken — a corrupt endpoint should not spin forever.
MAX_MALFORMED_FRAMES = 20


class OllamaChatProvider:
    """Streaming chat against a local Ollama daemon."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 180.0,
        tools_native: bool = False,
        context_tokens: int | None = None,
        client: httpx.Client | None = None,
        name: str = "ollama",
    ) -> None:
        self.name = name
        self.model = model
        # Normalise once: a configured "http://host:11434/" must not produce
        # "http://host:11434//api/chat", which some proxies 404.
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._tools_native = tools_native
        self._context_tokens = context_tokens
        self._client = client
        self._owns_client = client is None

    # -- transport ----------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=10.0))
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    # -- ChatProvider -------------------------------------------------------

    def capabilities(self) -> ModelCapabilities:
        """Static, configuration-driven. Native tools stay off unless opted in.

        Ollama advertises tool support per model, but a 4B model that accepts the
        schema and never emits a call is indistinguishable from one with nothing
        to do — so the honest default is the XML codec (plan §9, §33).
        """
        return ModelCapabilities(
            streaming=True,
            tools_native=self._tools_native,
            tools_emulated=True,
            json_schema=False,
            vision=False,
            reasoning=False,
            context_tokens=self._context_tokens,
        )

    def complete(self, request: ChatRequest) -> Iterator[StreamEvent]:
        payload = self._build_payload(request)
        url = f"{self.base_url}/api/chat"
        malformed = 0
        saw_done = False

        try:
            with self.client.stream("POST", url, json=payload) as response:
                self._raise_for_status(response)
                for line in response.iter_lines():
                    if request.cancel.is_cancelled():
                        # Leaving the context manager closes the connection, which
                        # is what actually stops generation on the daemon side.
                        return
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        frame = json.loads(raw)
                    except json.JSONDecodeError:
                        malformed += 1
                        if malformed > MAX_MALFORMED_FRAMES:
                            raise ProviderError(
                                "Ollama returned an unparseable stream "
                                f"({malformed} malformed frames)"
                            ) from None
                        continue
                    if not isinstance(frame, dict):
                        malformed += 1
                        continue
                    yield from self._frame_events(frame)
                    if frame.get("done"):
                        saw_done = True
                        break
        except httpx.HTTPStatusError:
            raise
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ProviderError(
                f"cannot reach the model runtime at {self.base_url} — is `ollama serve` running?"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ProviderError(
                f"the model runtime stopped responding after {self.timeout_s:.0f}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"transport error talking to {self.base_url}: {exc!r}") from exc

        if not saw_done and not request.cancel.is_cancelled():
            # A stream that ends without `done` truncated mid-response. Say so
            # rather than presenting a partial answer as complete.
            yield StreamEvent(
                type="error",
                error="the model stream ended without a completion marker",
                error_code="provider_error",
            )
        yield StreamEvent.done()

    # -- payload / frames ---------------------------------------------------

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": request.temperature}
        if request.seed is not None:
            options["seed"] = request.seed
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.stop:
            options["stop"] = list(request.stop)

        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [self._wire_message(m) for m in request.messages],
            "stream": True,
            "options": options,
        }
        if request.tools and self._tools_native:
            payload["tools"] = [spec.to_openai_schema() for spec in request.tools]
        return payload

    def _wire_message(self, message: Message) -> dict[str, Any]:
        """Ollama's message shape, with the `tool`-role quirk handled.

        Models without native tool support (gemma3:4b among them) have no `tool`
        role in their chat template. Measured against a local daemon: a
        ``role="tool"`` message is silently dropped, and the model then
        *fabricates* the file contents it never received — the exact failure the
        system policy forbids, caused by the transport rather than the model.

        So on the emulated path a tool result is delivered as a labelled user
        message. Its content already carries the `<untrusted source="tool:...">`
        wrapper from `ContextManager`, so the trust framing survives the change.
        """
        if message.role == "tool" and not self._tools_native:
            label = message.name or "tool"
            return {
                "role": "user",
                "content": f"Result of tool `{label}`:\n{message.content}",
            }
        wire: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "tool" and message.name:
            wire["tool_name"] = message.name
        if message.tool_calls:
            wire["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}}
                for c in message.tool_calls
            ]
        return wire

    def _frame_events(self, frame: dict[str, Any]) -> Iterator[StreamEvent]:
        if error := frame.get("error"):
            yield StreamEvent(type="error", error=str(error), error_code="provider_error")
            return

        message = frame.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                yield StreamEvent.delta(content)
            # Reasoning models stream hidden chain-of-thought in `thinking`. It is
            # deliberately dropped: it is not an answer, and re-injecting it into
            # the next request wastes the context budget a 4B model cannot spare.
            for call in self._parse_native_calls(message.get("tool_calls")):
                yield StreamEvent.call(call)

        if frame.get("done"):
            usage = self._parse_usage(frame)
            if usage is not None:
                yield StreamEvent(type="usage", usage=usage)

    @staticmethod
    def _parse_native_calls(raw: Any) -> list[ToolCall]:
        """Decode Ollama's native tool-call array, skipping anything malformed."""
        if not isinstance(raw, list):
            return []
        calls: list[ToolCall] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if not isinstance(arguments, dict):
                continue
            calls.append(ToolCall(name=name.strip(), arguments=arguments))
        return calls

    @staticmethod
    def _parse_usage(frame: dict[str, Any]) -> Usage | None:
        prompt = frame.get("prompt_eval_count")
        output = frame.get("eval_count")
        if not isinstance(prompt, int) and not isinstance(output, int):
            return None
        return Usage(
            input_tokens=prompt if isinstance(prompt, int) else 0,
            output_tokens=output if isinstance(output, int) else 0,
        )

    # -- errors -------------------------------------------------------------

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        # Read the (short) error body so the message is useful; a streaming
        # response has not been consumed at this point.
        try:
            response.read()
            detail = response.text.strip()[:300]
        except Exception:  # pragma: no cover - defensive
            detail = ""
        if status == 429:
            raise RateLimitError(f"model runtime is rate limiting (429): {detail}")
        if status in (401, 403):
            raise AuthenticationError(f"model runtime rejected credentials ({status})")
        if status == 404:
            raise ProviderError(
                f"model not found on the runtime (404). Try `ollama pull <model>`. {detail}"
            )
        raise ProviderError(f"model runtime returned HTTP {status}: {detail}")

    # -- doctor support -----------------------------------------------------

    def list_models(self) -> list[str]:
        """Local model tags, for `gemma4 doctor` / `gemma4 models`."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            data = response.json()
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ProviderError(
                f"cannot reach the model runtime at {self.base_url} — is `ollama serve` running?"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"cannot list models at {self.base_url}: {exc!r}") from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ProviderError("model runtime returned an invalid /api/tags response") from exc
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return []
        return [str(m.get("name")) for m in models if isinstance(m, dict) and m.get("name")]
