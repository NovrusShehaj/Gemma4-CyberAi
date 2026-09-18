"""The `ChatProvider` port (plan §9).

Separate from `InferenceEngine` on purpose. `InferenceEngine.generate()` is the
single-turn ``/api/generate`` contract that the hosted API and the evaluation
harness depend on (it satisfies `SupportsGenerate`); stretching it to carry a
messages array and tool calls would change scorecard behaviour. Chat+tools is a
different contract, so it gets a different port.

A provider owns exactly one thing: turning a vendor's wire protocol into
`StreamEvent`s. It does not decide permissions, does not read files, and does
not know what any tool does.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from gemma_cyber.agent.types import ChatRequest, ModelCapabilities, StreamEvent

__all__ = ["ChatProvider", "provider_context_floor"]

#: When a provider cannot report a context budget, the ContextManager assumes
#: this. Small and honest: over-estimating silently truncates mid-request.
DEFAULT_CONTEXT_FLOOR = 8192


@runtime_checkable
class ChatProvider(Protocol):
    """Streaming chat with optional tool schemas."""

    #: Stable identifier used in session metadata and `gemma4 models`.
    name: str
    #: The concrete model tag in use.
    model: str

    def capabilities(self) -> ModelCapabilities:
        """Report what this provider/model actually supports.

        Never optimistic. `tools_native` stays false until a provider has passed
        the contract tests, because a model that silently ignores tool schemas
        looks identical to a model with nothing to do.
        """
        ...

    def complete(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Stream one completion.

        Implementations must:

        * yield ``done`` exactly once, last, on a successful stream;
        * yield ``error`` (with ``error_code``) instead of raising, where the
          failure is recoverable enough for the runtime's retry table to act on;
        * poll ``request.cancel`` between chunks and stop promptly when set.
        """
        ...

    def close(self) -> None:
        """Release transport resources. Safe to call more than once."""
        ...


def provider_context_floor(capabilities: ModelCapabilities) -> int:
    """Context budget to plan with, given possibly-unknown capabilities."""
    return capabilities.context_tokens or DEFAULT_CONTEXT_FLOOR
