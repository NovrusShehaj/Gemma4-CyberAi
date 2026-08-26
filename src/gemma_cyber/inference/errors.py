"""Structured error hierarchy for the inference layer.

Callers (CLI, API, tests) discriminate on type rather than parsing strings, so
each surface can map an error to the right exit code / HTTP status consistently.
``OllamaError`` from the low-level client is wrapped into these before it reaches
a caller, so nothing above the engine imports the client's exceptions.
"""

from __future__ import annotations


class InferenceError(RuntimeError):
    """Base class for every error the inference layer raises."""


class ServiceUnavailableError(InferenceError):
    """The model runtime (Ollama) could not be reached at all."""


class ModelUnavailableError(InferenceError):
    """The runtime is up but the requested model/version is not present."""


class InferenceTimeoutError(InferenceError):
    """Generation exceeded the configured timeout after all retries."""


class RegistryError(InferenceError):
    """The model registry file is missing, malformed, or references an unknown model."""
