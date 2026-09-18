"""Typed error hierarchy for the `gemma4` agent.

Callers discriminate on type, never on message text: the runtime's error table
(plan §23) decides retry / surface-to-model / abort-turn per class, and the CLI
maps classes to stable exit codes. Messages are user-facing and must stay
secret-free and path-sanitised — see :func:`safe_message`.

Security note: every failure here is a *fail-closed* outcome. A path that cannot
be resolved, a tool call that cannot be validated, or a permission decision that
cannot be made all raise rather than degrade into "allow".
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "AgentError",
    "AuthenticationError",
    "CommandTimeout",
    "ConfigurationError",
    "ContextOverflowError",
    "LoopDetected",
    "PatchConflict",
    "PermissionDenied",
    "ProviderError",
    "RateLimitError",
    "SessionError",
    "ToolError",
    "ToolNotFound",
    "TurnTimeout",
    "UserCancelled",
    "WorkspaceEscape",
    "safe_message",
]


class AgentError(RuntimeError):
    """Base class for every error raised by `gemma_cyber.agent`.

    ``code`` is the stable identifier shown to users and written to the audit
    log; it never changes even if the human-readable message is reworded.
    """

    code = "agent_error"
    #: Whether the runtime should hand this back to the model as a tool result
    #: (so it can recover) rather than aborting the turn. See plan §23.
    to_model = False

    def __str__(self) -> str:  # pragma: no cover - trivial
        return super().__str__()


# -- provider / transport ---------------------------------------------------

class ProviderError(AgentError):
    """The chat provider failed (connection, 5xx, malformed stream)."""

    code = "provider_error"


class RateLimitError(ProviderError):
    """Provider returned 429. Retried with backoff, capped at 3 attempts."""

    code = "rate_limit"


class AuthenticationError(ProviderError):
    """Provider rejected credentials. Never retried — retrying cannot help."""

    code = "authentication_error"


# -- context ----------------------------------------------------------------

class ContextOverflowError(AgentError):
    """The packed request exceeds the model's context budget after compaction."""

    code = "context_overflow"


# -- tools ------------------------------------------------------------------

class ToolError(AgentError):
    """A tool failed in a way the model may be able to recover from."""

    code = "tool_error"
    to_model = True


class ToolNotFound(ToolError):
    """The model named a tool that is not registered, or is hidden by the mode.

    Hidden-by-mode is reported as "not found" deliberately: the model does not
    get to learn that a privileged tool exists in another mode.
    """

    code = "tool_not_found"


class PermissionDenied(ToolError):
    """`PermissionGuard` refused the call. Returned to the model; never retried."""

    code = "permission_denied"


class WorkspaceEscape(PermissionDenied):
    """A path resolved outside the workspace jail (traversal or symlink)."""

    code = "workspace_escape"


class PatchConflict(ToolError):
    """`fs.edit` preconditions failed: stale hash, or `old_string` not unique."""

    code = "patch_conflict"


class CommandTimeout(ToolError):
    """`shell.exec` exceeded its timeout and the process group was killed."""

    code = "command_timeout"


# -- runtime ----------------------------------------------------------------

class LoopDetected(AgentError):
    """The model repeated one identical tool call too many times."""

    code = "loop_detected"


class TurnTimeout(AgentError):
    """The wall-clock budget for a single user turn was exhausted."""

    code = "turn_timeout"


class UserCancelled(AgentError):
    """Ctrl+C / cancel token. Never retried, never surfaced as a model error."""

    code = "cancelled"


# -- configuration / persistence -------------------------------------------

class ConfigurationError(AgentError):
    """Configuration is structurally invalid or unsafe. Raised at startup."""

    code = "configuration_error"


class SessionError(AgentError):
    """A session could not be created, appended to, or loaded."""

    code = "session_error"


# -- message hygiene --------------------------------------------------------

def safe_message(exc: BaseException, *, workspace_root: Path | None = None) -> str:
    """Render ``exc`` for a user/model-visible surface without leaking paths.

    Absolute paths outside the workspace are the leak we care about: an OSError
    from a jail check can carry ``/Users/alice/.ssh/id_rsa``, and that string
    would then be echoed into the model's context. Paths *inside* the workspace
    are rewritten relative to the root (useful, not sensitive); anything that
    still looks like an absolute path elsewhere is replaced with ``<path>``.
    """
    text = str(exc) or exc.__class__.__name__
    if workspace_root is not None:
        root = str(workspace_root)
        text = text.replace(root + os.sep, "").replace(root, ".")
    home = os.path.expanduser("~")
    if home and home != "/":
        text = text.replace(home, "~")
    return _mask_absolute_paths(text)


def _mask_absolute_paths(text: str) -> str:
    """Replace remaining absolute-looking POSIX/Windows paths with ``<path>``."""
    out: list[str] = []
    for token in text.split(" "):
        stripped = token.strip("'\"()[],;:")
        looks_absolute = stripped.startswith("/") and len(stripped) > 1
        looks_windows = len(stripped) > 3 and stripped[1] == ":" and stripped[2] in "\\/"
        out.append("<path>" if (looks_absolute or looks_windows) else token)
    return " ".join(out)
