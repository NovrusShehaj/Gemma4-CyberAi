"""Text hygiene applied to everything that crosses a trust boundary.

Two boundaries, both mandatory (plan §16.6):

1. **Before the terminal.** Tool output and model text may contain escape
   sequences that move the cursor, rewrite the scrollback, set the window title,
   or embed OSC-8 hyperlinks — a file in the workspace is enough to attempt it.
2. **Before the next model request.** This one is the easy one to forget. If a
   tool's raw stdout is re-injected into the conversation, the escape sequences
   are still there on the *next* render, and the model has been handed a
   channel it should not have.

This module lives outside `ui/` on purpose: `tools/` must be able to sanitise
without importing the UI layer (plan §11 dependency direction).

Nothing here is a substitute for the permission model. Stripping escapes stops
terminal injection; it does not stop prompt injection.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

__all__ = [
    "TruncationResult",
    "redact_secrets",
    "redact_value",
    "sanitize_for_model",
    "sanitize_for_terminal",
    "strip_ansi",
    "truncate_with_digest",
]

# OSC (window title, OSC-8 hyperlinks, clipboard writes): ESC ] ... BEL | ST.
# Unterminated sequences are stripped to end-of-string on purpose — leaving the
# tail intact would let a crafted prefix re-open the sequence on the terminal.
_OSC = re.compile(r"(?:\x1b\]|\x9d)[\s\S]*?(?:\x07|\x1b\\|\x9c|$)")
# DCS / SOS / PM / APC — same "consume to ST" shape, same tail rule.
_STRING_CMD = re.compile(r"(?:\x1b[P^_X]|[\x90\x98\x9e\x9f])[\s\S]*?(?:\x1b\\|\x9c|\x07|$)")
# CSI: cursor movement, colours, scroll regions, DECSET/DECRST private modes.
_CSI = re.compile(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]?")
# Two-character escapes (ESC c full reset, ESC 7/8 cursor save, charset selects).
_SIMPLE_ESC = re.compile(r"\x1b[ -/]*[0-~]?")
# C0 controls we never want verbatim. \t and \n survive; \r is normalised first,
# and a bare \r is dropped because it can overwrite a rendered line in place.
_C0 = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")
# C1 controls (U+0080–U+009F). Legitimate text does not contain these.
_C1 = re.compile(r"[\x80-\x9f]")


def strip_ansi(text: str) -> str:
    """Remove escape sequences and control characters, preserving tabs/newlines.

    Order matters: string-terminated commands (OSC/DCS/APC) are consumed before
    CSI, otherwise a CSI-looking fragment *inside* an OSC payload would be
    removed first and leave the OSC introducer to swallow real output.
    """
    if not text:
        return text
    out = text.replace("\r\n", "\n")
    out = _OSC.sub("", out)
    out = _STRING_CMD.sub("", out)
    out = _CSI.sub("", out)
    out = _SIMPLE_ESC.sub("", out)
    out = out.replace("\r", "\n")
    out = _C0.sub("", out)
    out = _C1.sub("", out)
    return out


# -- secret redaction -------------------------------------------------------

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key", re.compile(
        r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----")),
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{12,20}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("bearer", re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]{8,}")),
    ("assignment", re.compile(
        r"(?i)\b([A-Z0-9_]*(?:api[_-]?key|secret|password|passwd|token|credential)"
        r"[A-Z0-9_]*)\s*[:=]\s*[\"']?([^\s\"',;)]{6,})")),
)


def redact_secrets(text: str) -> str:
    """Replace credential-looking substrings with ``[REDACTED:<kind>]``.

    Best effort, deliberately conservative on the generic ``key = value`` rule
    (it needs a credential-shaped *name*), because over-redacting a diff or a
    grep hit makes the agent useless. This runs on model-facing tool output and
    on audit records; it is not a reason to ever log a file body in the first
    place.
    """
    if not text:
        return text
    out = text
    for kind, pattern in _SECRET_PATTERNS:
        if kind == "bearer":
            out = pattern.sub(lambda m: f"{m.group(1)}[REDACTED:bearer]", out)
        elif kind == "assignment":
            out = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED:assignment]", out)
        else:
            out = pattern.sub(f"[REDACTED:{kind}]", out)
    return out


def redact_value(value: Any, *, max_chars: int = 200) -> Any:
    """Redact and bound an arbitrary JSON-ish value for the audit log."""
    if isinstance(value, str):
        cleaned = redact_secrets(strip_ansi(value))
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + f"...[+{len(cleaned) - max_chars} chars]"
        return cleaned
    if isinstance(value, dict):
        return {k: redact_value(v, max_chars=max_chars) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v, max_chars=max_chars) for v in value[:20]]
    return value


# -- truncation -------------------------------------------------------------

class TruncationResult:
    """Bounded text plus proof of what was dropped."""

    __slots__ = ("text", "truncated", "remainder_sha256", "omitted_bytes")

    def __init__(
        self, text: str, truncated: bool, remainder_sha256: str | None, omitted_bytes: int
    ) -> None:
        self.text = text
        self.truncated = truncated
        self.remainder_sha256 = remainder_sha256
        self.omitted_bytes = omitted_bytes


def truncate_with_digest(text: str, max_bytes: int) -> TruncationResult:
    """Cap ``text`` at ``max_bytes`` of UTF-8, hashing the remainder (plan §11.6).

    The digest is what makes truncation auditable: the model is told output was
    cut and gets a stable identifier for the omitted part instead of silently
    reasoning about a partial file as if it were whole.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return TruncationResult(text, False, None, 0)
    head, tail = encoded[:max_bytes], encoded[max_bytes:]
    digest = hashlib.sha256(tail).hexdigest()
    return TruncationResult(
        head.decode("utf-8", errors="ignore"), True, digest, len(tail)
    )


# -- the two boundary helpers ----------------------------------------------

def sanitize_for_terminal(text: str) -> str:
    """Everything rendered to the user goes through here first."""
    return strip_ansi(text)


def sanitize_for_model(text: str) -> str:
    """Everything re-injected into the conversation goes through here first.

    Escapes are stripped *and* secrets redacted: a tool that happened to read a
    credential should not persist it into `messages.jsonl`.
    """
    return redact_secrets(strip_ansi(text))
