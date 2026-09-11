"""ANSI/OSC stripping, secret redaction, and truncation with a digest.

The escape sequences here are real terminal-injection payloads: a workspace file
containing any of them reaches both the terminal and the next model request.
"""

from __future__ import annotations

import hashlib

import pytest

from gemma_cyber.agent.sanitize import (
    redact_secrets,
    redact_value,
    sanitize_for_model,
    sanitize_for_terminal,
    strip_ansi,
    truncate_with_digest,
)

# -- escape sequences -------------------------------------------------------

@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("\x1b]0;pwned\x07visible", "visible"),                     # OSC window title
        ("\x1b]2;title\x1b\\visible", "visible"),                    # OSC with ST
        ("\x1b]8;;http://evil.example\x07click\x1b]8;;\x07", "click"),  # OSC-8 hyperlink
        ("\x1b]52;c;cGF5bG9hZA==\x07x", "x"),                        # OSC-52 clipboard write
        ("\x1b[31mred\x1b[0m", "red"),                               # CSI colour
        ("\x1b[2J\x1b[Hcleared", "cleared"),                         # CSI clear screen
        ("\x1b[?1049hswitch", "switch"),                             # CSI private mode
        ("\x1bcreset", "reset"),                                     # ESC c full reset
        ("\x1bP+q544e\x1b\\dcs", "dcs"),                             # DCS
        ("\x1b_apc\x1b\\after", "after"),                            # APC
        ("\x9b31mcsi8bit", "csi8bit"),                               # 8-bit CSI
        ("a\x07b", "ab"),                                            # bare BEL
        ("a\x08b", "ab"),                                            # backspace overwrite
        ("keep\ttabs\nand\nnewlines", "keep\ttabs\nand\nnewlines"),
    ],
)
def test_escape_sequences_are_removed(payload: str, expected: str) -> None:
    assert strip_ansi(payload) == expected


def test_unterminated_osc_is_stripped_to_end_of_string() -> None:
    # Leaving the tail would let a crafted prefix re-open the sequence.
    assert strip_ansi("safe\x1b]0;never closed") == "safe"


def test_carriage_returns_cannot_overwrite_a_line() -> None:
    assert "\r" not in strip_ansi("real output\rFAKE OUTPUT")


def test_stripping_is_idempotent() -> None:
    payload = "\x1b]0;t\x07\x1b[31mx\x1b[0m"
    assert strip_ansi(strip_ansi(payload)) == strip_ansi(payload)


def test_plain_text_is_unchanged() -> None:
    text = "def main():\n    return 42  # ok [not markup]\n"
    assert strip_ansi(text) == text


# -- redaction --------------------------------------------------------------

@pytest.mark.parametrize(
    "secret",
    ["AKIAIOSFODNN7EXAMPLE",
     "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
     "github_pat_11ABCDEFG0abcdefghijklmnop",
     "sk-abcdefghijklmnopqrstuvwxyz",
     "xoxb-1234567890-abcdefghij",
     "AIzaSyA1234567890abcdefghijklmnopqrstuv",
     "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.abcdefghijkl"],
)
def test_known_credential_shapes_are_redacted(secret: str) -> None:
    assert secret not in redact_secrets(f"value: {secret} end")
    assert "REDACTED" in redact_secrets(f"value: {secret} end")


def test_private_key_blocks_are_redacted() -> None:
    block = ("-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\n-----END RSA PRIVATE KEY-----")
    assert "MIIEow" not in redact_secrets(block)


def test_credential_assignments_are_redacted() -> None:
    out = redact_secrets("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI\nDB_PASSWORD: hunter2xx")
    assert "wJalrXUtnFEMI" not in out and "hunter2xx" not in out


def test_ordinary_code_is_not_over_redacted() -> None:
    code = "def add(x, y):\n    return x + y  # token = word in a comment\n"
    assert redact_secrets(code).startswith("def add(x, y):")
    assert "return x + y" in redact_secrets(code)


def test_redact_value_bounds_and_recurses() -> None:
    out = redact_value({"path": "a.py", "new_string": "x" * 500,
                        "nested": {"api_key": "sk-abcdefghijklmnopqrst"}})
    assert "chars]" in out["new_string"]
    assert "sk-abcdefghij" not in str(out)


# -- the two boundaries -----------------------------------------------------

def test_terminal_boundary_strips_escapes() -> None:
    assert sanitize_for_terminal("\x1b]0;x\x07ok") == "ok"


def test_model_boundary_strips_and_redacts() -> None:
    payload = "\x1b[31mAKIAIOSFODNN7EXAMPLE\x1b[0m"
    cleaned = sanitize_for_model(payload)
    assert "\x1b" not in cleaned and "AKIAIOSF" not in cleaned


# -- truncation -------------------------------------------------------------

def test_truncation_reports_a_verifiable_digest() -> None:
    text = "a" * 100
    result = truncate_with_digest(text, 40)
    assert result.truncated and len(result.text) == 40
    assert result.omitted_bytes == 60
    assert result.remainder_sha256 == hashlib.sha256(b"a" * 60).hexdigest()


def test_short_text_is_untouched() -> None:
    result = truncate_with_digest("short", 100)
    assert not result.truncated and result.text == "short" and result.remainder_sha256 is None


def test_truncation_does_not_split_a_multibyte_character() -> None:
    result = truncate_with_digest("é" * 50, 25)
    assert "�" not in result.text
    result.text.encode("utf-8")  # must still be valid UTF-8
