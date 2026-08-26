"""Tests for scripts/verify_gguf_export.py (file/size/checksum logic).

GGUF-metadata parsing needs the optional `gguf` package + a real model file, so
it is not exercised here; the file existence, size gate, and checksum — the parts
that turn "export succeeded" into a checkable claim — are.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_gguf_export.py"
_spec = importlib.util.spec_from_file_location("verify_gguf_export", _SCRIPT)
assert _spec and _spec.loader
verify_gguf_export = importlib.util.module_from_spec(_spec)
sys.modules["verify_gguf_export"] = verify_gguf_export
_spec.loader.exec_module(verify_gguf_export)


def test_missing_file_reports_missing(tmp_path: Path) -> None:
    manifest = verify_gguf_export.verify_export(tmp_path / "nope.gguf")
    assert manifest["ok"] is False
    assert manifest["missing"] is True
    assert manifest["checks"][0]["name"] == "file_exists"
    assert manifest["checks"][0]["passed"] is False


def test_small_file_fails_size_gate_but_records_checksum(tmp_path: Path) -> None:
    f = tmp_path / "tiny.gguf"
    f.write_bytes(b"not a real model")
    manifest = verify_gguf_export.verify_export(f, min_size_mb=1.0)
    assert manifest["ok"] is False  # fails the size gate
    names = {c["name"]: c["passed"] for c in manifest["checks"]}
    assert names["file_exists"] is True
    assert names["min_size"] is False
    # Checksum is always computed and matches hashlib on the same bytes.
    expected = hashlib.sha256(b"not a real model").hexdigest()
    assert manifest["sha256"] == expected


def test_size_gate_passes_for_large_enough_file(tmp_path: Path) -> None:
    f = tmp_path / "big.gguf"
    f.write_bytes(b"\0" * (2 * 1024 * 1024))  # 2 MiB
    manifest = verify_gguf_export.verify_export(f, min_size_mb=1.0)
    names = {c["name"]: c["passed"] for c in manifest["checks"]}
    assert names["min_size"] is True
    # `gguf` package is absent in CI -> readability check is skipped, not failed.
    assert names["gguf_readable"] is True
    assert manifest["ok"] is True


def test_sha256_helper_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "blob.bin"
    payload = b"deadbeef" * 4096
    f.write_bytes(payload)
    assert verify_gguf_export.sha256_file(f) == hashlib.sha256(payload).hexdigest()
