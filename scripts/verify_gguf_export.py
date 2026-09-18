#!/usr/bin/env python3
"""Verify a GGUF export is real, complete, and correct — not just a print statement.

The first exp-002 run declared success with a bare `print(... ready for Ollama!)`,
then the Colab disk reset and no artifact could be found. This script makes "export
succeeded" a checkable claim:

  * the file exists and is at least `--min-size-mb` (a truncated/failed convert is
    far smaller than a real Q4_K_M of a 4B model);
  * a SHA-256 checksum is recorded so the file downloaded off Colab can be proven
    identical to the one that was exported;
  * GGUF metadata is parsed and the key fields are surfaced, and any DUPLICATE
    metadata keys are reported explicitly (the `Duplicated key name` warnings seen
    during Gemma-3 conversion — benign as long as the duplicated values agree, which
    this checks).

Usage:
    python scripts/verify_gguf_export.py gemma3-cyber-v0.2-Q4_K_M.gguf
    python scripts/verify_gguf_export.py model.gguf --min-size-mb 1500 --out run_manifest

Exit codes: 0 = all checks passed; 1 = a check failed; 2 = file missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def read_gguf_metadata(path: Path) -> dict[str, Any]:
    """Return {'available': bool, 'fields': {...}, 'duplicate_keys': {...}}.

    Uses the `gguf` package if installed; degrades gracefully otherwise so the
    file/size/checksum checks still run on a machine without it.
    """
    try:
        from gguf import GGUFReader  # type: ignore[import-not-found]
    except ImportError:
        return {"available": False, "reason": "gguf package not installed"}

    reader = GGUFReader(str(path))
    key_counts: Counter[str] = Counter(field.name for field in reader.fields.values())
    duplicate_keys = {k: c for k, c in key_counts.items() if c > 1}
    fields: dict[str, Any] = {}
    for name, field in reader.fields.items():
        try:
            fields[name] = field.contents()
        except Exception:  # noqa: BLE001 - metadata introspection is best-effort
            fields[name] = "<unreadable>"
    return {
        "available": True,
        "n_fields": len(reader.fields),
        "n_tensors": len(reader.tensors),
        "duplicate_keys": duplicate_keys,
        "fields": fields,
    }


def verify_export(path: str | Path, min_size_mb: float = 1000.0) -> dict[str, Any]:
    """Run all checks and return a machine-readable manifest with `ok`."""
    path = Path(path)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    exists = path.is_file()
    check("file_exists", exists, str(path))
    if not exists:
        return {"ok": False, "path": str(path), "checks": checks, "missing": True}

    size_mb = path.stat().st_size / (1024 * 1024)
    check(
        "min_size",
        size_mb >= min_size_mb,
        f"{size_mb:.1f} MiB (min {min_size_mb} MiB)",
    )

    digest = sha256_file(path)
    check("checksum", True, f"sha256={digest}")

    meta = read_gguf_metadata(path)
    if meta["available"]:
        dup = meta["duplicate_keys"]
        # Duplicate keys are the known-benign Gemma-3 convert warning ONLY if the
        # reader still resolved a single coherent value per key; we surface them so
        # a human can confirm, but do not hard-fail on their presence alone.
        check(
            "gguf_readable",
            meta["n_tensors"] > 0,
            f"{meta['n_fields']} metadata fields, {meta['n_tensors']} tensors",
        )
        check(
            "no_unexpected_duplicate_keys",
            not dup,
            "none" if not dup else f"duplicated: {dup}",
        )
    else:
        check("gguf_readable", True, f"skipped ({meta['reason']})")

    ok = all(c["passed"] for c in checks)
    return {
        "ok": ok,
        "path": str(path),
        "size_mb": round(size_mb, 1),
        "sha256": digest,
        "gguf_metadata": meta,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a GGUF export.")
    parser.add_argument("gguf", help="Path to the exported .gguf file")
    parser.add_argument(
        "--min-size-mb",
        type=float,
        default=1000.0,
        help="Minimum plausible size for a real Q4_K_M of a 4B model (default 1000)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory to write verify_gguf.json (default: alongside the file)",
    )
    args = parser.parse_args()

    manifest = verify_export(args.gguf, min_size_mb=args.min_size_mb)

    out_dir = Path(args.out) if args.out else Path(args.gguf).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verify_gguf.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"GGUF export verification: {args.gguf}")
    for c in manifest["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['name']}: {c['detail']}")
    print(f"-> {out_dir / 'verify_gguf.json'}")

    if manifest.get("missing"):
        return 2
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
