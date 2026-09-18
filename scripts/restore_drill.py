#!/usr/bin/env python3
"""Backup and restore the model registry (and print Ollama volume commands).

The application has no database. Durable state is:

  * ``data/models/registry.json`` — promotion audit trail (also in git)
  * Compose volume ``ollama-models`` — pulled GGUF weights

This script copies the registry file and prints the docker volume archive
commands. It does not invent RTO/RPO SLAs: recovery time is "restore the last
backup and restart compose"; data loss window is "since the last copy".

Usage:
    python scripts/restore_drill.py backup --dest /tmp/gc-backup/registry.json
    python scripts/restore_drill.py restore --src /tmp/gc-backup/registry.json
    python scripts/restore_drill.py volume-commands
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gemma_cyber.inference.config import DEFAULT_REGISTRY_PATH  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_registry(src: Path, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def restore_registry(src: Path, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def files_match(a: Path, b: Path) -> bool:
    return sha256_file(a) == sha256_file(b)


def volume_commands(project: str = "gemma4-cyberai") -> str:
    vol = f"{project}_ollama-models"
    return f"""# Archive the Ollama model volume (run on the host that has the volume):
docker run --rm \\
  -v {vol}:/src:ro \\
  -v "$(pwd)/backups:/backup" \\
  alpine tar czf /backup/ollama-models.tgz -C /src .

# Restore onto a clean compose project (volume must exist or be created empty):
docker volume create {vol}
docker run --rm \\
  -v {vol}:/dest \\
  -v "$(pwd)/backups:/backup:ro" \\
  alpine tar xzf /backup/ollama-models.tgz -C /dest

# Then:
#   docker compose up -d
#   curl -sS http://127.0.0.1:8000/v1/models
# Registry JSON should match the backup SHA; pulled tags should match `ollama list`.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Registry backup/restore drill.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_backup = sub.add_parser("backup", help="Copy registry.json to --dest")
    p_backup.add_argument("--src", type=Path, default=DEFAULT_REGISTRY_PATH)
    p_backup.add_argument("--dest", type=Path, required=True)

    p_restore = sub.add_parser("restore", help="Copy a backup onto --dest")
    p_restore.add_argument("--src", type=Path, required=True)
    p_restore.add_argument("--dest", type=Path, default=DEFAULT_REGISTRY_PATH)

    p_vol = sub.add_parser(
        "volume-commands", help="Print docker commands to archive/restore ollama-models"
    )
    p_vol.add_argument("--project", default="gemma4-cyberai")

    args = parser.parse_args(argv)
    if args.cmd == "backup":
        dest = backup_registry(args.src, args.dest)
        print(f"backed up {args.src} -> {dest}")
        print(f"sha256 {sha256_file(dest)}")
        return 0
    if args.cmd == "restore":
        dest = restore_registry(args.src, args.dest)
        print(f"restored {args.src} -> {dest}")
        print(f"sha256 {sha256_file(dest)}")
        if not files_match(args.src, dest):
            print("ERROR: restored file does not match backup", file=sys.stderr)
            return 1
        return 0
    print(volume_commands(args.project))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
