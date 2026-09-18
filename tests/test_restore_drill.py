"""Registry backup/restore helpers used by the DR drill."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from restore_drill import (  # noqa: E402
    backup_registry,
    files_match,
    restore_registry,
    sha256_file,
    volume_commands,
)


def test_backup_restore_roundtrip(tmp_path):
    src = tmp_path / "registry.json"
    src.write_text('{"schema": "gemma-cyber/model-registry@1", "models": []}\n')
    backup = backup_registry(src, tmp_path / "backups" / "registry.json")
    assert files_match(src, backup)
    dest = tmp_path / "restored" / "registry.json"
    restore_registry(backup, dest)
    assert files_match(src, dest)
    assert sha256_file(dest) == sha256_file(src)


def test_volume_commands_mention_archive_and_restore():
    text = volume_commands("gemma4-cyberai")
    assert "ollama-models" in text
    assert "tar czf" in text
    assert "tar xzf" in text
    assert "/v1/models" in text
