"""Deterministic MLX-LoRA data preparation for Apple-Silicon local training.

Why this module exists
----------------------
`gemma_cyber.training.sft` covers the CUDA/transformers/PEFT/TRL path (cloud). The
local path on Apple Silicon is `mlx-lm` LoRA, which reads a directory of
``{train,valid,test}.jsonl`` files. This module renders a validated
``TrainingItem`` dataset into that layout, deterministically, so a local run is
reproducible and split hygiene is auditable.

Format choice: mlx-lm's ``ChatDataset`` (``{"messages": [...]}``) applies the
tokenizer chat template. For ``google/gemma-3-4b-it`` that template folds a
leading ``system`` turn into the first user turn and emits ``model`` turns —
byte-identical to the repo's :func:`gemma_cyber.data.formatting.to_gemma_chat_text`
(verified in tests). With ``mask_prompt: true`` mlx-lm masks every prompt token,
matching the completion-only-loss semantics of the cloud collator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gemma_cyber.data.schema import load_training_dataset


def _split_index(item_id: str, val_fraction: float) -> bool:
    """Stable per-item train/valid assignment.

    Hash of the item id → uniform [0,1); ``True`` means the item goes to the
    validation split. Deterministic and independent of row order, so adding
    examples does not reshuffle existing ones.
    """
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < val_fraction


def build_mlx_dataset(
    source: str | Path,
    out_dir: str | Path,
    *,
    val_fraction: float = 0.12,
) -> dict[str, Any]:
    """Render ``source`` (a training JSONL) into ``out_dir/{train,valid}.jsonl``.

    Returns a manifest dict (counts, sha256 of each output, source sha256).
    """
    items = load_training_dataset(source)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_rows: list[str] = []
    valid_rows: list[str] = []
    for it in items:
        row = json.dumps({"messages": it.to_chat_dict()}, ensure_ascii=False)
        if _split_index(it.id, val_fraction):
            valid_rows.append(row)
        else:
            train_rows.append(row)

    # Guard: mlx-lm needs a non-empty validation set.
    if not valid_rows:
        raise ValueError("validation split is empty; raise val_fraction")

    manifest: dict[str, Any] = {
        "source": str(source),
        "source_sha256": _sha256_file(source),
        "val_fraction": val_fraction,
        "counts": {"train": len(train_rows), "valid": len(valid_rows), "total": len(items)},
        "outputs": {},
    }
    for name, rows in (("train", train_rows), ("valid", valid_rows)):
        path = out / f"{name}.jsonl"
        text = "\n".join(rows) + "\n"
        path.write_text(text, encoding="utf-8")
        manifest["outputs"][name] = {
            "path": str(path),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()
