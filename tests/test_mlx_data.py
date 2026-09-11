"""Tests for the MLX-LoRA data preparation (local Apple-Silicon training path)."""

from __future__ import annotations

import json

import pytest

from gemma_cyber.training.mlx_data import _split_index, build_mlx_dataset

pytestmark = pytest.mark.filterwarnings("ignore")

SRC = "data/training/sft_v0.2.jsonl"


def test_split_is_deterministic_and_stable() -> None:
    a = [_split_index(f"id-{i}", 0.12) for i in range(200)]
    b = [_split_index(f"id-{i}", 0.12) for i in range(200)]
    assert a == b
    frac = sum(a) / len(a)
    assert 0.03 < frac < 0.25  # roughly val_fraction, hash noise tolerated


def test_build_mlx_dataset_roundtrips(tmp_path) -> None:
    manifest = build_mlx_dataset(SRC, tmp_path / "mlx", val_fraction=0.12)
    counts: dict[str, int] = manifest["counts"]
    assert counts["train"] + counts["valid"] == counts["total"] == 277
    assert counts["valid"] > 0

    train_lines = (tmp_path / "mlx" / "train.jsonl").read_text().strip().splitlines()
    assert len(train_lines) == counts["train"]
    row = json.loads(train_lines[0])
    assert set(row) == {"messages"}
    assert row["messages"][-1]["role"] == "assistant"
    assert {m["role"] for m in row["messages"]} <= {"system", "user", "assistant"}


def test_build_is_reproducible(tmp_path) -> None:
    m1: dict = build_mlx_dataset(SRC, tmp_path / "a", val_fraction=0.12)
    m2: dict = build_mlx_dataset(SRC, tmp_path / "b", val_fraction=0.12)
    assert m1["outputs"]["train"]["sha256"] == m2["outputs"]["train"]["sha256"]
    assert m1["outputs"]["valid"]["sha256"] == m2["outputs"]["valid"]["sha256"]


def test_empty_validation_split_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="validation split is empty"):
        build_mlx_dataset(SRC, tmp_path / "z", val_fraction=0.0)
