"""Tests for the shared SFT helpers (`gemma_cyber.training.sft`).

These cover the pure logic that both the notebook and `scripts/train_qlora.py`
depend on — Gemma-3 rendering, completion-only masking boundary, and the
version-robust config/trainer kwargs selection — without needing torch/trl.
"""

from __future__ import annotations

import dataclasses

from gemma_cyber.training import (
    RESPONSE_TEMPLATE,
    completion_mask_start,
    format_for_sft,
    make_sft_config_kwargs,
    make_trainer_kwargs,
)


def test_format_for_sft_folds_system_and_emits_model_turn() -> None:
    messages = [
        {"role": "system", "content": "You are careful."},
        {"role": "user", "content": "Kerberoasting ID?"},
        {"role": "assistant", "content": "T1558.003, Credential Access."},
    ]
    text = format_for_sft(messages)["text"]
    # System folded into the first user turn; assistant emitted as `model`.
    assert text.startswith("<start_of_turn>user\nYou are careful.\n\nKerberoasting ID?")
    assert "<start_of_turn>model\nT1558.003" in text
    assert "assistant" not in text
    assert RESPONSE_TEMPLATE in text


def test_completion_mask_start_basic() -> None:
    # Marker [2,3] occurs at index 1; mask through it -> index one-past = 4.
    assert completion_mask_start([9, 2, 3, 4, 5], [2, 3]) == 3
    assert completion_mask_start([9, 1, 2, 3, 4], [2, 3]) == 4


def test_completion_mask_start_absent_returns_none() -> None:
    assert completion_mask_start([9, 1, 4, 5], [2, 3]) is None


def test_completion_mask_start_empty_template_returns_none() -> None:
    assert completion_mask_start([1, 2, 3], []) is None


def test_completion_mask_start_returns_first_occurrence() -> None:
    # Two model turns; loss must start after the FIRST marker only.
    labels = [0, 7, 8, 1, 7, 8, 2]
    assert completion_mask_start(labels, [7, 8]) == 3


def test_completion_mask_start_marker_at_end() -> None:
    assert completion_mask_start([0, 1, 7, 8], [7, 8]) == 4


@dataclasses.dataclass
class _NewSFTConfig:
    output_dir: str = ""
    num_train_epochs: int = 1
    max_length: int = 512  # recent TRL name


@dataclasses.dataclass
class _OldSFTConfig:
    output_dir: str = ""
    max_seq_length: int = 512  # older TRL name


def test_make_sft_config_kwargs_maps_max_length_and_filters_unknown() -> None:
    base = {"output_dir": "x", "num_train_epochs": 3, "packing": False, "bf16": True}
    kwargs = make_sft_config_kwargs(_NewSFTConfig, base, max_seq_length=1024)
    assert kwargs["max_length"] == 1024
    assert "max_seq_length" not in kwargs
    # Unknown-to-this-version fields are dropped so construction never errors.
    assert "packing" not in kwargs
    assert "bf16" not in kwargs
    assert kwargs["num_train_epochs"] == 3
    _NewSFTConfig(**kwargs)  # constructs cleanly


def test_make_sft_config_kwargs_falls_back_to_max_seq_length() -> None:
    kwargs = make_sft_config_kwargs(_OldSFTConfig, {"output_dir": "x"}, max_seq_length=2048)
    assert kwargs["max_seq_length"] == 2048
    assert "max_length" not in kwargs
    _OldSFTConfig(**kwargs)


class _ProcessingClassTrainer:
    def __init__(self, model, args, train_dataset, data_collator,  # noqa: ANN001
                 processing_class=None, peft_config=None):
        pass


class _LegacyTokenizerTrainer:
    def __init__(self, model, args, train_dataset, data_collator, tokenizer=None):  # noqa: ANN001
        pass


def test_make_trainer_kwargs_prefers_processing_class_and_passes_peft() -> None:
    kwargs = make_trainer_kwargs(
        _ProcessingClassTrainer,
        model="m", args="a", train_dataset="d", data_collator="c",
        tokenizer="tok", peft_config="lora",
    )
    assert kwargs["processing_class"] == "tok"
    assert "tokenizer" not in kwargs
    assert kwargs["peft_config"] == "lora"


def test_make_trainer_kwargs_legacy_tokenizer_and_no_peft() -> None:
    kwargs = make_trainer_kwargs(
        _LegacyTokenizerTrainer,
        model="m", args="a", train_dataset="d", data_collator="c",
        tokenizer="tok", peft_config="lora",
    )
    # This signature has no peft_config -> must not be passed (would TypeError).
    assert "peft_config" not in kwargs
    assert kwargs["tokenizer"] == "tok"
    assert "processing_class" not in kwargs
