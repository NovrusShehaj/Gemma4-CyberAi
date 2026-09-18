"""Shared QLoRA/SFT building blocks for Gemma-3 fine-tuning.

Single source of truth so `scripts/train_qlora.py` and
`notebooks/colab_qlora_training.ipynb` format, mask, and configure training
identically. Heavy deps (torch/transformers/trl) are imported lazily inside the
functions that need them, so this package imports cleanly in the CPU-only CI
environment and its pure helpers stay unit-testable there.
"""

from __future__ import annotations

from gemma_cyber.training.sft import (
    RESPONSE_TEMPLATE,
    build_completion_only_collator,
    completion_mask_start,
    format_for_sft,
    make_sft_config_kwargs,
    make_trainer_kwargs,
)

__all__ = [
    "RESPONSE_TEMPLATE",
    "build_completion_only_collator",
    "completion_mask_start",
    "format_for_sft",
    "make_sft_config_kwargs",
    "make_trainer_kwargs",
]
