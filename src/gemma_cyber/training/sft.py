"""SFT formatting, completion-only masking, and version-robust config helpers.

Why this module exists
----------------------
The Colab notebook and `scripts/train_qlora.py` previously implemented training
*differently*: the notebook rendered Gemma-3 turns with the repo's verified
``to_gemma_chat_text`` and masked the prompt with a vendored collator, while the
script relied on TRL applying the tokenizer chat template to a raw ``messages``
column — exactly the path ``gemma_cyber.data.formatting`` warns is unsafe for
Gemma (it rejects a standalone ``system`` role). That divergence meant the
"official" script could train on a different (or broken) format than the run we
actually validated. This module makes both paths share one implementation.

Heavy imports (torch/transformers/trl) are deferred into the functions that use
them so the pure helpers below import and test on a CPU-only box with no ML
stack installed.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

from gemma_cyber.data.formatting import to_gemma_chat_text

# The exact string that begins a Gemma-3 model turn in our chat format. Loss is
# computed only on tokens at/after this marker (the model's completion).
RESPONSE_TEMPLATE = "<start_of_turn>model\n"


def format_for_sft(messages: list[dict[str, str]]) -> dict[str, str]:
    """Render one chat example to the ``{"text": ...}`` row SFTTrainer consumes.

    Uses the repo's deterministic Gemma-3 formatter (folds ``system`` into the
    first user turn, emits ``model`` turns) rather than the tokenizer template.
    """
    return {"text": to_gemma_chat_text(messages)}


def completion_mask_start(
    label_ids: list[int], response_ids: list[int]
) -> int | None:
    """Number of leading tokens to mask so loss falls only on the completion.

    Returns the index one-past the FIRST occurrence of ``response_ids`` (the
    ``<start_of_turn>model\\n`` marker) in ``label_ids`` — every token before and
    including the marker should be set to the ignore index. Returns ``None`` when
    the marker is absent, signalling the caller to mask the whole example (it then
    contributes no loss rather than leaking prompt tokens into the objective).

    Pure list logic on purpose: it is the core correctness of the masking and is
    unit-tested without torch.
    """
    if not response_ids:
        return None
    first = response_ids[0]
    n = len(response_ids)
    last_start = len(label_ids) - n
    for i in range(last_start + 1):
        if label_ids[i] == first and label_ids[i : i + n] == response_ids:
            return i + n
    return None


def build_completion_only_collator(
    tokenizer: Any,
    response_template: str = RESPONSE_TEMPLATE,
    ignore_index: int = -100,
) -> Any:
    """A DataCollator that masks everything up to the first model turn.

    Recent TRL removed ``DataCollatorForCompletionOnlyLM``; this vendors a minimal
    equivalent on transformers' stable ``DataCollatorForLanguageModeling`` and
    reuses :func:`completion_mask_start` so the masking rule has a single tested
    implementation. Imports torch/transformers lazily (GPU-only environment).
    """
    from transformers import DataCollatorForLanguageModeling

    # add_special_tokens=False: `<start_of_turn>` is already a special token, so
    # this matches the ids that appear mid-sequence in the rendered text.
    response_ids = tokenizer.encode(response_template, add_special_tokens=False)

    class CompletionOnlyCollator(DataCollatorForLanguageModeling):
        def __init__(self) -> None:
            super().__init__(tokenizer=tokenizer, mlm=False)

        def torch_call(self, examples: Any) -> Any:
            batch = super().torch_call(examples)
            for i in range(len(batch["labels"])):
                labels = batch["labels"][i]
                cut = completion_mask_start(labels.tolist(), response_ids)
                if cut is None:
                    batch["labels"][i, :] = ignore_index
                else:
                    batch["labels"][i, :cut] = ignore_index
            return batch

    return CompletionOnlyCollator()


def make_sft_config_kwargs(
    sft_config_cls: type, base_kwargs: dict[str, Any], max_seq_length: int
) -> dict[str, Any]:
    """Filter kwargs to those the installed ``SFTConfig`` accepts.

    TRL renamed ``max_seq_length`` to ``max_length`` and drops/adds fields across
    versions. This maps the sequence-length key to whichever exists and removes
    any unrecognized field, so one call site works across TRL releases.
    """
    fields = {f.name for f in dataclasses.fields(sft_config_cls)}
    kwargs = dict(base_kwargs)
    if "max_length" in fields:
        kwargs["max_length"] = max_seq_length
    elif "max_seq_length" in fields:
        kwargs["max_seq_length"] = max_seq_length
    return {k: v for k, v in kwargs.items() if k in fields}


def make_trainer_kwargs(
    trainer_cls: type,
    *,
    model: Any,
    args: Any,
    train_dataset: Any,
    data_collator: Any,
    tokenizer: Any,
    peft_config: Any | None = None,
) -> dict[str, Any]:
    """Select trainer kwargs by the installed ``SFTTrainer`` signature.

    Handles the ``tokenizer`` -> ``processing_class`` rename and only passes
    ``peft_config`` when supported, so the same call works across TRL versions.
    """
    params = set(inspect.signature(trainer_cls).parameters)
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }
    if peft_config is not None and "peft_config" in params:
        kwargs["peft_config"] = peft_config
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    return kwargs
