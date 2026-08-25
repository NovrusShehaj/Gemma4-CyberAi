"""Deterministic Gemma-3 chat formatting for SFT.

Why this exists
---------------
TRL's `SFTTrainer` applies the tokenizer's chat template to a `messages` column. Gemma
chat templates historically **reject a standalone `system` role** (they expect system
content folded into the first user turn) and use the role name **`model`**, not
`assistant`. Our training data (`sft_v0.2`) uses `system` + `user` + `assistant`, so
relying on the template risks a silent format break or an outright error at train time.

This module renders the canonical Gemma-3 turn format directly, folding any `system`
message into the first user turn, so training has a reproducible, template-independent
`text` column. Format (one example, `<bos>` is added by the tokenizer):

    <start_of_turn>user
    {system}\\n\\n{user}<end_of_turn>
    <start_of_turn>model
    {assistant}<end_of_turn>

Only completion tokens (the `model` turns) should contribute to the loss; TRL's
completion-only / response-template handling or a data collator masks the prompt. See the
Colab notebook for the masking step that pairs with this format.
"""

from __future__ import annotations

START = "<start_of_turn>"
END = "<end_of_turn>"


def to_gemma_chat_text(messages: list[dict[str, str]]) -> str:
    """Render chat `messages` into Gemma-3 turn format.

    - A leading `system` message is folded into the first `user` turn.
    - `assistant` is emitted as Gemma's `model` role.
    - Raises ValueError on an empty conversation or an unknown role.
    """
    if not messages:
        raise ValueError("cannot format an empty message list")

    system_text = ""
    turns: list[str] = []
    pending_system_used = False
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        if role == "system":
            system_text = content
            continue
        if role == "user":
            body = content
            if system_text and not pending_system_used:
                body = f"{system_text}\n\n{content}"
                pending_system_used = True
            turns.append(f"{START}user\n{body}{END}")
        elif role == "assistant":
            turns.append(f"{START}model\n{content}{END}")
        else:
            raise ValueError(f"unknown role for Gemma formatting: {role!r}")

    if not turns:
        raise ValueError("conversation has no user/assistant turns")
    return "\n".join(turns)
