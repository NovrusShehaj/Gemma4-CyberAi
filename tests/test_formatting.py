"""Tests for deterministic Gemma-3 chat formatting."""

import pytest

from gemma_cyber.data.builder_v2 import build_sft_v2_dataset
from gemma_cyber.data.formatting import to_gemma_chat_text


def test_system_folded_into_first_user_turn():
    text = to_gemma_chat_text([
        {"role": "system", "content": "You are careful."},
        {"role": "user", "content": "What is Kerberoasting?"},
        {"role": "assistant", "content": "It is T1558.003."},
    ])
    # No standalone system turn; system is prepended to the user body.
    assert "<start_of_turn>system" not in text
    assert "<start_of_turn>user\nYou are careful.\n\nWhat is Kerberoasting?<end_of_turn>" in text
    # assistant becomes the Gemma `model` role.
    assert "<start_of_turn>model\nIt is T1558.003.<end_of_turn>" in text


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        to_gemma_chat_text([{"role": "tool", "content": "x"}])


def test_empty_raises():
    with pytest.raises(ValueError):
        to_gemma_chat_text([])


def test_every_v2_item_formats_and_is_nonempty():
    """The whole training set must render without error (no unknown roles/empties)."""
    for it in build_sft_v2_dataset():
        text = to_gemma_chat_text(it.to_chat_dict())
        assert text.startswith("<start_of_turn>user\n")
        assert text.rstrip().endswith("<end_of_turn>")
        assert "<start_of_turn>model\n" in text
