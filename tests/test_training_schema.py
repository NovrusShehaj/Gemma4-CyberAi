"""Unit tests for the training dataset schema and loader."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gemma_cyber.data.schema import (
    TrainingItem,
    TrainingMessage,
    TrainingMetadata,
    load_training_dataset,
)


def _valid_item_dict(**overrides) -> dict:
    base = {
        "id": "train-0001",
        "schema_version": "1.0",
        "messages": [
            {"role": "system", "content": "You are a cybersecurity assistant."},
            {"role": "user", "content": "Explain CIA triad."},
            {"role": "assistant", "content": "CIA stands for Confidentiality, Integrity, and Availability."},
        ],
        "metadata": {
            "task_type": "fundamentals",
            "domain": "general",
            "difficulty": "intro",
            "requires_evidence": False,
            "fabricated_premise": False,
            "source": "original",
            "license": "CC-BY-4.0",
            "provenance": "authored for gemma-cyber sft_v0.1",
            "tags": ["cia_triad", "theory"],
        },
    }
    base.update(overrides)
    return base


def test_valid_training_item():
    item = TrainingItem(**_valid_item_dict())
    assert item.id == "train-0001"
    assert len(item.messages) == 3
    assert item.messages[-1].role == "assistant"
    assert item.metadata.task_type == "fundamentals"
    assert item.metadata.license == "CC-BY-4.0"


def test_empty_message_content_fails():
    with pytest.raises(ValidationError):
        TrainingMessage(role="user", content="   ")


def test_item_requires_at_least_two_messages():
    data = _valid_item_dict(
        messages=[{"role": "assistant", "content": "Only assistant."}]
    )
    with pytest.raises(ValidationError):
        TrainingItem(**data)


def test_last_message_must_be_assistant():
    data = _valid_item_dict(
        messages=[
            {"role": "system", "content": "System prompt."},
            {"role": "assistant", "content": "Assistant first."},
            {"role": "user", "content": "User last."},
        ]
    )
    with pytest.raises(ValidationError):
        TrainingItem(**data)


def test_must_have_at_least_one_user_message():
    data = _valid_item_dict(
        messages=[
            {"role": "system", "content": "System prompt."},
            {"role": "assistant", "content": "Assistant only."},
        ]
    )
    with pytest.raises(ValidationError):
        TrainingItem(**data)


def test_empty_license_or_provenance_fails():
    with pytest.raises(ValidationError):
        TrainingMetadata(
            task_type="fundamentals",
            domain="general",
            provenance="test",
            license="   ",
        )

    with pytest.raises(ValidationError):
        TrainingMetadata(
            task_type="fundamentals",
            domain="general",
            provenance="",
            license="CC-BY-4.0",
        )


def test_to_chat_dict_and_render_text():
    item = TrainingItem(**_valid_item_dict())
    chat = item.to_chat_dict()
    assert len(chat) == 3
    assert chat[0] == {"role": "system", "content": "You are a cybersecurity assistant."}
    assert "Confidentiality" in item.render_text()


def test_load_training_dataset_success_and_duplicate_rejection(tmp_path: Path):
    item1 = _valid_item_dict(id="item-1")
    item2 = _valid_item_dict(id="item-2")
    file_path = tmp_path / "dataset.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(item1) + "\n")
        f.write(json.dumps(item2) + "\n")

    loaded = load_training_dataset(file_path)
    assert len(loaded) == 2
    assert loaded[0].id == "item-1"
    assert loaded[1].id == "item-2"

    # Test duplicate ID failure
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item1) + "\n")

    with pytest.raises(ValueError, match="duplicate id 'item-1'"):
        load_training_dataset(file_path)
