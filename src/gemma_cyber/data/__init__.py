"""Data tooling: schemas, validators, and contamination checks (PROJECT_PLAN.md §16)."""

from gemma_cyber.data.contamination import (
    ContaminationReport,
    check_contamination,
    jaccard_similarity,
    normalize_text,
)
from gemma_cyber.data.schema import (
    TrainingItem,
    TrainingMessage,
    TrainingMetadata,
    load_training_dataset,
)

__all__ = [
    "ContaminationReport",
    "TrainingItem",
    "TrainingMessage",
    "TrainingMetadata",
    "check_contamination",
    "jaccard_similarity",
    "load_training_dataset",
    "normalize_text",
]
