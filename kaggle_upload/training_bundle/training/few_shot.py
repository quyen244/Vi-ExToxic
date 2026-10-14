"""Load fixed, non-holdout exemplars for the few-shot inference ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FEW_SHOT_PATH = Path(__file__).with_name("few_shot_examples.json")
FEW_SHOT_LABELS = ("Implicit Toxicity", "Identity-Based Hate", "Ambiguous/Noise")
REQUIRED_FIELDS = (
    "input_text",
    "input_emotion",
    "semantic_decoding",
    "slang_interpretation",
    "contextual_conflict",
    "target",
    "thought_trace",
    "final_label",
    "confidence_score",
)


def load_few_shot_examples() -> list[dict[str, Any]]:
    """Read the fixed Vietnamese rare-class exemplars shipped with the package."""
    payload = json.loads(FEW_SHOT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Few-shot example file must contain a nonempty JSON list: {FEW_SHOT_PATH}")
    examples: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Few-shot example {index} must be a JSON object")
        missing = [field for field in REQUIRED_FIELDS if field not in item]
        if missing:
            raise ValueError(f"Few-shot example {index} is missing fields: {', '.join(missing)}")
        if item["final_label"] not in FEW_SHOT_LABELS:
            raise ValueError(
                f"Few-shot example {index} must target a selected minority label, "
                f"got {item['final_label']!r}"
            )
        examples.append(item)

    observed = {item["final_label"] for item in examples}
    if observed != set(FEW_SHOT_LABELS):
        raise ValueError(f"Few-shot examples must cover {FEW_SHOT_LABELS}; got {sorted(observed)}")
    return examples
