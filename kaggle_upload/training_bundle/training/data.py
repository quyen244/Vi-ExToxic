"""Read and validate CSV training examples from files, directories, or ZIPs."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from typing import TextIO

LABELS = (
    "Constructive/Clean",
    "Implicit Toxicity",
    "Explicit Hostility",
    "Identity-Based Hate",
    "Ambiguous/Noise",
)

REQUIRED_COLUMNS = ("input_text", "input_emotion", "final_label")
TARGET_MODES = ("no-cot", "cot", "cot-distilled", "cot-en", "label-only", "distilled")
_MODE_ALIASES = {"label-only": "no-cot", "distilled": "cot-distilled", "cot-en": "cot"}
CONFIDENCE_COLUMN = "confidence_score"
RATIONALE_COLUMNS = (
    "semantic_decoding",
    "slang_interpretation",
    "contextual_conflict",
    "target",
    "thought_trace",
)


def canonical_target_mode(target_mode: str) -> str:
    """Normalize legacy target-mode names to the current experiment names."""
    mode = _MODE_ALIASES.get(target_mode, target_mode)
    if mode not in {"no-cot", "cot", "cot-distilled"}:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    return mode

def _open_csv(path: Path, member: str | None) -> TextIO:
    """Open a CSV, a ZIP member, or a CSV inside an extracted directory."""
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        candidates = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if member:
            if member not in candidates:
                archive.close()
                raise ValueError(f"ZIP member is not a CSV file in archive: {member}")
            selected = member
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            archive.close()
            raise ValueError(
                f"ZIP contains {len(candidates)} CSV files; select one with --zip-member. "
                f"Available: {', '.join(candidates)}"
            )
        return _ZipTextHandle(archive, archive.open(selected, "r"))
    if path.is_dir():
        if member:
            relative_member = Path(member)
            if relative_member.is_absolute() or ".." in relative_member.parts:
                raise ValueError("--zip-member must be a relative path inside the dataset directory")
            selected_path = path / relative_member
            if not selected_path.is_file() and relative_member.parent == Path("."):
                matches = list(path.rglob(relative_member.name))
                if len(matches) == 1:
                    selected_path = matches[0]
            if not selected_path.is_file():
                raise FileNotFoundError(f"CSV member does not exist in dataset directory: {path / relative_member}")
            if selected_path.suffix.lower() != ".csv":
                raise ValueError(f"Dataset member is not a CSV file: {selected_path}")
            return selected_path.open("r", encoding="utf-8-sig", newline="")
        candidates = sorted(path.rglob("*.csv"))
        if len(candidates) != 1:
            raise ValueError(
                f"Dataset directory contains {len(candidates)} CSV files; select one with --zip-member. "
                f"Available: {', '.join(str(p.relative_to(path)) for p in candidates)}"
            )
        return candidates[0].open("r", encoding="utf-8-sig", newline="")
    if member:
        raise ValueError("--zip-member can only be used with a .zip or directory input")
    return path.open("r", encoding="utf-8-sig", newline="")


class _ZipTextHandle:
    """Small context-manager wrapper for a UTF-8 CSV stream inside ZIP."""

    def __init__(self, archive: zipfile.ZipFile, binary_stream: object) -> None:
        import io

        self.archive = archive
        self.stream = io.TextIOWrapper(binary_stream, encoding="utf-8-sig", newline="")

    def __enter__(self) -> TextIO:
        return self.stream

    def __exit__(self, *exc: object) -> None:
        self.stream.close()
        self.archive.close()


def read_examples(
    path: str | Path,
    zip_member: str | None = None,
    target_mode: str = "label-only",
) -> list[dict[str, str]]:
    """Load rows from a schema-checked CSV, retaining only relevant columns.

    An empty input_emotion is allowed and formatted as "Không rõ". CoT modes
    require a teacher thought trace and numeric confidence. CoT-distilled also
    requires the four structured rationale fields.
    """
    mode = canonical_target_mode(target_mode)
    source = Path(path).expanduser()
    if not source.is_file() and not source.is_dir():
        raise FileNotFoundError(f"Dataset file or directory does not exist: {source}")

    with _open_csv(source, zip_member) as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {source}")
        headers = [h.strip().lstrip("\ufeff") if h else "" for h in reader.fieldnames]
        required_columns = REQUIRED_COLUMNS
        if mode == "cot":
            required_columns += ("thought_trace", CONFIDENCE_COLUMN)
        elif mode == "cot-distilled":
            required_columns += RATIONALE_COLUMNS + (CONFIDENCE_COLUMN,)
        missing = [column for column in required_columns if column not in headers]
        if missing:
            raise ValueError(
                f"Dataset is missing required columns {missing}. Found columns: {headers}"
            )

        examples: list[dict[str, str]] = []
        for line_number, raw in enumerate(reader, start=2):
            row = {key.strip().lstrip("\ufeff"): (value or "").strip()
                   for key, value in raw.items() if key is not None}
            text = row.get("input_text", "")
            label = row.get("final_label", "")
            if not text or not label:
                raise ValueError(
                    f"Empty input_text or final_label at CSV row {line_number} in {source}"
                )
            if label not in LABELS:
                raise ValueError(
                    f"Unknown final_label {label!r} at row {line_number}; expected one of {LABELS}"
                )
            if mode in {"cot", "cot-distilled"}:
                rationale_columns = ("thought_trace",)
                if mode == "cot-distilled":
                    rationale_columns = RATIONALE_COLUMNS
                empty_rationale = [column for column in rationale_columns if not row.get(column, "")]
                if empty_rationale:
                    raise ValueError(
                        f"Empty teacher explanation field(s) {empty_rationale} at CSV row "
                        f"{line_number} in {source}"
                    )
                try:
                    confidence = float(row[CONFIDENCE_COLUMN])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric confidence_score at row {line_number} in {source}"
                    ) from exc
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(
                        f"confidence_score must be in [0, 1] at row {line_number} in {source}"
                    )
            example_columns = REQUIRED_COLUMNS + RATIONALE_COLUMNS + (CONFIDENCE_COLUMN,)
            example = {column: row.get(column, "") for column in example_columns}
            examples.append(example)

    if not examples:
        raise ValueError(f"Dataset contains no examples: {source}")
    return examples


