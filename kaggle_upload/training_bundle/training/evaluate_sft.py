"""Evaluate an adapter against a held-out CSV and report macro-F1."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from .data import LABELS, read_examples
from .prompts import TARGET_MODES, to_prompt_completion


def extract_label(text: str) -> str | None:
    """Extract final_label from JSON, with a conservative text fallback."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("final_label") in LABELS:
            return payload["final_label"]
    for label in sorted(LABELS, key=len, reverse=True):
        if text.strip().casefold() == label.casefold():
            return label
        field_match = re.search(
            rf"final_label\s*[\"']?\s*:\s*[\"']?({re.escape(label)})\b",
            text,
            flags=re.IGNORECASE,
        )
        if field_match:
            return label
    return None


def classification_report(
    y_true: list[str], y_pred: list[str | None]
) -> tuple[float, dict[str, dict[str, float | int]], dict[str, dict[str, int]]]:
    """Return fixed-label Macro-F1, per-class metrics, and a confusion matrix."""
    predicted_labels = (*LABELS, "<unparsed>")
    per_class: dict[str, dict[str, float | int]] = {}
    confusion = {label: {predicted: 0 for predicted in predicted_labels} for label in LABELS}
    f1_values: list[float] = []

    for expected, predicted in zip(y_true, y_pred):
        confusion[expected][predicted if predicted in LABELS else "<unparsed>"] += 1

    for label in LABELS:
        true_positive = confusion[label][label]
        support = sum(confusion[label].values())
        false_positive = sum(confusion[other][label] for other in LABELS if other != label)
        false_negative = support - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)

    return sum(f1_values) / len(LABELS), per_class, confusion


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError("Install training dependencies before running adapter evaluation.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Adapter evaluation requires a CUDA GPU in this Unsloth setup.")

    examples = read_examples(args.data, args.zip_member, args.target_mode)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    predictions: list[str | None] = []
    records: list[dict[str, str | None]] = []
    for index, row in enumerate(examples):
        prompt = to_prompt_completion(row, args.target_mode)["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = extract_label(generated)
        predictions.append(predicted)
        records.append({
            "row": str(index),
            "input_text": row["input_text"],
            "gold_label": row["final_label"],
            "predicted_label": predicted,
            "raw_generation": generated,
        })
        if (index + 1) % 25 == 0:
            print(f"Generated {index + 1}/{len(examples)}")

    gold = [row["final_label"] for row in examples]
    score, per_class, confusion = classification_report(gold, predictions)
    accuracy = sum(expected == predicted for expected, predicted in zip(gold, predictions)) / len(gold)
    report: dict[str, object] = {
        "adapter": str(args.adapter),
        "target_mode": args.target_mode,
        "dataset": str(args.data),
        "zip_member": args.zip_member,
        "examples": len(gold),
        "accuracy": accuracy,
        "macro_f1": score,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "gold_support": dict(Counter(gold)),
        "unparsed_predictions": sum(pred is None for pred in predictions),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--zip-member")
    parser.add_argument("--target-mode", choices=(*TARGET_MODES, "label-only", "distilled"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.adapter.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {args.adapter}")
    if args.max_new_tokens <= 0 or args.max_seq_length < 128:
        raise ValueError("max-new-tokens must be positive and max-seq-length at least 128")
    evaluate(args)


if __name__ == "__main__":
    main()
