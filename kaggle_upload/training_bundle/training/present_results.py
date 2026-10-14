"""Present the completed Kaggle evaluations as a readable Markdown comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NO_SHOT_EXPERIMENTS = (
    ("no-cot", "No-CoT (Vietnamese prompt)"),
    ("cot-distilled", "CoT-distilled (Vietnamese prompt)"),
    ("cot", "CoT (Vietnamese prompt)"),
    ("cot-en", "CoT (English prompt)"),
)
FEW_SHOT_MODES = ("no-cot", "cot", "cot-distilled")
TITLES = dict(NO_SHOT_EXPERIMENTS)


def _read_report(path: Path, mode: str, expected_few_shot: bool) -> dict[str, Any]:
    if not path.is_file():
        treatment = "few-shot" if expected_few_shot else "no-shot"
        raise FileNotFoundError(f"Missing {treatment} evaluation report for {mode}: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("target_mode") != mode:
        raise ValueError(f"{path} contains target_mode={report.get('target_mode')!r}; expected {mode!r}.")
    if bool(report.get("few_shot")) != expected_few_shot:
        raise ValueError(f"{path} has few_shot={report.get('few_shot')!r}; expected {expected_few_shot}.")
    return report


def _read_reports(results_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    no_shot = {
        mode: _read_report(results_dir / mode / "evaluation" / "evaluation.json", mode, False)
        for mode, _ in NO_SHOT_EXPERIMENTS
    }
    few_shot = {
        mode: _read_report(results_dir / mode / "evaluation-few-shot" / "evaluation.json", mode, True)
        for mode in FEW_SHOT_MODES
    }
    combined = [*no_shot.values(), *few_shot.values()]
    dataset_paths = {report.get("dataset") for report in combined}
    example_counts = {report.get("examples") for report in combined}
    supports = {json.dumps(report.get("gold_support"), sort_keys=True) for report in combined}
    if len(dataset_paths) != 1 or len(example_counts) != 1 or len(supports) != 1:
        raise ValueError("Evaluation reports do not use the same dataset, example count, and gold-label support")
    return no_shot, few_shot


def _percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_markdown(
    no_shot: dict[str, dict[str, Any]],
    few_shot: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Vi-ExToxic Evaluation Comparison",
        "",
        "All runs use the same 320-row held-out dataset and deterministic decoding.",
        "Few-shot examples come from the supplied example and the training split, never the holdout.",
        "",
        "## No-shot results",
        "",
        "| Run | Prompt language | Examples | Accuracy | Macro-F1 | Unparsed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for mode, title in NO_SHOT_EXPERIMENTS:
        report = no_shot[mode]
        language = "English" if mode == "cot-en" else "Vietnamese"
        lines.append(
            f"| {title} | {language} | {report['examples']} | {_percentage(report['accuracy'])} | "
            f"{_percentage(report['macro_f1'])} | {report['unparsed_predictions']} |"
        )

    cot_delta = (no_shot["cot-en"]["macro_f1"] - no_shot["cot"]["macro_f1"]) * 100
    delta_word = "higher" if cot_delta > 0 else "lower" if cot_delta < 0 else "the same"
    lines.extend(
        [
            "",
            "## CoT prompt-language comparison (no-shot)",
            "",
            f"English CoT Macro-F1 is {abs(cot_delta):.2f} percentage points {delta_word} than Vietnamese CoT.",
            "",
            "## Few-shot impact",
            "",
            "| Run | No-shot Accuracy | Few-shot Accuracy | No-shot Macro-F1 | Few-shot Macro-F1 | Macro-F1 change |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in FEW_SHOT_MODES:
        plain = no_shot[mode]
        shot = few_shot[mode]
        delta = (shot["macro_f1"] - plain["macro_f1"]) * 100
        lines.append(
            f"| {TITLES[mode]} | {_percentage(plain['accuracy'])} | {_percentage(shot['accuracy'])} | "
            f"{_percentage(plain['macro_f1'])} | {_percentage(shot['macro_f1'])} | {delta:+.2f} pp |"
        )

    labels = list(no_shot["no-cot"]["per_class"])
    lines.extend(
        [
            "",
            "## Per-class F1: no-shot",
            "",
            "| Label | Support | No-CoT | CoT-distilled | CoT-VI | CoT-EN |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in labels:
        support = no_shot["no-cot"]["per_class"][label]["support"]
        scores = [_percentage(no_shot[mode]["per_class"][label]["f1"]) for mode, _ in NO_SHOT_EXPERIMENTS]
        lines.append(f"| {label} | {support} | " + " | ".join(scores) + " |")

    lines.extend(
        [
            "",
            "## Per-class F1: few-shot impact",
            "",
            "| Label | Support | No-CoT (0 → 3) | CoT-VI (0 → 3) | CoT-distilled (0 → 3) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in labels:
        support = no_shot["no-cot"]["per_class"][label]["support"]
        comparisons = []
        for mode in ("no-cot", "cot", "cot-distilled"):
            before = no_shot[mode]["per_class"][label]["f1"]
            after = few_shot[mode]["per_class"][label]["f1"]
            comparisons.append(f"{_percentage(before)} → {_percentage(after)}")
        lines.append(f"| {label} | {support} | " + " | ".join(comparisons) + " |")

    total_support = sum(no_shot["no-cot"]["per_class"][label]["support"] for label in labels)
    rare = [
        f"{label} (n={no_shot['no-cot']['per_class'][label]['support']})"
        for label in labels
        if no_shot["no-cot"]["per_class"][label]["support"] < 10
    ]
    lines.extend(
        [
            "",
            f"Total held-out support: {total_support} examples.",
            f"Classes with fewer than 10 examples: {', '.join(rare) if rare else 'none'}.",
            "Interpret per-class scores with their support, especially for rare labels.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("/kaggle/working/outputs"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/outputs/comparison.md"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    no_shot, few_shot = _read_reports(args.results_dir)
    report = render_markdown(no_shot, few_shot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved comparison to: {args.output}")


if __name__ == "__main__":
    main()
