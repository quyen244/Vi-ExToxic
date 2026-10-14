"""CLI for no-CoT, CoT, and CoT plus structured-distillation QLoRA SFT."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import SFTConfig
from .data import read_examples
from .prompts import TARGET_MODES
from .trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/archive.zip"))
    parser.add_argument(
        "--zip-member",
        help="CSV member/path inside a ZIP or extracted directory (defaults to dataset_phase_1.csv for the project data)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument(
        "--target-mode",
        choices=(*TARGET_MODES, "label-only", "distilled"),
        required=True,
        help="no-cot: Vietnamese label-only; cot: Vietnamese CoT; cot-distilled: Vietnamese structured CoT; cot-en: English-instruction CoT",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-steps", type=int, default=-1, help="-1 trains for all epochs")
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Checkpoint directory to resume from, or 'latest' to use newest output-dir checkpoint",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the selected dataset and config without loading CUDA/Unsloth",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.zip_member is None and (
        args.data.suffix.lower() == ".zip" or args.data.is_dir()
    ):
        args.zip_member = "dataset_phase_1.csv"
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = SFTConfig(
        data=args.data,
        zip_member=args.zip_member,
        output_dir=args.output_dir,
        model_name=args.model,
        target_mode=args.target_mode,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    config.validate()
    rows = read_examples(config.data, config.zip_member, config.target_mode)
    prompt_language = "en" if config.target_mode == "cot-en" else "vi"
    print(f"Validated {len(rows)} rows from {config.data} for {config.target_mode} SFT (prompt language: {prompt_language}).")
    if not args.validate_only:
        train(config)


if __name__ == "__main__":
    main()
