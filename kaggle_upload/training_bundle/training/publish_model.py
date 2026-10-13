"""Merge a trained Unsloth LoRA adapter and publish the model to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"


def _read_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication is missing. Set HF_TOKEN, or add a Kaggle Notebook "
            "Secret named HF_TOKEN with write access to Hugging Face models."
        ) from exc
    if not token:
        raise RuntimeError(
            "Kaggle Secret HF_TOKEN is empty. Add a Hugging Face token with write access to models."
        )
    return token


def _model_card(base_model: str, target_mode: str, repo_id: str) -> str:
    mode_description = {
        "label-only": "predict the toxicity category without an explanation",
        "distilled": "predict the toxicity category and provide a concise explanation",
    }[target_mode]
    return f"""---
language:
  - vi
pipeline_tag: text-generation
base_model: {base_model}
tags:
  - unsloth
  - lora
  - vi-extoxic
---

# Vi-ExToxic Vietnamese Toxicity Model

This is the merged 16-bit model produced by supervised fine-tuning the `{base_model}` base model.

Training target: `{target_mode}`. The model is intended to {mode_description} for Vietnamese social-media comments.

The fine-tuning data contains teacher-generated labels. Treat predictions as research outputs, not as moderation decisions without human review.

## Load with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
```
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True, help="Directory saved by training.train_sft")
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repository, e.g. user/vi-extoxic-baseline")
    parser.add_argument("--target-mode", choices=("label-only", "distilled"), required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--public", action="store_true", help="Make the Hub repository public; private is the default")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.adapter.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {args.adapter}")
    if not (args.adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No adapter_config.json found in {args.adapter}")
    if args.max_seq_length < 128:
        raise ValueError("max-seq-length must be at least 128")
    if "/" not in args.repo_id or any(part in args.repo_id for part in ("..", " ")):
        raise ValueError("repo-id must use the owner/name format")

    try:
        from huggingface_hub import HfApi
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError("Install the dependencies from training/requirements.txt before publishing.") from exc

    token = _read_token()
    api = HfApi(token=token)
    api.whoami()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )

    print(f"Loading LoRA adapter from {args.adapter} for 16-bit merge...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Merging adapter and uploading to https://huggingface.co/{args.repo_id} ...")
    model.push_to_hub_merged(
        args.repo_id,
        tokenizer,
        save_method="merged_16bit",
        token=token,
        private=not args.public,
    )

    card = _model_card(args.base_model, args.target_mode, args.repo_id)
    with tempfile.TemporaryDirectory(prefix="vi-extoxic-model-card-") as temp_dir:
        card_path = Path(temp_dir) / "README.md"
        card_path.write_text(card, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add Vi-ExToxic model card",
        )
    print(f"Published merged model: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
