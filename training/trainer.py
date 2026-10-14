"""GPU training adapter backed by Unsloth and TRL."""

from __future__ import annotations

import inspect
import json
import logging
import hashlib
import importlib.metadata
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import SFTConfig
from .data import read_examples
from .prompts import PROMPT_VERSION, to_prompt_completion

LOGGER = logging.getLogger(__name__)


def _last_checkpoint(output_dir: Path) -> str | None:
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[-1])))


def _supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pass only args supported by the installed TRL version."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _accepts_var_kwargs(callable_obj: Any) -> bool:
    try:
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in inspect.signature(callable_obj).parameters.values()
        )
    except (TypeError, ValueError):
        return False


def train(config: SFTConfig) -> Path:
    """Run QLoRA SFT and save a reloadable adapter plus training metadata."""
    config.validate()

    try:
        import torch
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTConfig as TRLSFTConfig
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install training/requirements.txt in a supported "
            "CUDA environment (or use the Kaggle GPU setup in the README)."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Unsloth SFT requires a CUDA GPU. Enable a Kaggle GPU or run Docker with --gpus all.")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    examples = read_examples(config.data, config.zip_member, config.target_mode)
    records = [to_prompt_completion(row, config.target_mode) for row in examples]
    dataset = Dataset.from_list(records)
    LOGGER.info("Loaded %d rows from %s (%s)", len(records), config.data, config.target_mode)

    supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    compute_capability_major, _ = torch.cuda.get_device_capability()
    # Avoid FP16 GradScaler on pre-Ampere cards: this Unsloth QLoRA path can
    # produce BF16 adapter gradients, which the FP16 scaler cannot unscale.
    use_fp16_amp = not supports_bf16 and compute_capability_major >= 8
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=compute_dtype,
        load_in_4bit=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    FastLanguageModel.for_training(model)

    bf16 = supports_bf16
    training_args = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": config.seed,
        "data_seed": config.seed,
        "bf16": bf16,
        "fp16": use_fp16_amp,
        "report_to": "none",
        "completion_only_loss": True,
        "dataset_num_proc": 2,
        "packing": False,
    }
    LOGGER.info(
        "Using model compute dtype %s with trainer AMP settings bf16=%s, fp16=%s on compute capability %s.%s",
        compute_dtype,
        bf16,
        use_fp16_amp,
        compute_capability_major,
        torch.cuda.get_device_capability()[1],
    )
    config_parameters = inspect.signature(TRLSFTConfig).parameters
    if "max_length" in config_parameters or _accepts_var_kwargs(TRLSFTConfig):
        training_args["max_length"] = config.max_seq_length
    elif "max_seq_length" in config_parameters:
        training_args["max_seq_length"] = config.max_seq_length
    training_args = _supported_kwargs(TRLSFTConfig, training_args)
    args = TRLSFTConfig(**training_args)
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": args,
    }
    trainer_parameters = inspect.signature(SFTTrainer).parameters
    if "processing_class" in trainer_parameters or _accepts_var_kwargs(SFTTrainer):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if "max_seq_length" in trainer_parameters:
        trainer_kwargs["max_seq_length"] = config.max_seq_length
    trainer = SFTTrainer(**_supported_kwargs(SFTTrainer, trainer_kwargs))

    resume = config.resume_from_checkpoint
    if resume == "latest":
        resume = _last_checkpoint(config.output_dir)
        if resume is None:
            raise FileNotFoundError(f"No checkpoint-* directory found in {config.output_dir}")
    elif resume and not Path(resume).is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {resume}")

    train_result = trainer.train(resume_from_checkpoint=resume)
    adapter_dir = config.output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    trainer.save_model(str(adapter_dir))
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    metadata = asdict(config)
    metadata["data"] = str(config.data)
    metadata["output_dir"] = str(config.output_dir)
    metadata["resume_from_checkpoint"] = resume
    metadata["training_rows"] = len(examples)
    metadata["prompt_language"] = "en" if config.target_mode == "cot-en" else "vi"
    metadata["prompt_version"] = PROMPT_VERSION
    metadata["data_sha256"] = hashlib.sha256(config.data.read_bytes()).hexdigest()
    metadata["software_versions"] = {}
    for package in ("torch", "unsloth", "trl", "transformers", "datasets", "accelerate"):
        try:
            metadata["software_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["software_versions"][package] = "unknown"
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    (config.output_dir / "run_metadata.json").write_text(metadata_json, encoding="utf-8")
    (adapter_dir / "run_metadata.json").write_text(metadata_json, encoding="utf-8")
    LOGGER.info("Saved adapter and run metadata to %s", adapter_dir)
    return adapter_dir
