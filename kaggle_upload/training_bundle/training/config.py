"""Training configuration and validation, independent of CUDA imports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SFTConfig:
    data: Path
    zip_member: str | None
    output_dir: Path
    model_name: str
    target_mode: str
    max_seq_length: int
    epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    save_total_limit: int
    lora_rank: int
    lora_alpha: int
    seed: int
    max_steps: int
    resume_from_checkpoint: str | None

    def validate(self) -> None:
        if self.target_mode not in {"label-only", "distilled"}:
            raise ValueError("target_mode must be 'label-only' or 'distilled'")
        if self.max_seq_length < 128:
            raise ValueError("max_seq_length must be at least 128")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if min(self.logging_steps, self.save_steps, self.save_total_limit, self.lora_rank, self.lora_alpha) <= 0:
            raise ValueError("logging/save intervals, checkpoint limit and LoRA dimensions must be positive")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 (epoch-based) or a positive step limit")
